import argparse
from datetime import datetime
import os
import time
import torch
import numpy as np
import wandb
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.datasets import *
from model.models import *
from model.triplet_loss import TripletLoss
from util.seed_everything import seed_everything
from test import eval


seed = 42
# 42, 1126, 2000, 3407, 31415
seed_everything(seed)

print(f"global seed set to {seed}")

os.environ["WANDB_MODE"] = "offline"
t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--train_dataset_name", type=str, default="/data/cyc/dataset/train.flist", help="name of the dataset")
parser.add_argument("--test_dataset_name", type=str, default=None, help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=5e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-3, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="batch interval between model checkpoints")
parser.add_argument("--lambda_adv", type=float, default=1e-2, help="adversarial loss weight")
parser.add_argument("--lambda_tri", type=float, default=1e-1, help="triplet loss weight")
parser.add_argument("--alpha", type=float, default=0.3, help="triplet loss margin")
parser.add_argument("--save_images", default=f'{seed}/images', help="where to store images")
parser.add_argument("--save_models", default=f'{seed}/saved_models', help="where to save models")
parser.add_argument("--output", default=f'{seed}/output', help="where to save output")
parser.add_argument("--gpu", type=int, default=1, help="gpu number")
opt = parser.parse_args()
print(opt)
if not os.path.exists(str(seed)):
    os.makedirs(str(seed))
wandb.init(project="ExtendGAN", config=vars(opt), dir=f"{seed}", tags=[str(seed), "Extend"])
wandb.run.log_code(".")
os.makedirs(opt.save_images, exist_ok=True)
os.makedirs(opt.save_models, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

hr_shape = opt.hr_shape

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator256().to(device)


# torch.set_float32_matmul_precision('high')
# generator = torch.compile(generator)
# # discriminator = torch.compile(discriminator)

# Losses
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
criterion_triplet = TripletLoss(opt.alpha).to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(f"{seed}/saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load(f"{seed}/saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_G, T_max=opt.n_epochs, eta_min=1e-6)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

dataloader = DataLoader(
    UncroppingDatasetRS_train("%s" % opt.train_dataset_name),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    generator=g,
    worker_init_fn=seed_worker,
    # pin_memory=True,
)

test_dataloader = None
if opt.test_dataset_name is not None:
    test_dataloader = DataLoader(
        UncroppingDatasetRS_test("%s" % opt.test_dataset_name),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # pin_memory=True,
    )

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    D_loss = 0
    G_loss = 0
    content = 0
    adv = 0
    pixel = 0
    triplet = 0
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"].to(device)
        img_alpha = imgs["alpha"].to(device)
        clip = imgs['clip'].to(device)
        ref = imgs['ref'].to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        gen_hr = generator(clip)
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        gen_hr_d = gen_hr * img_alpha + imgs_lr
        pred_fake = discriminator(torch.cat([gen_hr_d, img_alpha, ref], dim=1))
        loss_GAN = -pred_fake.mean()
        loss_triplet, ssim_pred_src, ssim_pred_ref = criterion_triplet(gen_hr_d, imgs_hr, ref, img_alpha)
        loss_G = opt.lambda_adv * loss_GAN + loss_pixel + loss_triplet * opt.lambda_tri

        loss_G.backward()
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        pred_real = discriminator(torch.cat([imgs_hr, img_alpha, ref], dim=1))
        pred_fake = discriminator(torch.cat([gen_hr_d.detach(), img_alpha, ref], dim=1))
        loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        D_loss += loss_D.item()
        G_loss += loss_G.item()
        adv += loss_GAN.item()
        pixel += loss_pixel.item()
        triplet += loss_triplet.item()
        
    avg_D_loss = D_loss / len(dataloader)
    avg_G_loss = G_loss / len(dataloader)
    avg_adv_loss = adv / len(dataloader)
    avg_pixel_loss = pixel / len(dataloader)
    avg_triplet_loss = triplet / len(dataloader)

    print(
        'Epoch:{1}/{2} lr:{7} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} tri:{8} time:{0}'.format(
            datetime.now() - t_start, epoch + 1, opt.n_epochs, avg_D_loss,
            avg_G_loss, avg_adv_loss, avg_pixel_loss, scheduler.get_last_lr()[0], avg_triplet_loss))
    
    scheduler.step()
    
    wandb.log({"lr": scheduler.get_last_lr()[0], "epoch": epoch + 1, "D_loss": avg_D_loss, "G_loss": avg_G_loss, "adv": avg_adv_loss, "pixel":avg_pixel_loss, "tri": avg_triplet_loss})
    
    if (epoch + 1) % opt.sample_interval == 0:
        # Save example results
        img_grid = denormalize(torch.cat((imgs_lr, gen_hr, gen_hr_d, imgs_hr), -1))
        save_image(img_grid, opt.save_images + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)
        # wim = wandb.Image(opt.save_images + "/epoch-{}.png".format(epoch + 1), caption=f"sample{epoch + 1}")
        # wandb.log({"sample": wim})
    
    if (epoch + 1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), opt.save_models + "/generator_{}.pth".format(epoch + 1))

        if test_dataloader is not None:
            generator.eval()
            with torch.no_grad():
                if not os.path.exists(os.path.join(opt.output, str(epoch + 1))):
                    os.makedirs(os.path.join(opt.output, str(epoch + 1)))
                s = time.time()
                for i, imgs in enumerate(test_dataloader):
                    imgs_lr = imgs["lr"].to(device)
                    imgs_hr = imgs["hr"].to(device)
                    img_alpha = imgs["alpha"].to(device)
                    clip = imgs['clip'].to(device)
                    ref = imgs['ref'].to(device)

                    gen = generator(clip)
                    gen_f = gen * img_alpha + imgs_hr * (1 - img_alpha) 
                    save_image((gen_f + 1) * 0.5, os.path.join(opt.output, str(epoch + 1), f"Out_{i}.jpg"))
                    save_image(img_alpha.float(), os.path.join(opt.output, str(epoch + 1), f"Mask_{i}.jpg"))
                    save_image((ref + 1) * 0.5, os.path.join(opt.output, str(epoch + 1), f"Ref_{i}.jpg"))
                    save_image((imgs_hr + 1) * 0.5, os.path.join(opt.output, str(epoch + 1), f"GT_{i}.jpg"))
                    print(f'Output generated image gen_{i}')
                e = time.time()
                print(f'total {e - s}s')
                print(f'avg {(e - s) / len(test_dataloader)}s')
            log_dict = eval(os.path.join(opt.output, str(epoch + 1)))
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict)
            generator.train()
            