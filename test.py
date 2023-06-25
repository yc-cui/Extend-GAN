import argparse
import os
import time
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, mean_squared_error as compare_mse, structural_similarity as compare_ssim
import scipy.stats as stats

from pytorch_msssim import ssim as compare_ssim1, ms_ssim as compare_ms_ssim
import torch

from glob import glob
from PIL import Image
import pylab
import numpy as np


import torch
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data.datasets import UncroppingDatasetRS_test

from model.models import *


def eval(exp):
    gt_format = os.path.join(exp, "GT_*.jpg")
    pred_format = os.path.join(exp, "Out_*.jpg")
    ref_format = os.path.join(exp, "Ref_*.jpg")
    mask_format = os.path.join(exp, "Mask_*.jpg")
    
    gt = glob(gt_format)
    pred = glob(pred_format)
    assert len(gt) == len(pred)
    print(f"{exp}: {len(gt)} images")
    
    def extract_no(full_path):
        return os.path.basename(full_path)[-4]
    
    def comapre_mae(image0, image1):
        return np.mean(np.absolute(image0 - image1))
    
    gt.sort(key=lambda x: extract_no(x))
    pred.sort(key=lambda x: extract_no(x))
    
    
    psnr_list = []
    ssim_list = []
    mae_list = []
    mse_list =[]
    rmse_list = []
    cc_list = []

    for i in range(len(gt)):
        
        assert extract_no(gt[i]) == extract_no(pred[i])
        
        img_gt = pylab.imread(gt[i]) / 255
        img_pred = pylab.imread(pred[i]) / 255
        
        psnr = compare_psnr(img_gt, img_pred, data_range=1.)
        psnr_list.append(psnr)
        
        ssim = compare_ssim(img_gt, img_pred, win_size=11, data_range=1., multichannel=True, channel_axis=2)
        ssim_list.append(ssim)
        
        mae = comapre_mae(img_gt, img_pred)
        mae_list.append(mae)
        
        mse = compare_mse(img_gt, img_pred)
        mse_list.append(mse)
        rmse_list.append(np.sqrt(mse))
        
        corr, _ = stats.pearsonr(img_gt.flatten(), img_pred.flatten())
        cc_list.append(corr)
    
    
    ssim_list_1 = []
    ms_ssim_list = []
    with torch.no_grad():
        for i in range(len(gt)):
            assert extract_no(gt[i]) == extract_no(pred[i])

            img_gt = Image.open(gt[i]).convert("RGB")
            img_pred = Image.open(pred[i]).convert("RGB")

            img_gt = torch.from_numpy(np.array(img_gt).transpose(2, 0, 1)).unsqueeze(0).float()
            img_pred = torch.from_numpy(np.array(img_pred).transpose(2, 0, 1)).unsqueeze(0).float()

            ssim = compare_ssim1(img_gt, img_pred).numpy()
            ms_ssim = compare_ms_ssim(img_gt, img_pred).numpy()

            ssim_list_1.append(ssim)
            ms_ssim_list.append(ms_ssim)    
            
            
    print(f"PSNR: {np.mean(psnr_list)}")
    print(f"SSIM: {np.mean(ssim_list)}, {np.mean(ssim_list_1)}")
    print(f"MS_SSIM: {np.mean(ms_ssim_list)}")
    print(f"MAE: {np.mean(mae_list)}")
    print(f"MSE: {np.mean(mse_list)}")
    print(f"RMSE: {np.mean(rmse_list)}")
    print(f"CC: {np.mean(cc_list)}")
    
    ref = glob(ref_format)
    mask = glob(mask_format)
    assert len(gt) == len(pred) == len(ref) == len(mask)

    ref.sort(key=lambda x: extract_no(x))
    mask.sort(key=lambda x: extract_no(x))

    ssim_pred_gt_list = []
    ssim_pred_ref_list = []

    with torch.no_grad():
        cnt = 0
        for i in range(len(gt)):
            img_gt_ = Image.open(gt[i]).convert("RGB")
            img_pred_ = Image.open(pred[i]).convert("RGB")
            img_ref_ = Image.open(ref[i]).convert("RGB")
            img_mask_ = np.array(Image.open(mask[i]).convert("L"))

            img_gt = torch.from_numpy(np.array(img_gt_).transpose(2, 0, 1)).unsqueeze(0).float()
            img_pred = torch.from_numpy(np.array(img_pred_).transpose(2, 0, 1)).unsqueeze(0).float()
            img_ref = torch.from_numpy(np.array(img_ref_).transpose(2, 0, 1)).unsqueeze(0).float()
            img_mask = torch.from_numpy(img_mask_).bool().float().unsqueeze(0).unsqueeze(0).float()
            
            ssim1 = compare_ssim1(img_gt * img_mask, img_pred * img_mask).numpy()
            ssim2 = compare_ssim1(img_ref * img_mask, img_pred * img_mask).numpy()
            
            ssim_pred_gt_list.append(ssim1.item())
            ssim_pred_ref_list.append(ssim2.item())
            if ssim2 > ssim1:
                cnt += 1
    
    print(f"Ratio: {cnt / len(gt)}")

    log_dict = {
            "PSNR": np.mean(psnr_list), 
            "SSIM1": np.mean(ssim_list),
            "SSIM2": np.mean(ssim_list_1),
            "MS_SSIM": np.mean(ms_ssim_list),
            "MAE": np.mean(mae_list),
            "MSE": np.mean(mse_list),
            "RMSE": np.mean(rmse_list),
            "CC": np.mean(cc_list),
            "Ratio": cnt / len(gt),
    }
    
    return log_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='/data/cyc/dataset/test_crop_source.flist')
    parser.add_argument("--output", default='42/test', help="where to save output")
    parser.add_argument("--model", default="42/saved_models/generator_1000.pth", help="generator model pass")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="the ratio of pixels in the image are respectively masked")
    parser.add_argument("--image_size", type=int, default=256, help="test image size 256 or 257 or 512")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    opt = parser.parse_args()

    os.makedirs(opt.output, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device('cuda:{}'.format(opt.gpu))
    else:
        device = torch.device('cpu')
        
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    generator = Generator().to(device)

    generator.load_state_dict(torch.load(opt.model))
    generator.eval()

    dataloader = DataLoader(
        UncroppingDatasetRS_test("%s" % opt.image_path),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        # pin_memory=True,
    )


    # Calculate image
    with torch.no_grad():
        
        s = time.time()
        
        for i, imgs in enumerate(dataloader):
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            img_alpha = Variable(imgs["alpha"].type(Tensor))
            clip = Variable(imgs['clip'].type(Tensor))
            ref = Variable(imgs['ref'].type(Tensor))
            
            gen = generator(clip)
            gen_f = gen * img_alpha + imgs_hr * (1 - img_alpha)

            # save_image((img + 1) * 0.5, opt.output + f"/raw.png")
            # save_image((gen + 1) * 0.5, opt.output + f"/gen.png")
            save_image((gen_f + 1) * 0.5, os.path.join(opt.output, f"Out_{i}.jpg"))
            save_image(img_alpha, os.path.join(opt.output, f"Mask_{i}.jpg"))
            save_image((ref + 1) * 0.5, os.path.join(opt.output, f"Ref_{i}.jpg"))
            save_image((imgs_hr + 1) * 0.5, os.path.join(opt.output, f"GT_{i}.jpg"))
            print(f'Output generated image gen_{i}')

        e = time.time()
        
        print(f'total {e - s}s')
        print(f'avg {(e - s) / len(dataloader)}s')
        
        
        
