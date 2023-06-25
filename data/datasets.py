import glob
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from data.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox, right_cropping_bbox)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', 'tif', 'TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

std = np.array([0.5, 0.5, 0.5])
mean = np.array([0.5, 0.5, 0.5])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].add_(mean[c]).mul_(std[c])
    return torch.clamp(tensors, 0, 255)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


class UncroppingDatasetRS_train(torch.utils.data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((512, 512)),
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.vflip = transforms.RandomVerticalFlip(p=1)
        self.hflip = transforms.RandomHorizontalFlip(p=1)
        
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = "hybrid"
        # self.mask_mode = "fixed"
        self.image_size = image_size

    def __getitem__(self, index):
        vflip_flag = False
        hflip_flag = False
        ret = {}
        path = self.imgs[index]
        ref_path = path.replace('source', 'ref')
        img = self.tfs(self.loader(path))
        ref_image = self.tfs(self.loader(ref_path))
        
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.image_size)
        img = TF.crop(img, i, j, h, w)
        ref_image = TF.crop(ref_image, i, j, h, w)
        
        if random.random() > 0.5:
            img = self.vflip(img)
            vflip_flag = True
            if random.random() > 0.5:
                img = self.hflip(img)
                hflip_flag = True
            
        mask = self.get_mask()

        
        if vflip_flag:
            ref_image = self.vflip(ref_image)
            if hflip_flag:
                ref_image = self.hflip(ref_image)

        mask_img = img * (1. - mask)
        cond_image = torch.concat([mask, ref_image, mask_img], dim=0)
        # cond_image = torch.concat([ref_image, mask_img], dim=0)

        ret['hr'] = img.float()
        ret['lr'] = mask_img.float()
        ret['clip'] = cond_image.float()
        ret['alpha'] = mask.float()
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['ref'] = ref_image.float()
        
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass        
        elif self.mask_mode == 'fixed':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDatasetRS_test(torch.utils.data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = "fixed"
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        ref_path = path.replace('source', 'ref')
        img = self.tfs(self.loader(path))
        ref_image = self.tfs(self.loader(ref_path))
        mask = self.get_mask()
        mask_img = img * (1. - mask)
        cond_image = torch.concat([mask, ref_image, mask_img], dim=0)
        # cond_image = torch.concat([ref_image, mask_img], dim=0)

        ret['hr'] = img
        ret['lr'] = mask_img
        ret['clip'] = cond_image
        ret['alpha'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['ref'] = ref_image
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        mask = bbox2mask(self.image_size, right_cropping_bbox())
        return torch.from_numpy(mask).permute(2,0,1)
