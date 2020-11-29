from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os

###### unified size of imgs/masks is 400x400


class BasicDataset(Dataset):
    def __init__(self, imgs_dir):
        folders = os.listdir(imgs_dir)
        mask_paths = [] 
        img_paths = []
        for i, folder in enumerate(folders):
            path = os.path.join(imgs_dir, folder, 'pictures')
            ids = [splitext(file)[0] for file in os.listdir(path) if not file.startswith('.')]
            
            for idx in ids:
                mask_paths.append(os.path.join(imgs_dir, folder, 'masks', str(idx)) + '.npy')
                img_paths.append(os.path.join(imgs_dir, folder, 'pictures', str(idx)) + '.PNG')
        
        self.mask_paths = mask_paths
        self.img_paths = img_paths

        
        logging.info(f'Creating dataset with {len(mask_paths)} examples')

    def __len__(self):
        return len(self.mask_paths)

    @classmethod
    def preprocess(cls, img_or_mask, mask=False):
        # HWC to CHW
        if not mask:
            img_or_mask = np.array(img_or_mask)
            img_or_mask = np.expand_dims(img_or_mask, axis=2)

        else:
            img_or_mask = img_or_mask
        img_or_mask = img_or_mask.transpose((2, 0, 1))
        if img_or_mask.max() > 1:
            img_or_mask = img_or_mask / 255

        return img_or_mask

    def __getitem__(self, i):

        mask_file = self.mask_paths[i]
        img_file = self.img_paths[i]

        mask = np.load(mask_file)
        img = Image.open(img_file).convert('L')
        
        # create new image of desired size for padding
        ww = 600
        hh = 600
        
        wd_m, ht_m, _ = mask.shape
        wd_im, ht_im = img.size
        
        # compute center offset
        xx = (ww - wd_m) // 2
        yy = (hh - ht_m) // 2

        result_mask = np.pad(mask, ((xx, xx),(yy, yy), (0, 0)))
        
        xx = (ww - wd_im) // 2
        yy = (hh - ht_im) // 2

        img = np.array(img)
        result_img = np.pad(img, ((yy, yy),(xx, xx)))
        result_img = Image.fromarray(np.uint8(result_img))

        result_img = self.preprocess(result_img)
        
        result_mask = self.preprocess(result_mask, mask=True)

        return {
            'image': torch.from_numpy(result_img),
            'mask': torch.from_numpy(result_mask).type(torch.long)
        }
    
