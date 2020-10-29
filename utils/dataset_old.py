from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

###### unified size of imgs/masks is 400x400


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
#         print('masks dir: ')
#         print(masks_dir)
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mask=False):
        w, h = pil_img.size
#         print(w, h)
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        if mask:
            img_nd = np.array([img_nd[:, :, 2], img_nd[:, :, 1]])

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        if not mask:
            img_trans = img_nd.transpose((2, 0, 1))
        else:
            img_trans = img_nd

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        # create new image of desired size for padding
        ww = 550
        hh = 550

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        
        wd_m, ht_m = mask.size

        # color = (255 ,0, 0)
        result_mask = np.full((hh,ww,3), 0, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd_m) // 2
        yy = (hh - ht_m) // 2

        # copy img image into center of result image
        try:
            result_mask[yy:yy+ht_m, xx:xx+wd_m] = mask
        except:
#             with open('bad_files.txt', 'a') as f:
#                 f.write('Img file: \n')
#                 f.write(mask_file[0])
#                 f.write('\n')
            print(mask_file[0])
            print('error in result_mask[yy:yy+ht_m, xx:xx+wd_m] = mask!!!!')
#         print(mask)
#         print(result_mask)
        
        
        img = Image.open(img_file[0]).convert('L')
        wd_im, ht_im = img.size
        
        result_img = np.full((hh,ww), 0, dtype=np.uint8)
        xx = (ww - wd_im) // 2
        yy = (hh - ht_im) // 2
        try:
            result_img[yy:yy+ht_im, xx:xx+wd_im] = img
        except:
#             with open('bad_files.txt', 'a') as f:
#                 f.write('Mask file: \n')
#                 f.write(mask_file[0])
#                 f.write('\n')
            print(img_file[0])
            print('error in result_img[yy:yy+ht_im, xx:xx+wd_im] = img!!!!')
        
#         print(img)
#         print(result_img)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        result_img = Image.fromarray(np.uint8(result_img))
        result_mask = Image.fromarray(np.uint8(result_mask))
        
        result_img = self.preprocess(result_img, self.scale)
        result_mask = self.preprocess(result_mask, self.scale, mask)

        return {
            'image': torch.from_numpy(result_img).type(torch.FloatTensor),
            'mask': torch.from_numpy(result_mask).type(torch.FloatTensor)
        }
    
#         return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

