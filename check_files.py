import os
import numpy as np

DATA_PATH = '/home/natasha/unet4/axial_data'

dir_mode_lst = ['train', 'val', 'test']

for dir_mode in dir_mode_lst:
    dir_in_lst = os.listdir(os.path.join(DATA_PATH, dir_mode))
    for dir_in in dir_in_lst:
        pics = os.listdir(os.path.join(DATA_PATH, dir_mode, dir_in, 'pictures'))
        masks = os.listdir(os.path.join(DATA_PATH, dir_mode, dir_in, 'masks'))
        for i, pic in enumerate(pics):
            pics[i] = pic[:-4]
        for i, msk in enumerate(masks):
            mask_value = np.load(os.path.join(DATA_PATH, dir_mode, dir_in, 'masks', msk))
            if len(np.unique(mask_value)) == 1:
                print(f'mode = {dir_mode}, folder = {dir_in}, mask contains only 0 = {msk}')
            masks[i] = msk[:-4]
        pic_set = set(pics)
        mask_set = set(masks)
        absent_files = pic_set.difference(mask_set)
        print(f'mode = {dir_mode}, folder = {dir_in}, # pics = {len(pics)}, # masks = {len(masks)}, \
        len diff = {len(pics) - len(masks)} = {len(absent_files)}')
        print(f'Not present masks = {absent_files}\n')
        if len(pic_set.difference(mask_set)) != 0:
            for file_to_remove in absent_files:
                file_to_remove = file_to_remove + '.PNG'
                print(f'going to remove {file_to_remove}')
                path = os.path.join(DATA_PATH, dir_mode, dir_in, 'pictures', file_to_remove)
                os.remove(path)
                print(f'{file_to_remove} removed from {dir_in}\n')





