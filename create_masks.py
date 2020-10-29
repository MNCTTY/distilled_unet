import numpy as np
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt 
import cv2
from jsonmerge import merge
from tifffile import imsave
import os
from os.path import splitext
from glob import glob

# train
# folder_lst = ['Fifth_3',
#               'First_10',
#               'Forth_6',
#               'Second_0',
#               'Second_2',
#               'Second_4',
#               'Sixth_2',
#               'Sixth_3']

# test
# folder_lst = ['Fifth_5',
#               'Sixth_1']

# val
folder_lst = ['Fifth_2',
              'Forth_3',
              'Second_7',
              'Second_9',
              'Sixth_9']


for folder_name in folder_lst:
    dir_ = '/home/natasha/unet4/axial_data/val/' + folder_name + '/'
    json1 = 'T2Tra_' + folder_name + '_s.json'
    json2 = 'T2Tra_' + folder_name + '_h.json'

    with open(dir_ + json1) as file:
        j1 = json.load(file)

    with open(dir_ + json2) as file:
        j2 = json.load(file)

    raw_data = merge(j1, j2)
    all_keys = []

    for key in raw_data['_via_img_metadata'].keys():
        all_keys.append(key)

    pics_path = dir_ + 'pictures/'
    masks_path = dir_ + 'masks/'

    if not os.path.exists(masks_path):
        os.makedirs(masks_path)

    fucked_imgs = []
    zero_files = []
    not_found = []
    for k in range(0, len(all_keys)):
        filename = raw_data['_via_img_metadata'][all_keys[k]]['filename']

        try:
            image = Image.open(pics_path + filename)
            width = image.size[0]
            height = image.size[1]

            img1 = Image.new('L',(width, height), 0)
            img2 = Image.new('L',(width, height), 0)
            img3 = Image.new('L',(width, height), 0)
            img4 = Image.new('L',(width, height), 0)
            img5 = Image.new('L',(width, height), 0)
            img6 = Image.new('L',(width, height), 0)
            for i in range(len(raw_data['_via_img_metadata'][all_keys[k]]['regions'])):
                try:
                    xs = raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['shape_attributes']['all_points_x']
                    ys = raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['shape_attributes']['all_points_y']
                    all_pairs = []
                    for j in range(len(ys)):
                        all_pairs.append((xs[j], ys[j]))

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'disc':
                        ImageDraw.Draw(img1).polygon(all_pairs, outline=1, fill=1)

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'spinal canal':
                        ImageDraw.Draw(img2).polygon(all_pairs, outline=1, fill=1)

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'nerv':
                        ImageDraw.Draw(img3).polygon(all_pairs, outline=1, fill=1)

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'neural foramen':
                        ImageDraw.Draw(img4).polygon(all_pairs, outline=1, fill=1)

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'size of spinal canal':
                        ImageDraw.Draw(img5).polygon(all_pairs, outline=1, fill=1)

                    if raw_data['_via_img_metadata'][all_keys[k]]['regions'][i]['region_attributes']['A3'] == 'size of herniation':
                        ImageDraw.Draw(img6).polygon(all_pairs, outline=1, fill=1)


                except:
                    fucked_imgs.append(filename)

            mask1 = np.array(img1)
            mask2 = np.array(img2)
            mask3 = np.array(img3)
            mask4 = np.array(img4)
            mask5 = np.array(img5)
            mask6 = np.array(img6)

#             mask = mask1 + 2 * mask2 + 3 * mask3 + 4 * mask4 + 5 * mask5 + 6 * mask6
            mask = np.dstack((mask1, mask2, mask3, mask4, mask5, mask6))
            if len(np.unique(mask)) == 1:
                zero_files.append(filename)


        except:
            not_found.append(filename)


        if filename not in not_found:
            if filename not in fucked_imgs:
                if filename not in zero_files:
                    if len(np.unique(mask)) > 1:
                        filename = filename[:-3] + 'npy'
#                         print(filename)
#                         print(np.unique(mask))
                        np.save(masks_path + filename, mask)
