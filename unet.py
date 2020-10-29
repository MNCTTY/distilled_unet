import statistics as st
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import cv2

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        print(output.shape)
        #if net.n_classes > 1:
        #    probs = F.softmax(output, dim=1)
        #else:
        #    probs = torch.sigmoid(output)
        probs_0 = torch.sigmoid(output[:, 0, :, :])
        probs_1 = torch.sigmoid(output[:, 1, :, :])

        probs_0 = probs_0.squeeze(0)
        probs_1 = probs_1.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.width),#size[1]),
                transforms.ToTensor()
            ]
        )

        probs_0 = tf(probs_0.cpu())
        probs_1 = tf(probs_1.cpu())
        mask_0 = probs_0.squeeze().cpu().numpy()
        mask_1 = probs_1.squeeze().cpu().numpy()
        full_mask = np.array([mask_0, mask_1])#probs.squeeze().cpu().numpy()

    return full_mask# > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


model_name = 'sag_mid'
# sag_mid
# sag_mid_and_new
# sag_mid_and_new_and_elastic

model = '/home/karina/__migrated/unet4/ckpts_dir/mid_ckpts/CP_epoch700.pth'
# ckpts_dir/mid_ckpts/
# ckpts_dir/mid_ckpts_with_new_mid_data/
# ckpts_aug_dir/mid_elast_ckpts/

net = UNet(n_channels=1, n_classes=2)

logging.info("Loading model {}".format(model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
net.to(device=device)
net.load_state_dict(torch.load(model, map_location=device))

logging.info("Model loaded !")

scale = 0.5
mask_threshold = 0.5 #минимальная вероятность для рассматривания пикселя на маску

no_save = True
viz = False

list_of_dirs = [
    '/home/karina/__migrated/T2_Sag_Mid/test_data/'
]

img_list = os.listdir(list_of_dirs[0]+'imgs/') 
masks_list = os.listdir(list_of_dirs[0]+'masks/') 

img_src_pathes = []
for img in img_list:
    img_src_pathes.append(list_of_dirs[0]+'imgs/'+img)
    
mask_src_pathes = []
for mask in masks_list:
    mask_src_pathes.append(list_of_dirs[0]+'masks/'+mask)

iou_discs = []
iou_canal = []
iou_general = []
f1_discs = []
f1_canal = []
f1_general = []

bad_iou = []
bad_results = []

for i in range(0, len(img_src_pathes)):
    imgname = img_src_pathes[i]
    maskname = mask_src_pathes[i]
    
    img = Image.open(imgname)
    true_mask = cv2.imread(maskname)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=scale,
                       out_threshold=mask_threshold,
                       device=device)
    ########
    ### change the threshold param for better / worse results
    print('MASK', mask)
    res_treshold = 0.8
    mask = np.array([mask[0] > res_treshold, mask[1] > res_treshold])

    ########

    
    d, w, h = mask.shape
    
    try:
    
        # dics

    #     true_mask2 = cv2.resize(true_mask[:,:,0], (324, 320))
    #     true_mask2 = true_mask[:,:,0]
        true_mask2 = cv2.resize(true_mask[:,:,0], (h, w))

        pixelThreshold = 0.5
        bin_mask = np.where(mask[0] > pixelThreshold, 1, 0)

        unique, counts = np.unique(bin_mask + true_mask2, return_counts=True)
        inters = counts[-1]
        union = counts[-1]+counts[-2]

        iou = inters/union

        iou_discs.append(iou)
        iou_general.append(iou)

        if iou<0.5:
            bad_iou.append([iou, imgname])

    #     squares = 2*h*w
        un, cnts = np.unique(bin_mask, return_counts=True)
        un, cnts2 = np.unique(true_mask2, return_counts=True)
        squares = cnts[1] + cnts2[1]

        f1 = 2 * inters / squares
        f1_discs.append(f1)
        f1_general.append(f1)



        # canal

    #     true_mask2 = cv2.resize(true_mask[:,:,1], (324, 320))
    #     true_mask2 = true_mask[:,:,1]
        true_mask2 = cv2.resize(true_mask[:,:,1], (h, w))


        pixelThreshold = 0.5
        bin_mask = np.where(mask[1] > pixelThreshold, 1, 0)

        unique, counts = np.unique(bin_mask + true_mask2, return_counts=True)
        inters = counts[-1]
        union = counts[-1]+counts[-2]

        iou = inters/union

        iou_canal.append(iou)
        iou_general.append(iou)

        if iou<0.5:
            bad_iou.append([iou, imgname])

    #     squares = 2*h*w
        un, cnts = np.unique(bin_mask, return_counts=True)
        un, cnts2 = np.unique(true_mask2, return_counts=True)
        squares = cnts[1] + cnts2[1]

        f1 = 2 * inters / squares
        f1_canal.append(f1)
        f1_general.append(f1)
    except:
        print('Упс, ошибочка вышла(')
        bad_results.append([imgname, maskname, i])



