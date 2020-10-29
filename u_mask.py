import argparse
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import cv2

import matplotlib.pyplot as plt

import random

def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
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
                #transforms.Resize(full_img.width),#size[1]),
                transforms.Resize((full_img.height, full_img.width)),
                transforms.ToTensor()
            ]
        )

        probs_0 = tf(probs_0.cpu())
        probs_1 = tf(probs_1.cpu())
        mask_0 = probs_0.squeeze().cpu().numpy()
        mask_1 = probs_1.squeeze().cpu().numpy()
        full_mask = np.array([mask_0, mask_1])#probs.squeeze().cpu().numpy()

    return full_mask# > out_threshold


scale = 0.5
mask_threshold = 0.5 #минимальная вероятность для рассматривания пикселя на маску

no_save = True
viz = False


model = 'ckpts_dir/mid_ckpts/CP_epoch700.pth'

# /CP_epoch200.pth - класс!

net = UNet(n_channels=1, n_classes=2)

logging.info("Loading model {}".format(model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
net.to(device=device)
net.load_state_dict(torch.load(model, map_location=device))

logging.info("Model loaded !")

lst = []



####
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import disc

#####

TEST_MODE = "inference"

dataset = disc.DiscDataset()
dataset.load_disc(custom_DIR, "test")

# Must call before using the dataset
dataset.prepare()


for f in glob.glob("/home/vlad/Mask_RCNN/res/*_dif.png"):
    rst = f.split('_')
    image_id = rst[2]
    #image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    #modellib.load_image_gt(dataset, config, image_id)
   
    # f'{rst[2]}_dif.png'

    #my_file = '/home/vlad/Mask_RCNN/dataset/test/T2_Sag_Mid1/00010.PNG'
    img = Image.open(dataset.image_reference(image_id))

    mask = predict_img(net=net,
                   full_img=img,
                   out_threshold=mask_threshold,
                   device=device)

    res_treshold = 0.7

    mask = np.array([mask[0] > res_treshold, mask[1] > res_treshold], dtype=np.uint8)

    #my_img='/home/vlad/Mask_RCNN/00010_342_dif.png'
    #my_img = '/home/vlad/Mask_RCNN/dataset/test/T2_Sag_Mid1/00010.PNG'
    my_img = glob.glob(f"/home/vlad/Mask_RCNN/res/*_{image_id}_dif.png")
    d_img = Image.open(my_img)
    dif_img = np.array(d_img)

    vertebrae = cv2.Canny(image=mask[0]*255, threshold1=50, threshold2=150)
    canal = cv2.Canny(image=mask[1]*255, threshold1=50, threshold2=150)
    #vertebrae = mask[0]*255
    #canal = mask[1]*255
    rgb_contours = np.dstack([np.zeros_like(canal), np.zeros_like(canal), canal])
    rgb_contours2 = np.dstack([np.zeros_like(canal), np.zeros_like(canal), vertebrae])
    #img_np = np.array(img)
    #rgb_mrimg = np.dstack([img, img, img])


    #rgb_mrimg = resize(rgb_mrimg, (dif_img.shape[0], dif_img.shape[1], rgb_mrimg.shape[2]))
    #rgb_contours = resize(rgb_contours, (dif_img.shape[0], dif_img.shape[1], rgb_mrimg.shape[2]))

    #dif_img = np.resize(rgb_contours, (dif_img.shape[0], dif_img.shape[1], rgb_mrimg.shape[2]))

    #rgb_mrimg = cv2.resize(rgb_mrimg, dsize=(dif_img.shape[0], dif_img.shape[1]), interpolation=cv2.INTER_CUBIC)

    rgb_contours = cv2.resize(rgb_contours, dsize=(dif_img.shape[1], dif_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    rgb_contours2 = cv2.resize(rgb_contours2, dsize=(dif_img.shape[1], dif_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    rgb_contours = rgb_contours+rgb_contours2

    print('dif',dif_img.shape)
    #print('rgb',rgb_mrimg.shape)
    print('rgb_contours', rgb_contours.shape)


    #result_img = cv2.addWeighted(rgb_mrimg, 1, rgb_contours, 0.5, 0)
    result_img = cv2.addWeighted(dif_img[...,:3], 1, rgb_contours, 0.5, 0)

    #rgb_mrimg = np.resize(rgb_mrimg, (dif_img.shape[0], dif_img.shape[1], dif_img.shape[2]))
    print('dif',dif_img.shape)
    #print('rgb',rgb_mrimg.shape)
    print('rgb_contours', rgb_contours.shape)
    #save_path = os.path.abspath(os.path.join('tmp', str(uuid4()) + '.png'))
    cv2.imwrite(f'/home/vlad/res_u/{image_id}_rcnn_unet.png', result_img)
    #mask_path = save_path[:-4] + '_mask.png'
    #cv2.imwrite('/home/vlad/mask.png', rgb_contours)

