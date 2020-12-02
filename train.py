import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F 
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from IPython.display import clear_output

dir_train_img = '/home/natasha/unet/axial_data_new/train/'
dir_val_img = '/home/natasha/unet/axial_data_new/val/'
dir_checkpoint = 'ckpts_dir/axial_ckpts/'

img_scale = 1
n_channels=1 
n_classes=6 
bilinear=True


load = False
epochs = 150
batch_size = 1

warm_up = True
lr_param = 0.1
step_ratio = 0.1

temperature = 3
alpha = 0.1
beta = 1e-6


def kd_loss_function(output, target_output, temp):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()
        
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    
    return res
    
def adjust_learning_rate(warm_up, lr_param, step_ratio, optimizer, epoch):
    if warm_up and (epoch < 1):
        lr = 0.01
    elif 75 <= epoch < 130:
        lr = lr_param * (step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = lr_param * (step_ratio ** 2)
    elif epoch >=180:
        lr = lr_param * (step_ratio ** 3)
    else:
        lr = lr_param
    
    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
         
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.path.tar'))

        
def train_net(net,
              device,
              epochs,
              batch_size,
              lr=0.001,
              save_cp=True,
              epoch_bias=0):

    
    train_dataset = BasicDataset(dir_train_img)
    val_dataset = BasicDataset(dir_val_img)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

   
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    val_loss_score = 10
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
 
    
    optimizer = optim.Adam(net.module.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.module.n_classes > 2 else 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss()

    
    for epoch in range(epochs):
        net.module.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device).float()
                true_masks = true_masks.to(device=device).float()
                        
                masks_pred, soft_fin, btn1, btn2, btn3, btn4, soft_out1, soft_out2, soft_out3, soft_out4 = net.module(imgs)
            
                loss_original = criterion(masks_pred, true_masks)
                
                loss_orig_soft1 = criterion(soft_out1, true_masks)
                loss_orig_soft2 = criterion(soft_out2, true_masks)
                loss_orig_soft3 = criterion(soft_out3, true_masks)
                loss_orig_soft4 = criterion(soft_out4, true_masks)
                
                temp4 = masks_pred / temperature
                temp4 = torch.softmax(temp4, dim=1)
                
                
                loss1by6 = kd_loss_function(soft_out1, temp4.detach(), temperature) * (temperature**2)
                loss2by6 = kd_loss_function(soft_out2, temp4.detach(), temperature) * (temperature**2)
                loss3by6 = kd_loss_function(soft_out3, temp4.detach(), temperature) * (temperature**2)                
                loss4by6 = kd_loss_function(soft_out4, temp4.detach(), temperature) * (temperature**2)

                feature_loss_1 = feature_loss_function(btn1, masks_pred.detach()) 
                feature_loss_2 = feature_loss_function(btn2, masks_pred.detach()) 
                feature_loss_3 = feature_loss_function(btn3, masks_pred.detach()) 
                feature_loss_4 = feature_loss_function(btn4, masks_pred.detach()) 


                total_loss = (1 - alpha) * (loss_original + loss_orig_soft1 + loss_orig_soft2 + loss_orig_soft3 + loss_orig_soft4) + \
                            alpha * (loss1by6 + loss2by6 + loss3by6 + loss4by6) + \
                            beta * (feature_loss_1 + feature_loss_2 + feature_loss_3 + feature_loss_4)
   
                
                epoch_loss += total_loss.item()

                writer.add_scalar('Loss/train', total_loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': total_loss.item()})

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_value_(net.module.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % ((n_train + n_val) // (2 * batch_size)) == 0:
                    for tag, value in net.module.named_parameters():
                        tag = tag.replace('.', '/')
                        try:
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        except:
                            print('smth goes wrong!')

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.module.n_classes > 2:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.module.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        print(f'epoch_loss = {epoch_loss}')
        
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            with torch.no_grad():
                val_score = eval_net(net, val_loader, device)

            if val_loss_score > val_score:
                val_loss_score = val_score
                torch.save(net.module.state_dict(),
                           dir_checkpoint + f'best_epoch_{epoch + epoch_bias + 1}.pth')
                logging.info(f'Checkpoint {epoch + epoch_bias + 1} saved !')
                logging.info(f'Current min val_loss_score = {val_loss_score}')

    writer.close()
    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('I see' , torch.cuda.device_count(), ' gpus!')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)


    if load:
        net.load_state_dict(
            torch.load(load, map_location=device)
        )
        logging.info(f'Model loaded from {load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)