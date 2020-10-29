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


dir_train_img = '/home/natasha/unet4/axial_data/train/'

dir_val_img = '/home/natasha/unet4/axial_data/val/'

dir_checkpoint = 'ckpts_dir/axial_ckpts/'

img_scale = 1


def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        
    def train_net(net,
              device,
              epochs=5,
              batch_size=1,
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

                masks_pred = net.module(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.module.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % ((n_train + n_val) // (2 * batch_size)) == 0:
                    for tag, value in net.module.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

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
    
    print('now we are here!')
    print('I see' , torch.cuda.device_count(), ' gpus!')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = UNet(n_channels=1, n_classes=6, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        print('we are already here!')

    net.to(device)
    print('and we are here!')

    epochs = 150
    batch_size = 4

    load = False

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