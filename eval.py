import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.module.eval()
    # mask_type = torch.float32 if net.n_classes <= 2 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = nn.BCEWithLogitsLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device).float()
            true_masks = true_masks.to(device=device).float()

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.module.n_classes > 2:
                tot += criterion(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.module.train()
    return tot / n_val
