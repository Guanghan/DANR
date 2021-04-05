from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

def image_mixup(batch):
    alpha = 1.5
    lam = np.random.beta(alpha, alpha)

    flipped_input = torch.flip(batch['input'], [0])
    batch['input'] = lam * batch['input'] + (1 - lam) * flipped_input

    flipped_hm = torch.flip(batch['hm'], [0])
    batch['hm'] = lam * batch['hm'] + (1 - lam) * flipped_hm

    max_vals = batch['hm'].max(1)[0].max(1)[0].max(1)[0]  # max val for each sample in a mini-batch
    max_vals[max_vals == 0] = 1  # if max_val == 0, will return nan during normalization
    batch_size = max_vals.shape[0]
    max_vals = max_vals.reshape(batch_size, 1, 1, 1)
    batch['hm'] = batch['hm'] / max_vals   # normalize so that the maximum value is 1

    flipped_ind = torch.flip(batch['ind'], [0])
    batch['ind'] = torch.cat((batch['ind'], flipped_ind), 1)

    flipped_wh = torch.flip(batch['wh'], [0])
    batch['wh'] = torch.cat((batch['wh'], flipped_wh), 1)

    flipped_reg = torch.flip(batch['reg'], [0])
    batch['reg'] = torch.cat((batch['reg'], flipped_reg), 1)

    flipped_mask = torch.flip(batch['reg_mask'], [0])
    batch['reg_mask'] = torch.cat((batch['reg_mask'], flipped_mask), 1)

    return batch


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)
