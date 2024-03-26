from __future__ import absolute_import, division

import os
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.nn as nn




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

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=0.000,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=0.000)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
# codes adapted from https://github.com/Malefikus/MoCapDeform
class GMoF_unscaled(nn.Module):
    def __init__(self, rho=1):
        super(GMoF_unscaled, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return dist

def collision_check(scene_v, scene_vn, human_vertices, distChamfer):
    # compute the full-body chamfer distance
    contact_dist, _, contact_scene_idx, _ = distChamfer(human_vertices, scene_v)
    contact_scene_idx = contact_scene_idx.cpu().numpy()
    # generate collision mask via normal check
    batch_idx = torch.arange(scene_v.shape[0]).view(-1, 1)
    human2scene_norm = scene_v[batch_idx, contact_scene_idx] - human_vertices
    human2scene_norm = F.normalize(human2scene_norm, p=2, dim=-1)
    scene_norm = scene_vn[batch_idx, contact_scene_idx]
    collide_mask = torch.sum(human2scene_norm * scene_norm, dim=-1)
    collide_mask = collide_mask > 0

    collide_ids_human = torch.nonzero(collide_mask).T

    # return collide_ids_human and collide_ids_scene
    if collide_ids_human.numel():
        collide_ids_human = collide_ids_human.cpu().numpy()
        collide_ids_scene = contact_scene_idx[collide_ids_human[0], collide_ids_human[1]]
        collide_ids_scene = np.concatenate([np.expand_dims(collide_ids_human[0], 0), np.expand_dims(collide_ids_scene, 0)], axis=0)
    else:
        collide_ids_human = None
        collide_ids_scene = None

    return collide_ids_human, collide_ids_scene



