import os
import logging
import time
import glob
import random
import argparse

import os.path as path
import numpy as np
import cv2 as cv
import tqdm
import trimesh
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from models.twohand_diff import TwoHandDiff
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps, cond_generalized_steps
from common.utils_vis import Renderer

from utils.manolayer import ManoLayer

from pytorch3d.renderer import OrthographicCameras, PointLights, TexturesVertex
from pytorch3d.transforms import (
        rotation_6d_to_matrix,
        matrix_to_rotation_6d,
)

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        mano_layer['left'].shapedirs[:, 0, :] *= -1


class Diffhand(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        mano_path = {'right': os.path.join('misc/mano_v1_2/models', 'MANO_RIGHT.pkl'),
                     'left': os.path.join('misc/mano_v1_2/models', 'MANO_LEFT.pkl')}

        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}

        fix_shape(self.mano_layer)

        self.mano_mean = torch.Tensor(np.load('misc/mano_mean.npy')).to(self.device)
        self.mano_var = torch.Tensor(np.load('misc/mano_var.npy')).to(self.device)
        self.rel_mean = torch.Tensor(np.load('misc/rel_mean.npy')).to(self.device)
        self.rel_var = torch.Tensor(np.load('misc/rel_var.npy')).to(self.device)

        if (self.args.train and self.config.training.vis) or (not self.args.train and self.config.testing.vis):
            self.renderer = Renderer(1024, device=self.device)

            self.n_views = 4
            self.cameras = OrthographicCameras(focal_length = torch.ones((self.n_views,)).float().to(self.device)*8,
                                               principal_point = torch.zeros((self.n_views, 2)).float().to(self.device),
                                               R = torch.tensor([[[0, -1, 0], [0, 0, 1], [1, 0, 0]],
                                                                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                                                [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                                                                [[0, -1, 0], [1, 0, 0], [0, 0, -1]]]).float().to(self.device),
                                               T = torch.repeat_interleave(torch.tensor([[0.02, 0, 0.15]]).float().to(self.device), self.n_views, 0),
                                               in_ndc = True,
                                               device = self.device)

            self.point_lights = PointLights(location=[[0, -0.5, 0],
                                                      [0, 0, -0.5],
                                                      [0, 0.5, 0],
                                                      [0, 0, 0.5]], device=device)

            self.texture = TexturesVertex(verts_features=torch.ones((self.n_views, 778*2, 3)).to(self.device) * 200)

        self.loss = torch.nn.MSELoss()

        self.subdiv = None # warn
        self.subdiv_add = None # warn

    def prepare_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))

        if config.data.dataset == "interhand":
            from common.interhand_dataset import InterHandDataset 

            if self.args.train:
                self.train_dataset = InterHandDataset(config.data.dataset_path, 'train')
                self.val_dataset = InterHandDataset(config.data.dataset_path, 'val')
            else:
                self.test_dataset = InterHandDataset(config.data.dataset_path, 'test')

        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        self.model_diff = TwoHandDiff(config).cuda() 
        self.model_diff = torch.nn.DataParallel(self.model_diff)

        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])

    def mano_forward(self, anc_param, add_param, return_verts=False, normalize=True):

        if anc_param.get_device() >= 0:
            device = torch.device(f'cuda:{anc_param.get_device()}')
        else:
            device = torch.device('cpu')

        if len(anc_param.shape) == 1:
            anc_param = anc_param.unsqueeze(0)
            add_param = add_param.unsqueeze(0)
            n = 1
        else:
            n = anc_param.shape[0]

        if normalize:
            mano_mean = self.mano_mean.to(device)
            mano_var = self.mano_var.to(device)
            rel_mean = self.rel_mean.to(device)
            rel_var = self.rel_var.to(device)

            # to avoid in-place operation
            anc_param_ = anc_param.clone()
            add_param_ = add_param.clone()

            anc_param_[:, :-9] = (anc_param_[:, :-9] * (mano_var + 1e-6)) + mano_mean
            add_param_[:, :-9] = (add_param_[:, :-9] * (mano_var + 1e-6)) + mano_mean

            add_param_[:, -9:] = (add_param_[:, -9:] * (rel_var + 1e-6)) + rel_mean

            anc_param = anc_param_
            add_param = add_param_

        anc_pose = anc_param[:, :-19]
        anc_shape = anc_param[:, -19:-9]

        add_pose = add_param[:, :-19]
        add_shape = add_param[:, -19:-9]
        add_rot = rotation_6d_to_matrix(add_param[:, -9:-3])
        add_trans = add_param[:, -3:]

        anc_handV, anc_handJ = self.mano_layer['left'](torch.repeat_interleave(torch.eye(3, device=device).unsqueeze(0).float(), n, dim=0),
                                                       anc_pose.float(),
                                                       anc_shape.float(),
                                                       torch.zeros((n, 3), device=device).float()) # dummy value

        anc_handF = self.mano_layer['left'].get_faces()

        add_handV, add_handJ = self.mano_layer['right'](add_rot.float(),
                                                        add_pose.float(),
                                                        add_shape.float(),
                                                        torch.zeros((n, 3), device=device).float()) # dummy value

        add_handF = self.mano_layer['right'].get_faces()

        anc_handV = anc_handV - anc_handJ[:, 9, :].unsqueeze(1)
        anc_handJ = anc_handJ - anc_handJ[:, 9, :].unsqueeze(1)
        add_handV = add_handV - add_handJ[:, 9, :].unsqueeze(1) + add_trans.unsqueeze(1)
        add_handJ = add_handJ - add_handJ[:, 9, :].unsqueeze(1) + add_trans.unsqueeze(1)

        if return_verts:
            return anc_handV, add_handV, anc_handJ, add_handJ

        manos = []

        for i in range(n):
            handV = np.concatenate((anc_handV[i], add_handV[i]), axis=0)
            handF = np.concatenate((anc_handF, add_handF + anc_handV[i].shape[0]), axis=0)
            manos.append(trimesh.Trimesh(handV, handF, process=False))

        params = torch.cat((anc_param, add_param), dim=-1)
        handV = torch.cat((anc_handV, add_handV), dim=1)

        return manos, params, handV

    def mano_forward_gt(self, hand_dict, augment=True, normalize=True):

        swapped = False

        if augment:
            rand = random.randint(0, 1)

            if rand == 0:
                anchor_hand_type = 'left'
                add_hand_type = 'right'
            else:
                anchor_hand_type = 'right'
                add_hand_type = 'left'

                new_hand_dict = {}
                new_hand_dict['anchor'] = hand_dict['add']
                new_hand_dict['add'] = hand_dict['anchor']

                del hand_dict
                hand_dict = new_hand_dict

                swapped = True

        # obtain relative transformation
        anc_handV, anc_handJ = self.mano_layer[anchor_hand_type](hand_dict['anchor']['rot'].float(),
                                                                 hand_dict['anchor']['pose'].float(),
                                                                 hand_dict['anchor']['shape'].float(),
                                                                 trans=hand_dict['anchor']['trans'].float())

        add_handV, add_handJ = self.mano_layer[add_hand_type](hand_dict['add']['rot'].float(),
                                                                 hand_dict['add']['pose'].float(),
                                                                 hand_dict['add']['shape'].float(),
                                                                 trans=hand_dict['add']['trans'].float())

        rel_trans = (add_handJ[:, 9] - anc_handJ[:, 9]).squeeze()

        # compute relative rotation
        anc_rot = hand_dict['anchor']['rot']
        add_rot = hand_dict['add']['rot']

        if anchor_hand_type == 'right':
            anc_rot[:, :, 0] *= -1.
            add_rot[:, :, 0] *= -1.

        rel_rot = torch.inverse(anc_rot) @ add_rot
        rel_trans = (torch.inverse(anc_rot.float()) @ rel_trans.float().unsqueeze(-1)).squeeze()

        anc_rot[:] = 0.
        anc_rot[:, 0, 0] = 1.
        anc_rot[:, 1, 1] = 1.
        anc_rot[:, 2, 2] = 1.

        hand_dict['anchor']['rot'] = anc_rot
        hand_dict['add']['rot'] = rel_rot

        n = anc_rot.shape[0]

        # mano layer forward pass
        anc_handV, anc_handJ = self.mano_layer['left'](hand_dict['anchor']['rot'].float(),
                                                       hand_dict['anchor']['pose'].float(),
                                                       hand_dict['anchor']['shape'].float(),
                                                       trans=torch.zeros((n, 3)).float()) # dummy value

        add_handV, add_handJ = self.mano_layer['right'](hand_dict['add']['rot'].float(),
                                                        hand_dict['add']['pose'].float(),
                                                        hand_dict['add']['shape'].float(),
                                                        trans=torch.zeros((n, 3)).float()) # dummy value

        anc_handV = anc_handV - anc_handJ[:, 9, :].unsqueeze(1)
        anc_handJ = anc_handJ - anc_handJ[:, 9, :].unsqueeze(1)
        add_handV = add_handV - add_handJ[:, 9, :].unsqueeze(1) + rel_trans.unsqueeze(1)
        add_handJ = add_handJ - add_handJ[:, 9, :].unsqueeze(1) + rel_trans.unsqueeze(1)

        hand_dict['add']['trans'] = rel_trans
        hand_dict['anchor']['trans'][:] = 0.

        hand_dict['anchor']['rot'] = matrix_to_rotation_6d(hand_dict['anchor']['rot'])
        hand_dict['add']['rot'] = matrix_to_rotation_6d(hand_dict['add']['rot'])

        if not swapped:
            anc_valid = torch.Tensor([True,] * n).cuda().bool()
        else:
            anc_valid = hand_dict['anchor']['inter']

        add_valid = hand_dict['anchor']['inter']

        if normalize:
            if anc_rot.get_device() >= 0:
                device = torch.device(f'cuda:{anc_rot.get_device()}')
            else:
                device = torch.device('cpu')

            mano_mean = self.mano_mean.to(device)
            mano_var = self.mano_var.to(device)
            rel_mean = self.rel_mean.to(device)
            rel_var = self.rel_var.to(device)

            pose_mean = mano_mean[:-10]
            pose_var = mano_var[:-10]

            hand_dict['anchor']['pose'] = (hand_dict['anchor']['pose'] - pose_mean) / (pose_var + 1e-6)
            hand_dict['add']['pose'] = (hand_dict['add']['pose'] - pose_mean) / (pose_var + 1e-6)

            shape_mean = mano_mean[-10:]
            shape_var = mano_var[-10:]

            hand_dict['anchor']['shape'] = (hand_dict['anchor']['shape'] - shape_mean) / (shape_var + 1e-6)
            hand_dict['add']['shape'] = (hand_dict['add']['shape'] - shape_mean) / (shape_var + 1e-6)

            rot_mean = rel_mean[:-3]
            rot_var = rel_var[:-3]

            hand_dict['add']['rot'] = (hand_dict['add']['rot'] - rot_mean) / (rot_var + 1e-6)

            trans_mean = rel_mean[-3:]
            trans_var = rel_var[-3:]

            hand_dict['add']['trans'] = (hand_dict['add']['trans'] - trans_mean) / (trans_var + 1e-6)

        return hand_dict, anc_handV, add_handV, anc_handJ, add_handJ, anc_valid, add_valid

    def train(self):
        cudnn.benchmark = True

        args, config = self.args, self.config

        best_p1, best_epoch = 1000, 0
        stride = 1

        if config.data.dataset == 'interhand':
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                    num_workers=self.config.data.num_workers, drop_last=True, pin_memory=True)

            val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                        num_workers=self.config.data.num_workers, drop_last=False, pin_memory=True)

        else:
            raise KeyError('Invalid dataset')

        optimizer = get_optimizer(self.config, self.model_diff.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None

        start_epoch, step = 0, 0

        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma

        for epoch in range(start_epoch, self.config.training.n_epochs):
            print(f'[Test epoch] {epoch+1} / {n_epochs}') 
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()

            epoch_loss_diff = AverageMeter()

            for i, data in enumerate(train_dataloader):

                data, gt_anchor_verts, gt_add_verts, gt_anchor_joints, gt_add_joints, anc_valid, add_valid = self.mano_forward_gt(data)

                inter_mask = data['anchor']['inter']

                # anchor hand generation
                loss_dict = {}

                targets_3d = torch.cat([data['anchor']['pose'], data['anchor']['shape'], data['anchor']['rot'], data['anchor']['trans']], -1).to(self.device)

                cond = torch.zeros_like(targets_3d)

                data_time += time.time() - data_start
                step += 1

                # generate nosiy sample based on seleted time t and beta
                n = targets_3d.size(0)
                x = targets_3d
                e = torch.randn_like(x)

                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1)

                # generate x_t (refer to DDIM equation)
                cond_x = x = x * a.sqrt() + e * (1.0 - a).sqrt()

                anc_signal = output_signal = self.model_diff(x.float(), t.float(), cond.float())

                loss_dict['anc_param'] = self.loss(output_signal[anc_valid][:, :-9].float(), targets_3d[anc_valid][:, :-9].float())
                    
                # additional hand generation
                cond = targets_3d
                targets_3d = torch.cat([data['add']['pose'], data['add']['shape'], data['add']['rot'], data['add']['trans']], -1).to(self.device)

                # generate nosiy sample based on seleted time t and beta
                n = targets_3d.size(0)
                x = targets_3d
                e = torch.randn_like(x)

                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()

                add_signal = output_signal = self.model_diff(x.float(), t.float(), cond.float())

                loss_dict['add_param'] = self.loss(output_signal[add_valid].float(), targets_3d[add_valid].float())
                
                anc_verts, add_verts, anc_joints, add_joints = self.mano_forward(anc_signal.detach().cpu(), add_signal.detach().cpu(), return_verts=True)

                norm = add_joints[:, 9, :].unsqueeze(1)
                add_joints -= norm
                add_verts -= norm

                gt_norm = gt_add_joints[:, 9, :].unsqueeze(1)
                gt_add_joints -= gt_norm
                gt_add_verts -= gt_norm

                loss_dict['anc_verts'] = self.loss(anc_verts[anc_valid], gt_anchor_verts[anc_valid]) * 1E4
                loss_dict['add_verts'] = self.loss(add_verts[add_valid], gt_add_verts[add_valid]) * 1E4

                loss_dict['anc_joints'] = self.loss(anc_joints[anc_valid], gt_anchor_joints[anc_valid]) * 1E4
                loss_dict['add_joints'] = self.loss(add_joints[add_valid], gt_add_joints[add_valid]) * 1E4

                loss_diff = loss_dict['anc_param'] + loss_dict['add_param'] + loss_dict['anc_verts'] + loss_dict['add_verts'] + loss_dict['anc_joints'] + loss_dict['add_joints']

                optimizer.zero_grad()
                loss_diff.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)
                optimizer.step()

                epoch_loss_diff.update(loss_diff.item(), n)

                if self.config.model.ema:
                    ema_helper.update(self.model_diff)

                if i % 100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(train_dataloader), step, data_time, epoch_loss_diff.avg))

                    '''
                    pred_manos, _ = self.mano_forward(anc_signal.detach().cpu(), add_signal.detach().cpu())
                    gt_manos, _ = self.mano_forward(cond.detach().cpu(), targets_3d.detach().cpu())
                    input_noisy_manos, _ = self.mano_forward(x.detach().cpu(), cond_x.detach().cpu())

                    for i in range(20):
                        pred_manos[i].export(f'debug/output_signal_{i}.obj')
                        gt_manos[i].export(f'debug/gt_signal_{i}.obj')
                        input_noisy_manos[i].export(f'debug/input_{i}.obj')
                    '''

            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma)

            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                logging.info('test the performance of current model')

                self.test_hyber(is_train=True)
                p1 = epoch_loss_diff.avg

                if p1 <= best_p1:
                    best_p1 = p1
                    best_epoch = epoch

                    torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))

                logging.info('| Best Epoch: {:0>4d} Loss: {:.2f} | Epoch: {:0>4d} Loss: {:.2f} |'\
                    .format(best_epoch, best_p1, epoch, p1))


    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config = self.args, self.config
        test_times, test_timesteps, test_num_diffusion_timesteps = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps
        stride = 1

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        self.model_diff.eval()

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError

        if is_train:
            batch_size = self.config.training.vis_batch
            size = (batch_size, 64)
            n_epochs = 1

        else:
            batch_size = self.config.testing.vis_batch
            size = (batch_size, 64) 
            n_epochs = self.config.testing.vis_epoch

        total_params = torch.zeros((0, 128))
        total_verts = torch.zeros((0, 778*2, 3))
        total_manos = []

        for epoch in tqdm.tqdm(range(n_epochs)):
            data_time += time.time() - data_start

            # select diffusion step
            t = torch.ones(size[0]).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps

            # prepare the diffusion parameters
            e = torch.randn(size).to(self.device)
            b = self.betas
            e = e

            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1)

            cond = torch.zeros_like(e)
            _, anchor_param = generalized_steps(e, cond, seq, self.model_diff, self.betas, eta=self.args.eta)
            anchor_param = anchor_param[-1]

            cond = anchor_param
            cond[:, -9:] = torch.Tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.]).cuda().float()

            _, add_param = cond_generalized_steps(e, cond, seq, self.model_diff, self.betas, self, eta=self.args.eta)
            add_param = add_param[-1]

            # save outputs
            out_dir = os.path.join(self.args.exp, self.args.doc, 'res')
            mesh_dir = os.path.join(out_dir, 'meshes')
            input_dir = os.path.join(out_dir, 'inputs')
            img_dir = os.path.join(out_dir, 'renderings')

            out_dirs = [out_dir, mesh_dir, input_dir, img_dir]

            for dir in out_dirs:
                if not os.path.exists(dir):
                    os.system(f'mkdir -p {dir}')

            print(f'Saving results to {out_dir}...')

            pred_manos, denorm_param, verts = self.mano_forward(anchor_param.detach().cpu(), add_param.detach().cpu())

            total_params = torch.cat((total_params, denorm_param.detach().cpu()), dim=0)
            total_verts = torch.cat((total_verts, verts.detach().cpu()), dim=0)
            total_manos += pred_manos

        np.save(f'{out_dir}/params.npy', total_params.numpy())
        np.save(f'{out_dir}/verts.npy', total_verts.numpy())
        
        if (is_train and self.config.training.vis) or (not is_train and self.config.testing.vis):

            for i in tqdm.tqdm(range(len(total_manos))):

                total_manos[i].export(f'{mesh_dir}/mesh_{i}.obj')

                '''
                imgs, alphas = self.renderer.render(verts=torch.repeat_interleave(torch.Tensor(total_manos[i].vertices * 0.5).unsqueeze(0), self.n_views, 0), faces=torch.repeat_interleave(torch.Tensor(total_manos[i].faces).unsqueeze(0), self.n_views, 0), cameras=self.cameras, textures=self.texture, lights=self.point_lights)

                res_imgs = []

                for img, alpha in zip(imgs, alphas): 
                    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
                    bg_img = np.ones(img.shape).astype(np.uint8) * 255

                    mask = alpha.detach().cpu().numpy()[..., np.newaxis]

                    img_out = img * mask + bg_img * (1 - mask)

                    res_imgs.append(np.flip(np.transpose(img_out, [1, 0, 2]), 0))

                res_img = cv.hconcat(res_imgs)

                cv.imwrite(f'{img_dir}/img_{i}.png', res_img)
                '''
        return
