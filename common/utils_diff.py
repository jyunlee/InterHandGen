from __future__ import absolute_import, division

import os
import torch
import numpy as np
from torch.autograd import grad
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import (
        rotation_6d_to_matrix,
        matrix_to_rotation_6d,
        matrix_to_axis_angle,
        axis_angle_to_matrix,
)
from pytorch3d.ops import SubdivideMeshes
from common.dist_chamfer import chamferDist
from common.utils import *
from tqdm import tqdm


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x, cond, seq, model, b, **kwargs):

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long()).squeeze(-1)
            at_next = compute_alpha(b, next_t.long()).squeeze(-1)
            xt = xs[-1]
            x0_t = model(xt, t.float(), cond.float())
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * (x0_t - xt)
            xs.append(xt_next)

    return xs, x0_preds


def cond_generalized_steps(x, cond, seq, model, b, diffhand, cf_scale=0.1, **kwargs):

    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    
    guidance_weights = [20]

    for i in range(len(seq)-1):
        guidance_weights.append(guidance_weights[-1] * 0.98)

    guidance_weights.reverse()

    seq_last = seq[0]

    last_pen_loss = None

    idx = -1
    for i, j in zip(reversed(seq), reversed(seq_next)):

        idx += 1

        t = (torch.ones(n) * i).cuda()
        next_t = (torch.ones(n) * j).cuda()
        at = compute_alpha(b, t.long()).squeeze(-1)
        at_next = compute_alpha(b, next_t.long()).squeeze(-1)
        xt = xs[-1]

        with torch.no_grad():
            cond_x0_t = model(xt, t.float(), cond.float())
            uncond_x0_t = model(xt, t.float(), torch.zeros_like(cond)) 

        # classifier-free guidance
        x0_t = cond_x0_t + cf_scale * (cond_x0_t - uncond_x0_t)
        x0_t[:, :-9] = cond_x0_t[:, :-9]

        grad_list = []

        # penetration guidance
        if diffhand.config.testing.anti_pen:
            print(f' - Computing APG (step {idx+1}/{len(seq)})')
            for sample_idx in tqdm(range(x0_t.shape[0])):

                x0_t_ = x0_t[sample_idx].clone().unsqueeze(0).requires_grad_()
                cond_ = cond[sample_idx].clone().unsqueeze(0)

                anc_handV, add_handV, _, _ = diffhand.mano_forward(cond_, x0_t_, return_verts=True)
                anc_handF = torch.Tensor(diffhand.mano_layer['left'].get_faces().astype(int)).unsqueeze(0).repeat_interleave(anc_handV.shape[0], dim=0).cuda()

                anc_hand_mesh = Meshes(verts=anc_handV, faces=anc_handF)

                add_handF = torch.Tensor(diffhand.mano_layer['right'].get_faces().astype(int)).unsqueeze(0).repeat_interleave(add_handV.shape[0], dim=0).cuda()
                add_hand_mesh = Meshes(verts=add_handV, faces=add_handF)

                if diffhand.subdiv_add is None:
                    diffhand.subdiv_add = SubdivideMeshes(add_hand_mesh[0])

                for _ in range(2):
                    add_hand_mesh = diffhand.subdiv_add.subdivide_homogeneous(add_hand_mesh)

                add_handV = torch.stack(add_hand_mesh.verts_list())
                add_handF = torch.stack(add_hand_mesh.faces_list())
                add_handF = torch.repeat_interleave(add_handF, add_handV.shape[0], dim=0)

                add_hand_mesh = Meshes(verts=add_handV, faces=add_handF)

                add_handVN = add_hand_mesh.verts_normals_padded()

                if diffhand.subdiv is None:
                    diffhand.subdiv = SubdivideMeshes(anc_hand_mesh[0])

                for _ in range(2):
                    anc_hand_mesh = diffhand.subdiv.subdivide_homogeneous(anc_hand_mesh)

                anc_handV = torch.stack(anc_hand_mesh.verts_list())
                anc_handF = torch.stack(anc_hand_mesh.faces_list())
                anc_handF = torch.repeat_interleave(anc_handF, anc_handV.shape[0], dim=0)

                anc_hand_mesh = Meshes(verts=anc_handV, faces=anc_handF)
                anc_handVN = anc_hand_mesh.verts_normals_padded()

                distChamfer = chamferDist()
                contact_robustifier = GMoF_unscaled(rho = 5e-2)

                collide_ids_add, collide_ids_anchor = \
                        collision_check(anc_handV, anc_handVN, add_handV, distChamfer)

                if collide_ids_add is not None:
                    loss_col = contact_robustifier((add_handV[collide_ids_add[0], collide_ids_add[1]] -
                    anc_handV[collide_ids_anchor[0], collide_ids_anchor[1]])).mean() 

                    grad_col = grad(outputs=loss_col, inputs=x0_t_, retain_graph=True)[0]
                else:
                    grad_col = torch.zeros_like(x0_t_)

                grad_list.append(grad_col)
            
            tot_grad_col = torch.cat(grad_list, 0)

            scale = guidance_weights[idx]

            # use a low scale for the last iter
            if idx == len(seq) - 1:
                scale /= 10
                
            x0_t = x0_t - scale * tot_grad_col * 100

        x0_preds.append(x0_t)

        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * (x0_t - xt)
        xs.append(xt_next)

    return xs, x0_preds
