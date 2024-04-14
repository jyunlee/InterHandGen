import os.path as osp
import os
import datetime
import argparse
import trimesh
import torch
import importlib
import numpy as np
import sys as _sys

from tqdm import tqdm

from utils.fid import calculate_fid
from utils.diversity import calculate_diversity
from utils.kid import calculate_kid
from utils.precision_recall import precision_and_recall
from utils.pointnet2_twohand import *

def parse_args():

    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--sample_num", type=int, default=10000, help="Number of samples to evaluate")
    parser.add_argument("--doc", type=str, default='default', help="Model directory (doc argument used in the training script)")

    return parser.parse_args()


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def normalize(verts):

    verts_min = verts.min(-1)[0]
    verts_max = verts.max(-1)[0]

    center = (verts_min + verts_max) / 2
    scale = (verts_max - verts_min).max(-1)[0]

    verts = (verts - center.unsqueeze(-1)) / scale.unsqueeze(-1).unsqueeze(-1)

    return verts


def preprocess(verts):

    verts = verts.cpu() * 10

    side_label = torch.cat((torch.repeat_interleave(torch.Tensor([[0, 1]]), 778, dim=0), torch.repeat_interleave(torch.Tensor([[1, 0]]), 778, dim=0)), 0)
    side_label = side_label.T
    side_label = torch.repeat_interleave(side_label.unsqueeze(0), verts.shape[0], 0)

    verts = torch.cat((verts, side_label), dim=1)

    return verts


def evaluate(exp, n_samples=10000, res='res', fast=False, p_batch=64):

    model = get_model(99, normal_channel=False)
    model = model.cuda()
    model = model.eval()

    checkpoint = torch.load('../misc/pointnet2_twohand.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    pred_verts = np.load(osp.join('..', 'exp', exp, res, 'verts.npy'))
    pred_verts = torch.Tensor(pred_verts).cuda().transpose(1,2)

    pred_verts = preprocess(pred_verts[:n_samples])
    pred_feats = torch.zeros((0, 256))

    print('Extracting pointnet pred feature...')
    for i in tqdm(range(n_samples//p_batch + 1)):
        with torch.no_grad():

            if i != n_samples//p_batch: 
                kpts, _, actv = model(pred_verts[i*p_batch : (i+1)*p_batch].cuda(), return_feat=True)
            else: 
                _, _, actv = model(pred_verts[i*p_batch :].cuda(), return_feat=True)

            pred_feats = torch.cat((pred_feats, actv.detach().cpu()), axis=0)
            del actv

    pred_feats = pred_feats.numpy()

    gt_feats_path = os.path.join('..', 'misc', f'verts_gt_feat.npy')

    if not os.path.exists(gt_feats_path):
        print('Extracting pointnet gt feature...')
        gt_verts = np.load(osp.join('..', 'exp', 'verts_gt.npy'))
        gt_verts = torch.Tensor(gt_verts).transpose(1,2)

        gt_verts = preprocess(gt_verts[:n_samples])
        gt_feats = torch.zeros((0, 256))
        n = n_samples

        for i in tqdm(range(n//p_batch + 1)):
            with torch.no_grad():

                if i != n//p_batch: 
                    _, _, actv = model(gt_verts[i*p_batch : (i+1)*p_batch].cuda(), return_feat=True)
                else: 
                    _, _, actv = model(gt_verts[i*p_batch :].cuda(), return_feat=True)

                gt_feats = torch.cat((gt_feats, actv.detach().cpu()), axis=0)
                del actv

        gt_feats = gt_feats.numpy()
        np.save(gt_feats_path, gt_feats)
    else:
        gt_feats = np.load(gt_feats_path)

    # choose n_samples
    pred_perm = np.random.permutation(pred_feats.shape[0])[:n_samples]
    gt_perm = np.random.permutation(gt_feats.shape[0])[:n_samples]

    pred_feats = pred_feats[pred_perm]
    gt_feats = gt_feats[gt_perm]

    generated_stats = calculate_activation_statistics(pred_feats)
    real_stats = calculate_activation_statistics(gt_feats)

    print("evaluation results:\n")

    fid = calculate_fid(generated_stats, real_stats)
    print(f"FID score: {fid}\n")

    print("calculating KID...")
    kid = calculate_kid(gt_feats, pred_feats)
    (m, s) = kid
    print('KID : %.4f (%.4f)\n' % (m, s))

    dataset_diversity = calculate_diversity(gt_feats)
    generated_diversity = calculate_diversity(pred_feats)
    print(f"Diversity of generated motions: {generated_diversity}")
    print(f"Diversity of dataset motions: {dataset_diversity}\n")

    if fast:
        print("Skipping precision-recall calculation\n")
        precision = recall = None
    else:
        # For precision-recall, use 1000 samples to match the original metric definition in MDM [Tevet et al., ICLR 2023].
        # Since it's based on KNN, sample number affects the neighborhood size.
        print("calculating precision recall...")

        idx = torch.randperm(n_samples)[:1000]  
        precision, recall = precision_and_recall(pred_feats[idx], gt_feats[idx])
        print(f"precision: {precision}")
        print(f"recall: {recall}\n")

    metrics = {'fid': fid, 'kid': kid[0], 'diversity_gen': generated_diversity.cpu().item(), 'diversity_gt':  dataset_diversity.cpu().item(),
                 'precision': precision, 'recall':recall}
    return metrics


def evaluate_penetration(exp, idx, return_dict, res='res', gt=False):

    if gt:
        return NotImplementedError
    else:
        mesh_dir = osp.join('..', 'exp', exp, res, 'meshes')

    mesh = trimesh.load(osp.join(mesh_dir, f'mesh_{idx}.obj'), process=False)
    
    verts, faces = mesh.vertices, mesh.faces

    left_mesh = trimesh.Trimesh(verts[:778], faces[:1538], process=False)
    right_mesh = trimesh.Trimesh(verts[778:], faces[1538:] - 778, process=False)
    
    try:
        pen = trimesh.boolean.intersection([left_mesh, right_mesh]).volume
    except:
        pen = 0.

    return_dict[idx] = pen


if __name__ == '__main__':

    args = parse_args()
    exp, n_samples = args.doc, args.sample_num

    print(f'Start computing evaluation metrics for {n_samples} samples...')
    evaluate(exp, n_samples)

    print(f'Start computing penetration for {n_samples} samples...')

    # multi-processing by default
    import multiprocessing

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []

    for i in range(n_samples):
        p = multiprocessing.Process(target=evaluate_penetration, args=(exp, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in tqdm(jobs):
        proc.join()

    pen_list = return_dict.values()
    penetration = sum(pen_list) / len(pen_list)
    print(f'penetration: {penetration*10**6}')
    
    
