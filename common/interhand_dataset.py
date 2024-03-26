import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import random
import pickle
from glob import glob

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.manolayer import ManoLayer, rodrigues_batch

from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

class InterHandDataset():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split

        mano_path = {'right': os.path.join('misc/mano_v1_2/models', 'MANO_RIGHT.pkl'),
                     'left': os.path.join('misc/mano_v1_2/models', 'MANO_LEFT.pkl')}
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        hand_dict = {}

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        anchor_hand_type = 'left'
        add_hand_type = 'right'

        anchor_params = data['mano_params'][anchor_hand_type]
        add_params = data['mano_params'][add_hand_type]

        hand_dict['anchor'] = {'pose': anchor_params['pose'][0],
                               'shape': anchor_params['shape'][0],
                               'rot': anchor_params['R'][0],  
                               'trans': anchor_params['trans'].squeeze(),
                               'inter': True
                               }

        hand_dict['add'] = {'pose': add_params['pose'][0],
                            'shape': add_params['shape'][0],
                            'rot': add_params['R'][0],
                            'trans': add_params['trans'].squeeze(),
                            'inter': True
                           }

        return hand_dict 

