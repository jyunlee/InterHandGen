import sys
sys.path.insert(0, '/workspace/InterDiff/InterDiff/app/PointNet2/models')

import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class=42*3,normal_channel=True):
        super(get_model, self).__init__()
        normal_channel = True

        in_channel = 2 if normal_channel else 0 

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 2048], True)
        self.fc1 = nn.Linear(2048, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, return_feat=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        act = x = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(x))
        x = self.fc3(x)

        if return_feat:
            return x, l3_points, act
        else:
            return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.mse_loss(pred, target)

        return total_loss


