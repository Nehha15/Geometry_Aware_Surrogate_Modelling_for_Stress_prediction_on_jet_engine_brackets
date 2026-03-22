"""
tnet.py
=======
Spatial Transformer Network (T-Net) from PointNet.
Learns a 3x3 rotation matrix to align input point clouds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k    = k
        self.conv1 = nn.Conv1d(k, 64,   1)
        self.conv2 = nn.Conv1d(64, 128,  1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  k * k)
        self.bn4   = nn.BatchNorm1d(512)
        self.bn5   = nn.BatchNorm1d(256)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc3.bias = nn.Parameter(torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x.view(B, self.k, self.k)


def tnet_regularization_loss(trans: torch.Tensor) -> torch.Tensor:
    """Orthogonality regularization: ||I - A*A^T||^2_F"""
    k  = trans.shape[1]
    I  = torch.eye(k, device=trans.device).unsqueeze(0)
    AA = torch.bmm(trans, trans.transpose(1, 2))
    return torch.mean(torch.norm(I - AA, dim=(1, 2)) ** 2)
