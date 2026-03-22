"""
pointnet_plus.py
================
PointNet++ for Von Mises stress regression.

FIX v3:
  - n_phys_features defaults to 13 (was 4)
  - regressor input: 1024 + 3 + 13 = 1040
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.tnet import TNet, tnet_regularization_loss


def shared_mlp(in_ch, out_chs, bn=True):
    layers, ch = [], in_ch
    for out_ch in out_chs:
        layers.append(nn.Conv1d(ch, out_ch, 1))
        if bn: layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        ch = out_ch
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, out_channels):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample
        self.mlp     = shared_mlp(in_channels + 3, out_channels)

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        centroids = _fps(xyz, self.npoint)
        new_xyz   = _index_points(xyz, centroids)
        idx         = _ball_query(xyz, new_xyz, self.radius, self.nsample)
        grouped_xyz = _index_points(xyz, idx) - new_xyz.unsqueeze(2)
        if points is not None:
            grouped = torch.cat([grouped_xyz, _index_points(points, idx)], -1)
        else:
            grouped = grouped_xyz
        g = grouped.view(B * self.npoint, self.nsample, -1).permute(0, 2, 1)
        g = self.mlp(g)
        return new_xyz, torch.max(g, dim=2)[0].view(B, self.npoint, -1)


class PointNetPlusPlus(nn.Module):
    """
    Args:
        n_load_directions: 3  (ver/hor/dia — tor has no point clouds)
        n_phys_features:  13  (EDA-selected, redundant inertia dropped)
    Regressor input: 1024 + 3 + 13 = 1040
    """
    def __init__(self, use_tnet=True, dropout=0.4,
                 n_load_directions=3, n_phys_features=13):
        super().__init__()
        self.use_tnet = use_tnet

        if use_tnet:
            self.tnet = TNet(k=3)

        self.sa1 = SetAbstraction(512, 0.2, 32,  0,   [64,  64,  128])
        self.sa2 = SetAbstraction(128, 0.4, 64,  128, [128, 128, 256])
        self.sa3 = SetAbstraction(1,   2.0, 128, 256, [256, 512, 1024])

        in_dim = 1024 + n_load_directions + n_phys_features

        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, xyz, load_dir=None, phys_feat=None):
        trans = None
        if self.use_tnet:
            trans = self.tnet(xyz.permute(0, 2, 1))
            xyz   = torch.bmm(xyz, trans)
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _,      l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.squeeze(1)
        if load_dir   is not None: x = torch.cat([x, load_dir],   dim=1)
        if phys_feat  is not None: x = torch.cat([x, phys_feat],  dim=1)
        return self.regressor(x), trans


def _fps(xyz, npoint):
    B, N, _ = xyz.shape
    device  = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dists     = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[torch.arange(B), farthest].unsqueeze(1)
        d = ((xyz - c) ** 2).sum(dim=2)
        dists    = torch.min(dists, d)
        farthest = dists.argmax(dim=1)
    return centroids


def _index_points(pts, idx):
    B    = pts.shape[0]
    vs   = list(idx.shape); vs[1:] = [1] * (len(vs)-1)
    rs   = list(idx.shape); rs[0]  = 1
    bi   = torch.arange(B, device=pts.device).view(vs).repeat(rs)
    return pts[bi, idx]


def _ball_query(xyz, new_xyz, radius, nsample):
    dists   = torch.cdist(new_xyz, xyz)
    idx     = dists.argsort(dim=2)[:, :, :nsample]
    cen     = idx[:, :, 0:1].expand_as(idx)
    outside = torch.gather(dists, 2, idx) > radius
    idx[outside] = cen[outside]
    return idx