import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- basic indexing helpers ----------

def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]

    # Make batch_indices with same number of dims as idx
    # idx.dim() == 2 -> shape (B,1)
    # idx.dim() == 3 -> shape (B,1,1)
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(*view_shape)

    # Expand to match idx
    batch_indices = batch_indices.expand_as(idx)
    # Now we can index
    out = points[batch_indices, idx, :]   # (..., C)

    return out



def farthest_point_sample(xyz, npoint):
    """
    xyz: (B, N, 3)
    npoint: number of points to sample
    returns: (B, npoint) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    radius: local region radius
    nsample: max number of points in each ball
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3) centers

    return: group_idx: (B, S, nsample)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N)
    group_idx = group_idx.repeat(B, S, 1)  # (B, S, N)

    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx[sqrdists > radius ** 2] = N    # mark far points with dummy index

    # take the closest nsample points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # (B, S, nsample)

    # if some groups have < nsample valid points, pad by first index
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def square_distance(src, dst):
    """
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M) squared distances
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        """
        npoint:   number of centers to sample
        radius:   ball query radius
        nsample:  number of neighbors in each local group
        in_channel: input feature dim (C)
        mlp:      list like [64, 64, 128]
        group_all: if True, use all points as one group (no FPS)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel + 3  # we concat xyz-relative coords
        layers = []
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        xyz:    (B, N, 3) input xyz coordinates
        points: (B, N, C) input features (can be None for first layer)

        returns:
            new_xyz:    (B, S, 3) sampled centers
            new_points: (B, S, D) new features
        """
        B, N, _ = xyz.shape

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            grouped_xyz = xyz.view(B, 1, N, 3)       # (B, 1, N, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # (B,1,N,3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
            else:
                grouped_points = None
        else:
            # 1) FPS
            fps_idx = farthest_point_sample(xyz, self.npoint)      # (B, S)
            new_xyz = index_points(xyz, fps_idx)                   # (B, S, 3)

            # 2) group neighbors
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B,S,K)
            grouped_xyz = index_points(xyz, group_idx)                             # (B,S,K,3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)                  # relative coords

            if points is not None:
                grouped_points = index_points(points, group_idx)                   # (B,S,K,C)
            else:
                grouped_points = None

        if points is not None:
            # concat relative coordinates and features
            new_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)   # (B,S,K,3+C)
        else:
            new_features = grouped_xyz_norm

        # (B,S,K,D) -> (B,D,S,K) for Conv2d
        new_features = new_features.permute(0, 3, 1, 2)
        new_features = self.mlp(new_features)      # (B,D',S,K)
        new_features = torch.max(new_features, dim=-1)[0]  # max over K -> (B,D',S)

        # transpose to (B,S,D')
        new_features = new_features.permute(0, 2, 1)
        return new_xyz, new_features
    
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        """
        in_channel: concatenated dim of (unknown_features + interpolated known_features)
        mlp: e.g. [256, 256]
        """
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate from (xyz2, points2) -> (xyz1, points1)

        xyz1:    (B, N, 3) target points (denser)
        xyz2:    (B, S, 3) source points (sparser)
        points1: (B, C1, N) features at xyz1 (skip-connected, can be None)
        points2: (B, C2, S) features at xyz2

        returns:
            new_points: (B, D, N)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # just repeat global feature
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # find 3 nearest neighbors in xyz2 for each point in xyz1
            dists = square_distance(xyz1, xyz2)          # (B,N,S)
            dists, idx = dists.sort(dim=-1)              # (B,N,S)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # (B,N,3)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm                    # (B,N,3)

            grouped_points = index_points(points2.permute(0,2,1), idx)  # (B,N,3,C2)
            grouped_points = grouped_points.permute(0,3,1,2)           # (B,C2,N,3)

            # weighted sum
            interpolated_points = torch.sum(grouped_points * weight.unsqueeze(1), dim=-1)  # (B,C2,N)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)  # (B,C1+C2,N)
        else:
            new_points = interpolated_points

        new_points = self.mlp(new_points)  # (B,D,N)
        return new_points

    
    
