import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- basic indexing helpers ----------

def index_points(points, idx):
    """
    Batched indexing helper.

    Args:
        points: (B, N, C) tensor
            B = batch size
            N = number of points
            C = number of channels/features per point

        idx: (B, S) or (B, S, K) long tensor
            Indices of points we want to select, per batch.

    Returns:
        out: (B, S, C) or (B, S, K, C)
            The selected points, with the same index shape as idx,
            and C features in the last dimension.
    """
    device = points.device
    B = points.shape[0]

    # We want a batch index tensor that has the same number of dimensions
    # as idx, so that we can do:
    #   points[batch_indices, idx, :]
    #
    # Example:
    #   idx.dim() == 2 -> shape (B, 1)
    #   idx.dim() == 3 -> shape (B, 1, 1)
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(*view_shape)

    # Expand batch_indices to match idx shape, so at every position
    # we know "which batch" we are indexing into:
    #   batch_indices: (B, S) or (B, S, K)
    batch_indices = batch_indices.expand_as(idx)

    # Now we can index:
    #   points[batch_indices, idx, :]
    # This uses advanced indexing in PyTorch and gives shape (..., C),
    # where ... is the shape of idx (B, S) or (B, S, K).
    out = points[batch_indices, idx, :]

    return out


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS).

    Args:
        xyz: (B, N, 3)
            Input points (just coordinates).
        npoint: int
            Number of centers to sample per batch.

    Returns:
        centroids: (B, npoint) long
            Indices of chosen center points for each batch.
    """
    device = xyz.device
    B, N, _ = xyz.shape

    # Will store indices of chosen centers
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # For each point, we maintain its distance to the closest chosen center.
    # Initialize them to a huge value.
    distance = torch.full((B, N), 1e10, device=device)

    # Initialize "farthest" with a random point index for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        # 1) record this farthest index as the i-th centroid
        centroids[:, i] = farthest

        # 2) get the coordinates of the current farthest points
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B,1,3)

        # 3) compute squared distance from every point to this centroid
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)          # (B,N)

        # 4) if this distance is smaller than our current "closest center distance",
        #    update that distance (we always keep the min distance to any center)
        mask = dist < distance
        distance[mask] = dist[mask]

        # 5) next farthest = the point that currently has the largest closest-center distance
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    For each center point in new_xyz, find up to `nsample` neighbors within `radius`.

    Args:
        radius: float
            Neighborhood radius.
        nsample: int
            Max number of neighbors to keep.
        xyz: (B, N, 3)
            All points.
        new_xyz: (B, S, 3)
            Center points (subset).

    Returns:
        group_idx: (B, S, nsample) long
            Indices of neighbors in `xyz` for each center.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # Start with all indices [0..N-1] replicated for each center:
    # group_idx[b,s,:] = [0,1,2,...,N-1]
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N)
    group_idx = group_idx.repeat(B, S, 1)  # (B, S, N)

    # squared distances from centers to all points
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    # Any point with distance > radius^2 is "masked out" by setting its index to N
    # (N is an invalid index, equal to "dummy")
    group_idx[sqrdists > radius ** 2] = N

    # Sort along the last dim so that "real" indices (0..N-1) come before N
    # Then keep only the first nsample entries
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # (B, S, nsample)

    # If some centers had fewer than nsample valid neighbors, their remaining
    # positions are filled with N. We now replace those N with the first valid
    # neighbor index for that center (so we don't leave invalid indices).
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)  # (B,S,nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def square_distance(src, dst):
    """
    Compute pairwise squared distances between two point sets.

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        dist: (B, N, M)
            dist[b, i, j] = || src[b, i, :] - dst[b, j, :] ||^2
    """
    B, N, C = src.shape
    _, M, _ = dst.shape

    # -2 * a.b term
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))  # (B, N, M)

    # + ||a||^2 term
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)   # broadcast over M

    # + ||b||^2 term
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)   # broadcast over N

    return dist

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        """
        Args:
            npoint:   number of centers to sample (ignored if group_all=True)
            radius:   ball query radius
            nsample:  max number of neighbors per center
            in_channel: input feature dimension (C); does NOT include xyz.
            mlp:      list like [64, 64, 128]; output feature dim will be mlp[-1].
            group_all: if True, group all points into a single set (global SA).
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # We always concatenate local xyz-relative coords (3) to the features.
        # So first conv sees (C + 3) channels.
        last_channel = in_channel + 3
        layers = []
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        Args:
            xyz:    (B, N, 3) input point coordinates
            points: (B, N, C) input features per point, or None for first layer

        Returns:
            new_xyz:    (B, S, 3) coordinates of sampled centers
            new_points: (B, S, D) output features for each center
                        D = mlp[-1]
        """
        B, N, _ = xyz.shape

        if self.group_all:
            # Global SA: a single center for each batch.
            new_xyz = xyz.mean(dim=1, keepdim=True)      # (B,1,3)
            grouped_xyz = xyz.view(B, 1, N, 3)           # (B,1,N,3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # (B,1,N,3)

            if points is not None:
                grouped_points = points.view(B, 1, N, -1)  # (B,1,N,C)
            else:
                grouped_points = None
        else:
            # 1) Sample centers with FPS: (B,S) indices
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)         # (B,S,3)

            # 2) Group neighbors for each center using ball query
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B,S,K)
            grouped_xyz = index_points(xyz, group_idx)             # (B,S,K,3)

            # Normalize coords so each group is expressed relative to its center
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # (B,S,K,3)

            if points is not None:
                grouped_points = index_points(points, group_idx)    # (B,S,K,C)
            else:
                grouped_points = None

        # Concatenate coordinates and features along the last dim (channel dim)
        if points is not None:
            # (B,S,K,3+C)
            new_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            # (B,S,K,3)
            new_features = grouped_xyz_norm

        # Shape for Conv2d: (B, D_in, S, K)
        new_features = new_features.permute(0, 3, 1, 2)

        # Shared MLP + nonlinearity
        new_features = self.mlp(new_features)      # (B,D',S,K)

        # Max-pool over K neighbors → (B,D',S)
        new_features = torch.max(new_features, dim=-1)[0]

        # Return as (B,S,D')
        new_features = new_features.permute(0, 2, 1)
        return new_xyz, new_features
   
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        """
        Args:
            in_channel: total input channels to this FP layer:
                        C1 (from points1) + C2 (interpolated from points2).
            mlp: list like [256, 256] giving output feature sizes.
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
        Interpolate features from (xyz2, points2) to (xyz1, points1).

        Args:
            xyz1:    (B, N, 3) target points (denser, e.g. finer layer)
            xyz2:    (B, S, 3) source points (sparser, e.g. coarser layer)
            points1: (B, C1, N) features at xyz1 (skip-connected), or None
            points2: (B, C2, S) features at xyz2

        Returns:
            new_points: (B, D, N)
                Updated features at xyz1 after interpolation + MLP.
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # If there's only 1 source point, just tile its features N times.
            interpolated_points = points2.repeat(1, 1, N)  # (B,C2,N)
        else:
            # 1) Compute pairwise distances between xyz1 and xyz2
            dists = square_distance(xyz1, xyz2)          # (B,N,S)

            # 2) Sort to get nearest neighbors
            dists, idx = dists.sort(dim=-1)              # (B,N,S)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # keep 3 nearest (B,N,3)

            # 3) Compute inverse-distance weights for these 3 neighbors
            dist_recip = 1.0 / (dists + 1e-8)            # avoid division by zero
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm                   # (B,N,3)

            # 4) Gather 3 neighbor features from points2
            # points2: (B,C2,S) → (B,S,C2)
            # But index_points expects (B,S,C), so we transpose.
            grouped_points = index_points(points2.permute(0, 2, 1), idx)  # (B,N,3,C2)
            grouped_points = grouped_points.permute(0, 3, 1, 2)          # (B,C2,N,3)

            # 5) Weighted sum over the 3 neighbors → interpolated features
            interpolated_points = torch.sum(grouped_points * weight.unsqueeze(1), dim=-1)  # (B,C2,N)

        # Concatenate with skip-connection features from points1 if provided
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)  # (B,C1+C2,N)
        else:
            new_points = interpolated_points  # (B,C2,N)

        # Run through shared MLP (Conv1d)
        new_points = self.mlp(new_points)  # (B,D,N)
        return new_points
   
    
