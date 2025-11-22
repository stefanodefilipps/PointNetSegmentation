import torch.nn as nn
from segmentation_models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstraction
import torch.nn.functional as F
import torch


class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes, input_dim=3):
        """
        PointNet++ style segmentation network (encoder-decoder on point clouds).

        Args:
            num_classes: number of segmentation classes per point.
            input_dim:   number of input channels per point:
                         - 3  -> (x, y, z)
                         - 6  -> (x, y, z, nx, ny, nz), etc.
        """
        super().__init__()

        # ==============================================================
        # ENCODER: Set Abstraction (SA) layers
        # Each SA layer:
        #   - samples fewer "center" points
        #   - groups neighbors around centers
        #   - runs a local PointNet on each group
        # ==============================================================

        # SA1: From N points to 1024 centers, no extra input features (in_channel=0).
        #      Only xyz-relative coords go into the MLP.
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,          # number of centers to sample
            radius=0.1,           # neighborhood radius
            nsample=32,           # max points per local group
            in_channel=0,         # no extra features, only xyz
            mlp=[32, 32, 64],     # output feature dim = 64
            group_all=False,
        )

        # SA2: 1024 -> 256 centers, input features come from SA1 (64-dim).
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64,        # feature dim from SA1
            mlp=[64, 64, 128],    # output feature dim = 128
            group_all=False,
        )

        # SA3: 256 -> 64 centers, input features from SA2 (128-dim).
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128,       # feature dim from SA2
            mlp=[128, 128, 256],  # output feature dim = 256
            group_all=False,
        )

        # SA4: global set abstraction.
        #      group_all=True means we treat all points as one group,
        #      so we get a single global point with a 1024-dim feature.
        self.sa4 = PointNetSetAbstraction(
            npoint=None,          # ignored when group_all=True
            radius=None,          # ignored
            nsample=None,         # ignored
            in_channel=256,       # feature dim from SA3
            mlp=[256, 512, 1024], # output feature dim = 1024
            group_all=True,
        )

        # ==============================================================
        # DECODER: Feature Propagation (FP) layers
        # Each FP layer:
        #   - interpolates features from a coarser (sparser) level
        #     to a finer (denser) level
        #   - concatenates skip-connection features from the encoder
        #   - refines them with an MLP (Conv1d)
        # ==============================================================

        # FP4: from level 4 (global, 1 point with 1024-dim) back to level 3 (64 points).
        # in_channel = 1024 (from SA4) + 256 (from SA3 skip-connection)
        self.fp4 = PointNetFeaturePropagation(
            in_channel=1024 + 256,
            mlp=[256, 256],
        )

        # FP3: from level 3 (64 pts, 256-dim) to level 2 (256 pts).
        # in_channel = 256 (interpolated from FP4) + 128 (SA2 features)
        self.fp3 = PointNetFeaturePropagation(
            in_channel=256 + 128,
            mlp=[256, 256],
        )

        # FP2: from level 2 (256 pts, 256-dim) to level 1 (1024 pts).
        # in_channel = 256 (interpolated) + 64 (SA1 features)
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 64,
            mlp=[256, 128],
        )

        # FP1: from level 1 (1024 pts, 128-dim) to level 0 (original N pts).
        # in_channel = 128 (interpolated from FP2) + input_dim (original per-point features, e.g. xyz)
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + input_dim,
            mlp=[128, 128, 128],
        )

        # ==============================================================
        # Per-point segmentation head
        # Takes final per-point features (B, 128, N) and predicts logits
        # for each class at each point.
        # ==============================================================

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, N)
               B = batch size
               C = input_dim (xyz + optional extras)
               N = number of points

        Returns:
            logits:    (B, num_classes, N)
            l4_points: (B, 1, 1024) global feature (can be used for inspection)
            l3_points: (B, 64, 256) mid-level features (also for inspection)
        """
        B, C, N = x.shape

        # Split coordinates and extra features
        xyz = x[:, :3, :].transpose(1, 2)   # (B, N, 3)
        if C > 3:
            points = x[:, 3:, :].transpose(1, 2)  # (B, N, C-3)
        else:
            points = None

        # ---------------- Encoder (downsampling) ----------------

        # Level 0: original points
        l0_xyz, l0_points = xyz, points                       # (B,N,3), (B,N,C-3 or None)

        # SA1: N -> 1024 points, 64-dim features
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)       # (B,1024,3), (B,1024,64)

        # SA2: 1024 -> 256 points, 128-dim features
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)       # (B,256,3), (B,256,128)

        # SA3: 256 -> 64 points, 256-dim features
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)       # (B,64,3), (B,64,256)

        # SA4: global (1 point), 1024-dim features
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)       # (B,1,3), (B,1,1024)

        # ---------------- Decoder (upsampling) ----------------

        # FP4: upsample from level 4 -> level 3
        # xyz1 = l3_xyz (targets, 64 pts), xyz2 = l4_xyz (source, 1 pt)
        # points1 = l3_points^T (B,256,64), points2 = l4_points^T (B,1024,1)
        l3_points_fp = self.fp4(
            l3_xyz, l4_xyz,
            l3_points.transpose(1, 2),
            l4_points.transpose(1, 2),
        )  # (B,256,64)

        # FP3: upsample from level 3 -> level 2
        # xyz1 = l2_xyz (256 pts), xyz2 = l3_xyz (64 pts)
        # points1 = l2_points^T (B,128,256), points2 = l3_points_fp (B,256,64)
        l2_points_fp = self.fp3(
            l2_xyz, l3_xyz,
            l2_points.transpose(1, 2),
            l3_points_fp,
        )  # (B,256,256)

        # FP2: upsample from level 2 -> level 1
        # xyz1 = l1_xyz (1024 pts), xyz2 = l2_xyz (256 pts)
        # points1 = l1_points^T (B,64,1024), points2 = l2_points_fp (B,256,256)
        l1_points_fp = self.fp2(
            l1_xyz, l2_xyz,
            l1_points.transpose(1, 2),
            l2_points_fp,
        )  # (B,128,1024)

        # FP1: upsample from level 1 -> level 0 (original N pts)
        # xyz1 = l0_xyz (N pts), xyz2 = l1_xyz (1024 pts)
        # points1 = original input features x (B,C,N),
        # points2 = l1_points_fp (B,128,1024)
        l0_feats = x  # (B,C,N) original features as skip-connection
        l0_points_fp = self.fp1(
            l0_xyz, l1_xyz,
            l0_feats,
            l1_points_fp,
        )  # (B,128,N)

        # Final per-point classifier
        logits = self.seg_head(l0_points_fp)  # (B,num_classes,N)

        # Return logits + two inner feature maps you might want to visualize later
        return logits, l4_points, l3_points

    def compute_loss(self, points, labels):
        """
        Compute segmentation loss and basic metrics.

        Args:
            points: (B, C, N) input to the network (same as x in forward).
            labels: (B, N)    ground-truth class index per point.

        Returns:
            loss: scalar tensor (cross-entropy)
            logs: dict with detached values for easy logging/printing.
        """
        logits, _, _ = self(points)  # logits: (B,num_classes,N)

        # Standard per-point cross-entropy
        loss = F.cross_entropy(logits, labels.to(points.device))

        # Accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=1)      # (B,N)
            acc = (preds == labels).float().mean()

        # Note: you're currently using the same `loss` for ce and reg here
        # (no explicit regularizer in this PointNet++ version).
        logs = {
            "loss": loss.detach(),
            "acc": acc.detach(),
            "ce": loss.detach(),
            "reg": loss.detach(),  # placeholder if you later add a real reg term
        }
        return loss, logs
