import torch.nn as nn
from segmentation_models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstraction
import torch.nn.functional as F
import torch

class PointNet2Segmentation(nn.Module):
    def __init__(self, num_classes, input_dim=3):
        """
        input_dim: number of input channels (3 for xyz, 3+ for normals, etc.)
        """
        super().__init__()

        # ----- Encoder: Set Abstraction layers -----
        # SA1
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=0,
            mlp=[32, 32, 64],
            group_all=False,
        )
        # SA2
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64,
            mlp=[64, 64, 128],
            group_all=False,
        )
        # SA3
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128,
            mlp=[128, 128, 256],
            group_all=False,
        )
        # SA4 (global)
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256,
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # ----- Decoder: Feature Propagation layers -----
        self.fp4 = PointNetFeaturePropagation(
            in_channel=1024 + 256,  # SA3 feat + SA4 global feat
            mlp=[256, 256],
        )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=256 + 128,   # SA2 feat + upsampled
            mlp=[256, 256],
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 64,    # SA1 feat + upsampled
            mlp=[256, 128],
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + input_dim,  # input xyz-features + upsampled
            mlp=[128, 128, 128],
        )

        # final per-point classifier
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x):
        """
        x: (B, C, N), typically C=3 (xyz) or more
        returns: logits (B, num_classes, N)
        """
        B, C, N = x.shape

        xyz = x[:, :3, :].transpose(1, 2)   # (B,N,3)
        if C > 3:
            points = x[:, 3:, :].transpose(1, 2)  # (B,N,C-3)
        else:
            points = None

        # Encoder
        l0_xyz, l0_points = xyz, points
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # (B,1024,3), (B,1024,64)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B,256,3), (B,256,128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B,64,3), (B,64,256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B,1,3),  (B,1,1024)

        # Decoder
        l3_points_fp = self.fp4(l3_xyz, l4_xyz, l3_points.transpose(1,2), l4_points.transpose(1,2))
        # l3_points_fp: (B,256,64)

        l2_points_fp = self.fp3(
            l2_xyz, l3_xyz,
            l2_points.transpose(1,2),
            l3_points_fp
        )  # (B,256,256)

        l1_points_fp = self.fp2(
            l1_xyz, l2_xyz,
            l1_points.transpose(1,2),
            l2_points_fp
        )  # (B,128,1024)

        # for level 0 we pass xyz (C=input_dim) as features
        l0_feats = x  # (B,C,N)
        l0_points_fp = self.fp1(
            l0_xyz, l1_xyz,
            l0_feats,
            l1_points_fp
        )  # (B,128,N)

        logits = self.seg_head(l0_points_fp)  # (B,num_classes,N)
        return logits,l4_points, l3_points
    
    def compute_loss(self, points, labels):
        """
        points: (B, C, N)
        labels: (B, N)
        Returns:
            loss: scalar tensor
            logs: dict with components (for printing)
        """
        logits,_,_ = self(points)

        loss = F.cross_entropy(logits, labels.to(points.device))

        # accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        logs = {
            "loss": loss.detach(),
            "acc": acc.detach(),
            "ce": loss.detach(),
            "reg": loss.detach(),

        }
        return loss, logs
