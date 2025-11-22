import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_transform_regularizer(trans):
    """
    trans: (B, K, K)
    returns scalar regularization loss
    """
    if trans is None:
        return torch.tensor(0.0, device=next(iter(trans.parameters())).device)

    B, K, _ = trans.shape
    I = torch.eye(K, device=trans.device).unsqueeze(0).expand(B, -1, -1)
    # || A A^T - I ||_F
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize last layer biases to identity
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        # x: (B, k, N)
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        x = torch.max(x, 2)[0]  # global max pool -> (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)

        # Add identity
        id_matrix = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + id_matrix  # (B, k*k)

        x = x.view(-1, self.k, self.k)  # (B, k, k)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=False, input_dim=3):
        super().__init__()
        self.feature_transform = feature_transform

        self.input_tnet = TNet(k=input_dim)

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.feature_tnet = TNet(k=64)
        else:
            self.feature_tnet = None

    def forward(self, x):
        """
        x: (B, C, N) where typically C=3 (xyz) or more if you add features
        """
        B, C, N = x.shape

        # Input transform
        trans = self.input_tnet(x)  # (B, C, C)
        x = torch.bmm(trans, x)     # (B, C, N)

        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # Optional feature transform on 64-dim features
        if self.feature_transform:
            f_trans = self.feature_tnet(x)       # (B, 64, 64)
            x = torch.bmm(f_trans, x)           # (B, 64, N)
        else:
            f_trans = None

        point_features = x.clone()  # (B, 64, N)

        x = F.relu(self.bn2(self.conv2(x)))      # (B, 128, N)
        x = self.bn3(self.conv3(x))              # (B, 1024, N)

        # Global max pool
        global_feat = torch.max(x, 2)[0]         # (B, 1024)

        return global_feat, point_features, trans, f_trans
    

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes, feature_transform=False, input_dim=3, lambda_feat_reg=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.lambda_feat_reg = lambda_feat_reg

        self.encoder = PointNetEncoder(
            feature_transform=feature_transform,
            input_dim=input_dim,
        )

        # The original PointNet concatenates:
        # - global feature (1024) repeated per point
        # - point-wise 64-dim features from early layer
        # So input to seg head is 1024 + 64 = 1088
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        x: (B, C, N)
        returns: (B, num_classes, N)
        """
        B, C, N = x.shape

        global_feat, point_feat, trans, f_trans = self.encoder(x)
        # global_feat: (B, 1024)
        # point_feat: (B, 64, N)

        # Repeat global feature per point
        global_expanded = global_feat.unsqueeze(-1).repeat(1, 1, N)  # (B, 1024, N)

        # Concatenate with point-wise features
        x = torch.cat([point_feat, global_expanded], dim=1)  # (B, 1088, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv4(x)  # (B, num_classes, N)

        return x, trans, f_trans
    
    def compute_loss(self, points, labels):
        """
        points: (B, C, N)
        labels: (B, N)
        Returns:
            loss: scalar tensor
            logs: dict with components (for printing)
        """
        logits, _, f_trans = self(points)

        ce = F.cross_entropy(logits, labels)  # main segmentation loss

        reg = torch.tensor(0.0, device=points.device)
        if self.feature_transform and f_trans is not None and self.lambda_feat_reg > 0:
            reg = feature_transform_regularizer(f_trans) * self.lambda_feat_reg

        loss = ce + reg

        # accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()

        logs = {
            "loss": loss.detach(),
            "ce": ce.detach(),
            "reg": reg.detach(),
            "acc": acc.detach(),
        }
        return loss, logs
