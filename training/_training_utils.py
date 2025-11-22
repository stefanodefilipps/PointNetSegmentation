import torch
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_accuracy(logits, labels):
    """
    logits: (B, C, N)
    labels: (B, N)
    """
    preds = logits.argmax(dim=1)  # (B, N)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total