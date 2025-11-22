# scripts/train_pointnet.py
import torch
from torch.utils.data import DataLoader

from segmentation_models import PointNetSegmentation
from datasets import AirplaneSurfaceDataset
from segmentation_models._plots import plot_gt_vs_pred
from training.trainer import evaluate, fit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AirplaneSurfaceDataset(num_samples=4000, random_rotation=True)
    val_dataset   = AirplaneSurfaceDataset(num_samples=500, random_rotation=True)

    train_dataset.view_sample(0)  # Optional: visualize a sample
    plot_gt_vs_pred(
        model=PointNetSegmentation(num_classes=5, feature_transform=True, input_dim=3).to(device),
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PointNetSegmentation(num_classes=5, feature_transform=True, input_dim=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    _= evaluate(
        model,
        val_loader,
        device,
        0
    ) # Initial evaluation before training


    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        checkpoint_path="checkpoints/best_model_pointnet_airplanes.pt",
        es_min_delta=1e-4
    )

    plot_gt_vs_pred(
        model=model,
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

if __name__ == "__main__":
    main()
