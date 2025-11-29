# scripts/train_pointnet.py
import torch
from torch.utils.data import DataLoader

from datasets.airplane_dataset import PrecomputedAirplaneSurfaceDataset
from datasets.shapenetparts import ShapeNetPartDataset
from segmentation_models._plots import plot_gt_vs_pred
from segmentation_models.pointnet2 import PointNet2Segmentation
from training.trainer import evaluate, fit

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_dataset = PrecomputedAirplaneSurfaceDataset(num_samples=4000, random_rotation=True)
    # val_dataset   = PrecomputedAirplaneSurfaceDataset(num_samples=500, random_rotation=False)

    root = "/home/stefano/stefano_repo/data/archive/PartAnnotation"  # folder containing shapenetcore_partanno_segmentation_benchmark_v0

    train_dataset = ShapeNetPartDataset(
        root=root,
        split="train",
        num_points=2048,
        class_choice=["Chair"],   # or None for all
        normalize=False,
    )

    val_dataset = ShapeNetPartDataset(
        root=root,
        split="test",
        num_points=2048,
        class_choice=["Airplane"],   # or None for all
        normalize=False,
    )

    train_dataset.view_sample(0)  # Optional: visualize a sample
    plot_gt_vs_pred(
        model=PointNet2Segmentation(num_classes=train_dataset.num_seg_classes, input_dim=3).to(device),
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = PointNet2Segmentation(num_classes=train_dataset.num_seg_classes, input_dim=3).to(device)

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
        checkpoint_path="checkpoints/best_model_pointnet2_chairs.pt",
        es_min_delta=0.0,
        es_patience=50
    )

    plot_gt_vs_pred(
        model=model,
        device=device,
        dataset=train_dataset,
        idx=0
    )  # Optional: visualize predictions before training

if __name__ == "__main__":
    main()
