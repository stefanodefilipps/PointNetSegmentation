import torch
from datasets.airplane_dataset import AirplaneSurfaceDataset
from datasets.shapenetparts import ShapeNetPartDataset
from segmentation_models._plots import plot_gt_vs_pred
from segmentation_models.pointnet2 import PointNet2Segmentation
from training.trainer import evaluate
from torch.utils.data import DataLoader



root = "/home/stefano/stefano_repo/data/archive/PartAnnotation"  # folder containing shapenetcore_partanno_segmentation_benchmark_v0


val_dataset = ShapeNetPartDataset(
    root=root,
    split="test",
    num_points=2048,
    class_choice=["Airplane"],   # or None for all
    normalize=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PointNet2Segmentation(num_classes=val_dataset.num_seg_classes, input_dim=3)

checkpoint = torch.load("checkpoints/best_model_pointnet2_airplanes.pt", map_location=device)

model.load_state_dict(checkpoint["model_state"])
start_epoch = checkpoint["epoch"]
best_val_loss = checkpoint["best_val_loss"]
model.to(device)
model.eval()

val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)    

evaluate(
    model,
    val_loader,
    device,
    start_epoch
)

print("Restored model from epoch", start_epoch, "with best val loss", best_val_loss)

plot_gt_vs_pred(
    model=model,
    device=device,
    dataset=val_dataset,
    idx=0
)  # Optional: visualize predictions before training