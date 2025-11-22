import torch
from datasets.airplane_dataset import AirplaneSurfaceDataset
from segmentation_models._plots import overlay_pointnet_transforms, plot_gt_vs_pred, visualize_pointnet_transforms
from segmentation_models.pointnet import PointNetSegmentation
from training.trainer import evaluate
from torch.utils.data import DataLoader



val_dataset   = AirplaneSurfaceDataset(num_samples=500, random_rotation=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PointNetSegmentation(num_classes=5, feature_transform=True, input_dim=3)

checkpoint = torch.load("checkpoints/best_model_pointnet_airplanes.pt", map_location=device)

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

# say you want to use the validation set of airplanes / cubes / cones
overlay_pointnet_transforms(
    model=model,
    dataset=val_dataset,
    device=device,
    num_examples=2,     # how many shapes to overlay
    # indices=[0, 10, 20, 30, 40],  # optional: choose which ones
)