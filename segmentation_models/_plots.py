from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_gt_vs_pred(model, device, dataset, idx=None):
    model.eval()
    if idx is None:
        idx = np.random.randint(0, len(dataset))

    pts, labels = dataset[idx]      # pts: (3, N), labels: (N,)
    N = pts.shape[1]

    with torch.no_grad():
        pts_t = pts.unsqueeze(0).to(device)   # (1, 3, N)
        logits, _, _ = model(pts_t)          # (1, C, N)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (N,)

    pts_np = pts.numpy().T          # (N, 3)
    labels_np = labels.numpy()

    # Map to colors
    gt_colors = dataset.face_colors[labels_np]
    pred_colors = dataset.face_colors[preds]

    # Shared limits
    max_range = (pts_np.max(axis=0) - pts_np.min(axis=0)).max()
    mid = pts_np.mean(axis=0)
    bounds = [
        (mid[0] - max_range/2, mid[0] + max_range/2),
        (mid[1] - max_range/2, mid[1] + max_range/2),
        (mid[2] - max_range/2, mid[2] + max_range/2),
    ]

    fig = plt.figure(figsize=(12, 5))

    # Ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], c=gt_colors, s=5)
    ax1.set_title(f"Ground Truth (idx={idx})")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_xlim(*bounds[0]); ax1.set_ylim(*bounds[1]); ax1.set_zlim(*bounds[2])

    # Predictions
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], c=pred_colors, s=5)
    ax2.set_title("Prediction")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_xlim(*bounds[0]); ax2.set_ylim(*bounds[1]); ax2.set_zlim(*bounds[2])

    plt.tight_layout()
    plt.show()

    # quick per-point accuracy on this sample
    sample_acc = (preds == labels_np).mean()
    print(f"Sample {idx} point accuracy: {sample_acc*100:.2f}%")