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

def visualize_pointnet_transforms(
    model,
    dataset,
    device,
    num_examples: int = 3,
    indices=None,
    use_labels_as_color: bool = True,
):
    """
    Visualize how PointNet's learned input transform (3x3) canonicalizes point clouds.

    Args:
        model:           PointNet segmentation model (forward returns logits, trans, f_trans)
        dataset:         PyTorch Dataset returning (points, labels), points: (3,N)
        device:          torch.device
        num_examples:    how many samples to visualize
        indices:         optional list of dataset indices to visualize; if None, use range(num_examples)
        use_labels_as_color: if True, color points by their segmentation label
    """
    model.eval()

    if indices is None:
        indices = list(range(num_examples))
    else:
        indices = list(indices)[:num_examples]

    for idx in indices:
        pts, labels = dataset[idx]    # pts: (3,N), labels: (N,)
        pts_np = pts.numpy().T        # (N,3)
        labels_np = labels.numpy()    # (N,)

        # run through model to get transform
        with torch.no_grad():
            pts_batch = pts.unsqueeze(0).to(device)  # (1,3,N)
            logits, trans, f_trans = model(pts_batch)

        T = trans[0].detach().cpu().numpy()         # (3,3)

        # apply transform: (N,3) @ (3,3)^T -> (N,3)
        pts_trans = pts_np @ T.T

        # --- optional: compute some stats on T ---
        det = np.linalg.det(T)
        orth_err = np.linalg.norm(T @ T.T - np.eye(3))
        print(f"\nSample idx={idx}")
        print("Transform matrix T:\n", T)
        print(f"det(T) = {det:.4f}, ||T T^T - I||_F = {orth_err:.4e}")

        # choose colors
        if use_labels_as_color:
            # simple colormap based on labels
            cmap = plt.get_cmap("tab10")
            norm_labels = labels_np.astype(float)
            if norm_labels.max() > 0:
                norm_labels = norm_labels / norm_labels.max()
            colors = cmap(norm_labels)
        else:
            colors = "k"

        # for nicer viewing: same axis limits for both plots
        all_pts = np.vstack([pts_np, pts_trans])
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
        mid = all_pts.mean(axis=0)
        xlim = (mid[0] - max_range/2, mid[0] + max_range/2)
        ylim = (mid[1] - max_range/2, mid[1] + max_range/2)
        zlim = (mid[2] - max_range/2, mid[2] + max_range/2)

        # --- plotting ---
        fig = plt.figure(figsize=(10, 5))

        # original
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], c=colors, s=5)
        ax1.set_title(f"Original (idx={idx})")
        ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_zlim(*zlim)
        ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

        # transformed
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.scatter(pts_trans[:, 0], pts_trans[:, 1], pts_trans[:, 2], c=colors, s=5)
        ax2.set_title("After input transform T")
        ax2.set_xlim(*xlim); ax2.set_ylim(*ylim); ax2.set_zlim(*zlim)
        ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

        plt.tight_layout()
        plt.show()

def overlay_pointnet_transforms(
    model,
    dataset,
    device,
    num_examples: int = 5,
    indices=None,
):
    """
    Overlay multiple point clouds before and after PointNet's input transform.

    Args:
        model:       PointNet model with forward(...) -> (logits, trans, f_trans)
        dataset:     dataset[idx] -> (points, labels), points: (3,N)
        device:      torch.device
        num_examples: how many samples to visualize
        indices:     optional list of dataset indices to use (otherwise [0..num_examples-1])
    """
    model.eval()

    if indices is None:
        indices = list(range(num_examples))
    else:
        indices = list(indices)[:num_examples]

    # store all points
    originals = []   # list of (N,3)
    transformed = [] # list of (N,3)

    # colors per-sample
    base_cmap = plt.get_cmap("tab10")

    for i, idx in enumerate(indices):
        pts, labels = dataset[idx]         # pts: (3,N), labels: (N,)
        pts_np = pts.numpy().T             # (N,3)

        with torch.no_grad():
            pts_batch = pts.unsqueeze(0).to(device)  # (1,3,N)
            logits, trans, f_trans = model(pts_batch)

        T = trans[0].detach().cpu().numpy()  # (3,3)

        pts_trans = pts_np @ T.T             # (N,3)

        originals.append(pts_np)
        transformed.append(pts_trans)

        det = np.linalg.det(T)
        orth_err = np.linalg.norm(T @ T.T - np.eye(3))
        print(f"\nSample idx={idx}")
        print("T =\n", T)
        print(f"det(T) = {det:.4f}, ||T T^T - I||_F = {orth_err:.4e}")

    # compute common limits for both plots
    all_pts = np.vstack(originals + transformed)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    mid = all_pts.mean(axis=0)
    xlim = (mid[0] - max_range/2, mid[0] + max_range/2)
    ylim = (mid[1] - max_range/2, mid[1] + max_range/2)
    zlim = (mid[2] - max_range/2, mid[2] + max_range/2)

    # --- plotting ---
    fig = plt.figure(figsize=(12, 5))

    # left: originals
    ax1 = fig.add_subplot(121, projection="3d")
    for i, pts_np in enumerate(originals):
        color = base_cmap(i / max(1, (len(originals) - 1)))
        ax1.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    c=[color], s=4, label=f"s{i}")
    ax1.set_title("Original point clouds (overlaid)")
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.set_zlim(*zlim)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc="upper right", fontsize=8)

    # right: transformed
    ax2 = fig.add_subplot(122, projection="3d")
    for i, pts_np in enumerate(transformed):
        color = base_cmap(i / max(1, (len(transformed) - 1)))
        ax2.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    c=[color], s=4, label=f"s{i}")
    ax2.set_title("After PointNet input transform (overlaid)")
    ax2.set_xlim(*xlim); ax2.set_ylim(*ylim); ax2.set_zlim(*zlim)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()