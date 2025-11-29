import os
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset


import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetPartDataset(Dataset):
    """
    ShapeNetPart-style dataset with:
      - .pts point files
      - .seg segmentation files
      - extra substructure inside points_label/ by part.

    Expected structure:

      root/
        shapenetcore_partanno_segmentation_benchmark_v0/
          synsetoffset2category.txt
          02691156/                # Airplane
            points/
              shape_0001.pts
              shape_0002.pts
              ...
            points_label/
              shape_0001.seg       (optional combined labels)
              shape_0002.seg       (optional)
              ...
              part_0/
                shape_0001.seg     (binary, 0/1 per point)
                shape_0002.seg
              part_1/
                shape_0001.seg
                ...
              ...

    If a combined .seg exists directly under points_label/, we use it.
    Otherwise we:
      - read all subfolders under points_label/ (each is a part)
      - read shape_XXXX.seg from each (0/1 masks)
      - stack into (num_parts, N), then argmax over parts to get part id per point.

    Returns:
        points: (N,3) float32  – normalized
        seg:    (N,) int64     – part label (0..num_parts-1)
        cls:    int64          – category index (0..num_categories-1)
        shape_id: str          – useful for debugging/visualization
    """

    def __init__(
        self,
        root,
        split="train",          # 'train', 'val', 'test'
        num_points=2048,
        class_choice=None,      # e.g. ['Airplane', 'Chair'] or None for all
        normalize=True,
        augment=False,
        train_val_test_ratio=(0.8, 0.1, 0.1),
        random_seed=42,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"
        self.root = root
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.train_val_test_ratio = train_val_test_ratio
        self.random_seed = random_seed

        # ---------- base dirs ----------
        self.base_dir = self.root
        self.catfile = osp.join(self.base_dir, "synsetoffset2category.txt")

        # ---------- read category mapping ----------
        self.cat2id = {}
        self.id2cat = {}
        with open(self.catfile, "r") as f:
            for line in f:
                cat_name, cat_id = line.strip().split()
                self.cat2id[cat_name] = cat_id
                self.id2cat[cat_id] = cat_name

        # filter by object class (category) if requested
        if class_choice is not None:
            class_choice = [c.capitalize() for c in class_choice]
            self.cat2id = {k: v for k, v in self.cat2id.items() if k in class_choice}

        self.classes = sorted(self.cat2id.keys())
        self.class_to_idx = {cat: i for i, cat in enumerate(self.classes)}

        # ---------- build full datapath ----------
        # each entry: (cat_name, shape_id, pts_path, labels_root)
        self.datapath = []
        for cat_name in self.classes:
            cat_id = self.cat2id[cat_name]
            cat_dir = osp.join(self.base_dir, cat_id)
            pts_dir = osp.join(cat_dir, "points")
            labels_root = osp.join(cat_dir, "points_label")

            if not osp.isdir(pts_dir):
                continue

            pts_files = sorted(f for f in os.listdir(pts_dir) if f.endswith(".pts"))
            for fn in pts_files:
                shape_id = osp.splitext(fn)[0]
                pts_path = osp.join(pts_dir, fn)
                # points_label may contain combined seg + subfolders
                self.datapath.append((cat_name, shape_id, pts_path, labels_root))

        # ---------- NEW: compute global max seg label ----------
        self.max_seg_label = self._compute_max_seg_label()
        self.num_seg_classes = self.max_seg_label + 1
        print(f"[ShapeNetPartCustomDataset] max_seg_label={self.max_seg_label}, "
              f"num_seg_classes={self.num_seg_classes}")

        full_indices = np.arange(len(self.datapath))

        # ---------- build internal train/val/test split ----------
        self.indices = self._build_internal_split(full_indices)

        print(
            f"[ShapeNetPartCustomDataset] split={split}, "
            f"categories={self.classes}, "
            f"samples={len(self.indices)}"
        )

    def _compute_max_seg_label(self):
        """
        Scan all shapes once and find the maximum segmentation label.

        Uses the same _load_seg_labels(...) that __getitem__ uses,
        so it works whether you have combined .seg files or per-part folders.
        """
        max_label = -1
        for (cat_name, shape_id, pts_path, labels_root) in self.datapath:
            # We only need the number of points N to load seg correctly
            pts = self._load_points(pts_path)
            N = pts.shape[0]
            seg = self._load_seg_labels(labels_root, shape_id, N)  # (N,)
            # In case there are negative labels (ignore), filter them out
            if (seg >= 0).any():
                max_label = max(max_label, int(seg[seg >= 0].max()))
        if max_label < 0:
            raise RuntimeError("No valid segmentation labels found when computing max_seg_label.")
        return max_label

    # -----------------------------
    # internal random split
    # -----------------------------
    def _build_internal_split(self, full_indices):
        r_train, r_val, r_test = self.train_val_test_ratio
        total = r_train + r_val + r_test
        r_train /= total
        r_val   /= total
        r_test  /= total

        N = len(full_indices)
        rng = np.random.RandomState(self.random_seed)
        perm = rng.permutation(N)
        full_indices = full_indices[perm]

        n_train = int(N * r_train)
        n_val   = int(N * r_val)
        n_test  = N - n_train - n_val

        train_idx = full_indices[:n_train]
        val_idx   = full_indices[n_train:n_train + n_val]
        test_idx  = full_indices[n_train + n_val:]

        if self.split == "train":
            return train_idx.tolist()
        elif self.split == "val":
            return val_idx.tolist()
        elif self.split == "test":
            return test_idx.tolist()
        else:
            return full_indices.tolist()

    def __len__(self):
        return len(self.indices)

    # -----------------------------
    # simple augment: scale + jitter
    # -----------------------------
    def _augment(self, points):
        scale = np.random.uniform(0.8, 1.25)
        points = points * scale
        sigma = 0.01
        clip = 0.05
        jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
        points = points + jitter
        return points

    # -----------------------------
    # load points (.pts)
    # -----------------------------
    def _load_points(self, pts_path):
        pts = np.loadtxt(pts_path).astype(np.float32)
        # allow (N,3) or (3,N)
        if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
            pts = pts.T
        return pts  # (N,3)

    # -----------------------------
    # load segmentation labels
    #   try combined .seg first
    #   else reconstruct from per-part subfolders
    # -----------------------------
    def _load_seg_labels(self, labels_root, shape_id, num_points):
        # 1) combined seg file directly in points_label/
        combined_path = osp.join(labels_root, shape_id + ".seg")
        if osp.exists(combined_path):
            seg = np.loadtxt(combined_path).astype(np.int64)  # (N,)
            assert seg.shape[0] == num_points, \
                f"Combined seg {combined_path}: len {seg.shape[0]} != {num_points}"
            return seg

        # 2) no combined file → look for subfolders (one per part)
        part_dirs = [
            d for d in os.listdir(labels_root)
            if osp.isdir(osp.join(labels_root, d))
        ]
        part_dirs.sort()  # stable ordering: part_0, part_1, ...

        if len(part_dirs) == 0:
            raise RuntimeError(f"No seg file or part subfolders in {labels_root}")

        seg_matrix = np.zeros((len(part_dirs), num_points), dtype=np.int64)

        for pidx, pname in enumerate(part_dirs):
            seg_path = osp.join(labels_root, pname, shape_id + ".seg")
            if not osp.exists(seg_path):
                continue
            vec = np.loadtxt(seg_path).astype(np.int64)  # (N,)
            assert vec.shape[0] == num_points, \
                f"Part seg {seg_path}: len {vec.shape[0]} != {num_points}"
            seg_matrix[pidx] = vec

        # assume one-hot along parts; argmax -> part id
        seg = seg_matrix.argmax(axis=0)  # (N,)
        return seg

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        cat_name, shape_id, pts_path, labels_root = self.datapath[real_idx]
        cls = self.class_to_idx[cat_name]

        # load points
        points = self._load_points(pts_path)  # (N,3)
        N = points.shape[0]

        # load segmentation labels
        seg = self._load_seg_labels(labels_root, shape_id, N)  # (N,)

        # sampling
        if self.num_points is not None and self.num_points < N:
            choice = np.random.choice(N, self.num_points, replace=False)
            points = points[choice, :]
            seg = seg[choice]
        elif self.num_points is not None and self.num_points > N:
            choice = np.random.choice(N, self.num_points, replace=True)
            points = points[choice, :]
            seg = seg[choice]

        # normalize
        if self.normalize:
            centroid = points.mean(axis=0)
            points = points - centroid
            furthest = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
            points = points / (furthest + 1e-6)

        # augment (train only)
        if self.augment and self.split == "train":
            points = self._augment(points)

        points = torch.from_numpy(points).float()        # (N,3)
        points = points.transpose(1, 0)                  # (3, N)
        seg = torch.from_numpy(seg).long()               # (N,)
        cls = torch.tensor(cls, dtype=torch.long)        # scalar

        return points, seg
    
    def view_sample(self, idx):
        # Get one sample
        pts, labels = self[idx]   # pts: (3, N), labels: (N,)
        pts = pts.numpy().T        # -> (N, 3) for plotting
        labels = labels.numpy()    # -> (N,)

        print("Points shape:", pts.shape)
        print("Labels shape:", labels.shape)
        print("Unique labels:", np.unique(labels))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Create a color map for 6 faces
        cmap = plt.get_cmap('tab10')  # has distinct colors
        colors = cmap(labels / labels.max())  # normalize labels into [0, 1]

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        ax.scatter(x, y, z, c=colors, s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Box Surface Segmentation Example')

        # Make aspect ratio roughly equal
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
        mid = pts.mean(axis=0)
        for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
            axis(m - max_range/2, m + max_range/2)

        plt.show()
