from matplotlib import pyplot as plt
import numpy as np

def _sample_airplane_surface(
    num_body_points=1024,
    num_wing_points=512,
    num_tailplane_points=256,
    num_fin_points=256,
    add_noise=True,
    noise_sigma=0.005,
    random_rotation=False,
):
    """
    Synthetic airplane made of:
      0: fuselage (cylinder with caps)
      1: left wing (rectangle in x-y)
      2: right wing (rectangle in x-y)
      3: horizontal tailplane (rectangle in x-y, near tail)
      4: vertical fin (rectangle in x-z, near tail)

    Returns:
        points: (N, 3) float32
        labels: (N,) int64
    """

    # -------------------------
    # Random overall dimensions
    # -------------------------
    L = np.random.uniform(3.0, 5.0)   # fuselage length along x
    R = np.random.uniform(0.2, 0.4)   # fuselage radius

    # Wing
    wing_span = np.random.uniform(1.0, 2.0)    # half-span beyond fuselage
    wing_chord = np.random.uniform(1.0, 1.8)   # chord length along x
    x_wing = np.random.uniform(-0.2 * L, 0.2 * L)  # center of wing chord

    # Tailplane (horizontal)
    tail_span = np.random.uniform(0.5, 1.2)
    tail_chord = np.random.uniform(0.5, 1.2)
    x_tail = np.random.uniform(-0.6 * L, -0.4 * L)  # near the back
    z_tail = np.random.uniform(0.0, 0.3)           # slightly above fuselage center

    # Vertical fin
    fin_height = np.random.uniform(0.4, 0.8)
    fin_chord = np.random.uniform(0.4, 0.8)
    x_fin = x_tail
    y_fin = 0.0  # centered

    points_list = []
    labels_list = []

    # -------------------------
    # 0) Fuselage: cylinder + caps
    # -------------------------
    n_body_side = int(num_body_points * 0.7)
    n_body_caps = num_body_points - n_body_side
    n_cap_front = n_body_caps // 2
    n_cap_back = n_body_caps - n_cap_front

    # Side surface (uniform in x and theta)
    x_side = np.random.uniform(-L / 2, L / 2, size=n_body_side)
    theta_side = 2 * np.pi * np.random.rand(n_body_side)
    y_side = R * np.cos(theta_side)
    z_side = R * np.sin(theta_side)
    body_side = np.stack([x_side, y_side, z_side], axis=1)

    # Front cap (disk at x = +L/2)
    u_front = np.random.rand(n_cap_front)
    r_front = R * np.sqrt(u_front)
    theta_front = 2 * np.pi * np.random.rand(n_cap_front)
    y_front = r_front * np.cos(theta_front)
    z_front = r_front * np.sin(theta_front)
    x_front = np.full(n_cap_front, L / 2)
    body_front = np.stack([x_front, y_front, z_front], axis=1)

    # Back cap (disk at x = -L/2)
    u_back = np.random.rand(n_cap_back)
    r_back = R * np.sqrt(u_back)
    theta_back = 2 * np.pi * np.random.rand(n_cap_back)
    y_back = r_back * np.cos(theta_back)
    z_back = r_back * np.sin(theta_back)
    x_back = np.full(n_cap_back, -L / 2)
    body_back = np.stack([x_back, y_back, z_back], axis=1)

    body_points = np.concatenate([body_side, body_front, body_back], axis=0)
    body_labels = np.zeros(body_points.shape[0], dtype=np.int64)  # class 0

    points_list.append(body_points)
    labels_list.append(body_labels)

    # -------------------------
    # 1) Wings: left & right (flat rectangles in x-y at z ~ 0)
    # -------------------------
    def sample_wing(side, n_points):
        """
        side: 'left' or 'right'
        """
        if side == 'right':
            y_min = R
            y_max = R + wing_span
        else:  # left
            y_min = -(R + wing_span)
            y_max = -R

        x_min = x_wing - wing_chord / 2
        x_max = x_wing + wing_chord / 2

        x = np.random.uniform(x_min, x_max, size=n_points)
        y = np.random.uniform(y_min, y_max, size=n_points)
        z = np.zeros(n_points)  # at z = 0 (approx wing plane)

        return np.stack([x, y, z], axis=1)

    # split wing points between left and right
    n_wing_left = num_wing_points // 2
    n_wing_right = num_wing_points - n_wing_left

    right_wing_points = sample_wing('right', n_wing_right)
    left_wing_points = sample_wing('left', n_wing_left)

    right_labels = np.full(n_wing_right, 1, dtype=np.int64)  # class 1
    left_labels  = np.full(n_wing_left, 1, dtype=np.int64)   # class 1

    points_list.extend([left_wing_points, right_wing_points])
    labels_list.extend([left_labels, right_labels])

    # -------------------------
    # 2) Horizontal tailplane (rectangle in x-y at z = z_tail)
    # -------------------------
    x_min_tail = x_tail - tail_chord / 2
    x_max_tail = x_tail + tail_chord / 2
    y_min_tail = -tail_span
    y_max_tail = tail_span

    x_tail_pts = np.random.uniform(x_min_tail, x_max_tail, size=num_tailplane_points)
    y_tail_pts = np.random.uniform(y_min_tail, y_max_tail, size=num_tailplane_points)
    z_tail_pts = np.full(num_tailplane_points, z_tail)

    tailplane_points = np.stack([x_tail_pts, y_tail_pts, z_tail_pts], axis=1)
    tailplane_labels = np.full(num_tailplane_points, 2, dtype=np.int64)  # class 2

    points_list.append(tailplane_points)
    labels_list.append(tailplane_labels)

    # -------------------------
    # 3) Vertical fin (rectangle in x-z at y = 0)
    # -------------------------
    x_min_fin = x_fin - fin_chord / 2
    x_max_fin = x_fin + fin_chord / 2
    z_min_fin = 0.0
    z_max_fin = fin_height

    x_fin_pts = np.random.uniform(x_min_fin, x_max_fin, size=num_fin_points)
    z_fin_pts = np.random.uniform(z_min_fin, z_max_fin, size=num_fin_points)
    y_fin_pts = np.full(num_fin_points, y_fin)

    fin_points = np.stack([x_fin_pts, y_fin_pts, z_fin_pts], axis=1)
    fin_labels = np.full(num_fin_points, 3, dtype=np.int64)  # class 3

    points_list.append(fin_points)
    labels_list.append(fin_labels)

    # -------------------------
    # Combine all parts
    # -------------------------
    points = np.concatenate(points_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # -------------------------
    # Optional: random global rotation
    # -------------------------
    if random_rotation:
        # Random Euler angles (xyz)
        angles = np.random.uniform(0.0, 2.0 * np.pi, size=3)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                       [ 0,  1, 0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]])

        R = Rz @ Ry @ Rx
        points = points @ R.T

    # Optional small noise
    if add_noise:
        points = points + np.random.normal(scale=noise_sigma, size=points.shape)

    return points.astype(np.float32), labels.astype(np.int64)

import torch
from torch.utils.data import Dataset

class AirplaneSurfaceDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_body_points=1024,
        num_wing_points=512,
        num_tailplane_points=256,
        num_fin_points=256,
        add_noise=True,
        noise_sigma=0.005,
        random_rotation=False,
        preprocess=False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_body_points = num_body_points
        self.num_wing_points = num_wing_points
        self.num_tailplane_points = num_tailplane_points
        self.num_fin_points = num_fin_points
        self.add_noise = add_noise
        self.noise_sigma = noise_sigma
        self.face_colors = np.array([
            [1.0, 0.0, 0.0],  # 0: +X -> red
            [0.0, 1.0, 0.0],  # 1: -X -> green
            [0.0, 0.0, 1.0],  # 2: +Y -> blue
            [1.0, 1.0, 0.0],  # 3: -Y -> yellow
            [1.0, 0.0, 1.0],  # 4: +Z -> magenta
            [0.0, 1.0, 1.0],  # 5: -Z -> cyan
        ])
        self.num_classes = 5
        self.random_rotation = random_rotation
        self.preprocess_flag = preprocess

    def preprocess(self, points):
        # points: (3, N)

        # 1. center
        centroid = points.mean(axis=1, keepdims=True)
        points = points - centroid

        # 2. scale to unit sphere
        scale = np.max(np.linalg.norm(points, axis=0))
        points = points / scale

        return points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pts_np, labels_np = _sample_airplane_surface(
            num_body_points=self.num_body_points,
            num_wing_points=self.num_wing_points,
            num_tailplane_points=self.num_tailplane_points,
            num_fin_points=self.num_fin_points,
            add_noise=self.add_noise,
            noise_sigma=self.noise_sigma,
            random_rotation=self.random_rotation,
        )

        if self.preprocess_flag:
            pts_np = self.preprocess(pts_np)  # (N, 3)

        pts = torch.from_numpy(pts_np).transpose(0, 1)  # (3, N)
        labels = torch.from_numpy(labels_np)            # (N,)
        return pts, labels
    
import torch
from torch.utils.data import Dataset

# assuming you already have this somewhere:
# from data.airplane import sample_airplane_surface

class PrecomputedAirplaneSurfaceDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_body_points=1024,
        num_wing_points=512,
        num_tailplane_points=256,
        num_fin_points=256,
        add_noise=True,
        noise_sigma=0.005,
        random_rotation=False,
        seed: int | None = None,
        preprocess=False,
    ):
        """
        Precompute a fixed set of synthetic airplanes.

        After construction, __getitem__ is deterministic and returns
        the same point clouds every epoch.
        """
        super().__init__()
        self.num_samples = num_samples
        self.face_colors = np.array([
            [1.0, 0.0, 0.0],  # 0: +X -> red
            [0.0, 1.0, 0.0],  # 1: -X -> green
            [0.0, 0.0, 1.0],  # 2: +Y -> blue
            [1.0, 1.0, 0.0],  # 3: -Y -> yellow
            [1.0, 0.0, 1.0],  # 4: +Z -> magenta
            [0.0, 1.0, 1.0],  # 5: -Z -> cyan
        ])
        self.num_classes = 4
        self.preprocess_flag = preprocess

        # optional: make generation reproducible
        rng_state = None
        if seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)

        points_list = []
        labels_list = []

        for _ in range(num_samples):
            pts_np, labels_np = _sample_airplane_surface(
                num_body_points=num_body_points,
                num_wing_points=num_wing_points,
                num_tailplane_points=num_tailplane_points,
                num_fin_points=num_fin_points,
                add_noise=add_noise,
                noise_sigma=noise_sigma,
                random_rotation=random_rotation,
            )
            # pts_np: (N, 3), labels_np: (N,)

            if self.preprocess_flag:
                pts_np = self.preprocess(pts_np)  # (N, 3)

            pts = torch.from_numpy(pts_np).transpose(0, 1)  # (3, N)
            labels = torch.from_numpy(labels_np)            # (N,)

            points_list.append(pts)
            labels_list.append(labels)

        # stack into big tensors for fast access
        self.points = torch.stack(points_list, dim=0)   # (num_samples, 3, N)
        self.labels = torch.stack(labels_list, dim=0)   # (num_samples, N)

        # restore rng if we changed it
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)

    def preprocess(self, points):
        # points: (3, N)

        # 1. center
        centroid = points.mean(axis=1, keepdims=True)
        points = points - centroid

        # 2. scale to unit sphere
        scale = np.max(np.linalg.norm(points, axis=0))
        points = points / scale

        return points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Just slice the precomputed tensors
        return self.points[idx], self.labels[idx]

    
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