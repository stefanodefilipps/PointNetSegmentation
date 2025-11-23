# PointNet & PointNet++: Architecture Overview and Differences

This repository includes implementations of **PointNet**, **PointNet++**, and **DGCNN** for point cloud segmentation. This README explains the first two: PointNet and PointNet++, how they work, and why PointNet++ improves on the original PointNet architecture.

---

# 📌 PointNet

## 🧱 Architecture Overview

PointNet has a **simple and elegant architecture** built around three main components:

### 1. **Shared MLP (per-point feature extractor)**

Each point ( (x, y, z) ) is passed independently through several fully connected layers:

```
(3) → 64 → 64 → 64 → 128 → 1024
```

These layers are applied *point-wise* and identically to all points, ensuring permutation invariance.

### 2. **Symmetric Aggregation (Max Pooling)**

A global feature is produced by applying **max pooling across all points**:

```
global_feature = max_pool({point_features})  # → (1024,)
```

This step makes the network invariant to the order of points.

### 3. **T-Net (Spatial Transformer Networks)**

Two learned transformations:

* **Input Transform (3×3)** — canonicalizes raw points
* **Feature Transform (64×64)** — canonicalizes point features

Both are small PointNet-like networks predicting a transformation matrix.

### 4. **Segmentation Head**

For segmentation, the global feature is concatenated to per-point features and passed through another MLP:

```
(1024 + point_feat) → 512 → 256 → num_classes
```

PointNet’s structure is essentially:

```
 points → shared MLP → global max-pool → global feature
             │                          │
             └─ concatenated per-point ─┘
                         ↓
                segmentation MLP
```

PointNet was introduced in the paper:
**"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al., 2017)**.

It was the first neural architecture to operate **directly on unordered point clouds** without converting them to voxels or meshes.

## 🔧 Key Ideas

### 1. **Shared MLP on each point**

Each point goes through the same multilayer perceptron (MLP).
This ensures:

* permutation invariance,
* weight sharing,
* treating points independently.

### 2. **Symmetric aggregation function (Max Pooling)**

After processing points individually, PointNet aggregates all point features using **global max pooling**.

This produces a single global feature representing the entire object.

### 3. **T-Net (Learned Spatial Transform)**

PointNet learns a **3×3 transformation matrix** that canonicalizes the input point cloud.

This helps overcome variations in rotation/pose.

There is also an optional feature-transform T-Net (64×64) for intermediate features.

### 4. **Per-point segmentation head**

For segmentation tasks:

* The global feature is concatenated back with per-point features.
* A final point-wise MLP predicts class logits per point.

## 2. **Feature Propagation (FP) Layers**

These layers upsample features back to the original point resolution.

They use:

* **Inverse-distance weighted interpolation** to project coarse features to denser layers
* **Skip connections** from encoder layers

Each FP layer is a 1×1 convolution (MLP) refining the interpolated features.

The decoder structure mirrors the encoder:

```
1 → 64 → 256 → 1024 → original N points
```

---

## ⚙️ Strengths

* Extremely simple and fast
* Order-invariant
* Works well for classification

## ⚠️ Weaknesses

* **No local geometry awareness** (points treated independently)
* Sensitive to non-uniform sampling
* Struggles with fine-grained details and small structures

---

# 📌 PointNet++

## 🧱 Architecture Overview

PointNet++ extends PointNet by introducing a **hierarchical feature learning** structure, similar to CNNs.

It has two major modules:

---

## 1. **Set Abstraction (SA) Layers**

Each SA layer performs three steps:

### **✓ Sampling**

Uses **Farthest Point Sampling (FPS)** to pick a subset of well‑distributed points.

### **✓ Grouping**

For each sampled point, neighbors inside a radius are gathered:

```
ball_query(center, radius) → local_patch
```

### **✓ Local Feature Learning (mini‑PointNet)**

Each local patch is passed through a tiny PointNet:

```
MLP → max-pool → local_feature
```

This extracts **local geometric patterns**, something original PointNet lacked.

A stack of SA layers builds a hierarchy:

```
N points → 1024 → 256 → 64 → 1
```

Introduced in:
**"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (Qi et al., 2017)**

PointNet++ addresses PointNet's biggest limitation: **lack of local feature extraction**.

## 🔧 Key Ideas

### 1. **Hierarchical Set Abstraction (SA)**

Similar to CNNs building spatial hierarchies, PointNet++:

* samples a subset of points using **Farthest Point Sampling (FPS)**,
* groups neighbors using **ball query**,
* computes features on each group using a **mini PointNet**.

Each Set Abstraction layer reduces the number of points but increases feature richness.

### 2. **Local Neighborhoods**

PointNet++ explicitly models **local structures**, something PointNet could not do.

### 3. **Multi-Scale Grouping (MSG)** & **Multi-Resolution Grouping (MRG)**

To handle non-uniform point density:

* MSG builds several neighborhoods with different radii.
* MRG mixes features from different resolutions.

### 4. **Feature Propagation (FP)**

The decoder upsamples features back to the original points using:

* inverse-distance weighted interpolation,
* skip connections from early layers.

## ⚙️ Strengths

* Learns **local + global** geometric features
* Robust to varying densities
* Excellent for segmentation: part-level, semantic, etc.

## ⚠️ Weaknesses

* Slower and more complex than PointNet
* Requires careful tuning of radii and sampling

---

# 🔍 PointNet vs PointNet++: Summary Table

| Aspect                        | PointNet | PointNet++                         |
| ----------------------------- | -------- | ---------------------------------- |
| Local geometry                | ❌ None   | ✅ Yes (ball query + mini-PointNet) |
| Hierarchy                     | ❌ Flat   | ✅ Multi-scale hierarchical         |
| Robust to non-uniform density | ❌ No     | ✅ Yes (MSG/MRG)                    |
| Sampling                      | ❌ None   | ✅ Farthest Point Sampling          |
| Segmentation performance      | ⭐ Basic  | ⭐⭐⭐ Excellent                      |
| Speed                         | 🚀 Fast  | ⏳ Slower                           |

---

# 📁 Repository Structure Overview

```
segmentation_models/
│
├── pointnet.py              # PointNet implementation
├── pointnet2.py             # PointNet++ implementation
├── pointnet2_utils.py       # FPS, Ball Query, Set Abstraction, FP
├── dgcnn.py                 # DGCNN implementation
└── _plots.py                # Visualization tools
```

---

# 🧠 When Should You Use Which?

### Use **PointNet** when:

* you need simplicity
* classification
* fast inference is more important than accuracy

### Use **PointNet++** when:

* segmentation matters
* your shapes have fine details
* point clouds have uneven density

### Use **DGCNN** when:

* you want state-of-the-art segmentation
* you prefer dynamic graph neighborhoods
