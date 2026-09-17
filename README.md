# Inter IIT Tech Meet 15.0 Prepathon — Computer Vision

## Task A: Automated Segmentation of Solar Filaments

This repository contains the implementation for **Task A: Automated Segmentation of Solar Filaments** from the **Inter IIT Tech Meet 15.0 Prepathon — Computer Vision Problem Statement**.

The project develops an end-to-end semantic segmentation pipeline for detecting solar filaments in native-resolution **2048 × 2048 ground-based H-alpha (Hα) observations** from the **MAGFiLO** dataset.

The central engineering objective is to preserve extremely thin filament structures while rejecting visually similar solar-background patterns and maintaining practical inference speed on consumer-grade GPU hardware.

---

## Overview

Solar filaments are elongated structures of relatively cool, dense plasma suspended in the solar corona. In H-alpha observations, they appear as dark absorption structures against the brighter solar disk.

Automated segmentation is challenging because filament structures:

- Can contain **1–3 pixel-wide barbs and threads** at native resolution.
- Occupy only a very small fraction of the total image area.
- Can resemble **sunspots, active regions, chromospheric fibrils, and quiet-Sun granulation**.
- Need to be predicted as **coherent structures** rather than disconnected foreground pixel islands.

The complete pipeline was therefore designed around three objectives:

> **Preserve fine structures → Reject difficult background → Produce coherent masks efficiently**

---

## Dataset

### MAGFiLO

The project uses the **MAGFiLO** dataset containing manually annotated solar filaments from H-alpha observations.

| Property | Value |
|---|---|
| Image resolution | **2048 × 2048** |
| Image type | 8-bit grayscale |
| Channels | 1 |
| Modality | H-alpha (Hα) |
| Annotation | Polygonal filament contours |
| Additional metadata | Filament-related metadata including spine/chirality information |

The native image resolution is retained during evaluation because aggressive downsampling can remove the thin filament structures that the model is expected to segment.

---

## Dataset Engineering

### Annotation Deduplication

During dataset inspection, the raw annotations revealed a mismatch between annotation records and physical image files:

```text
Annotation records : 1,154
Physical images    :   707
```

The discrepancy resulted from multiple annotation records corresponding to filament instances associated with the same physical image.

Instead of modifying the original annotation files, the pipeline resolves this dynamically in memory using Python dictionaries and `dict.setdefault`:

```python
image_dict = {
    img["id"]: img["file_name"]
    for img in coco["images"]
}

ann_dict = {}

for ann in coco["annotations"]:
    ann_dict.setdefault(
        ann["image_id"],
        []
    ).append(ann)
```

This allows fragmented polygon annotations belonging to the same physical image to be grouped and merged during processing without altering the source dataset.

---

## Extreme Foreground Imbalance

The segmentation task is dominated by background pixels.

| Statistic | Value |
|---|---:|
| Mean foreground coverage | **0.3883%** |
| Median foreground coverage | **0.3298%** |
| Minimum coverage | **0.0078%** |
| Maximum coverage | **1.7280%** |
| Approximate background | **99.61%** |
| Total annotated foreground pixels | **18,795,931** |
| Evaluated pixels | **> 4.8 × 10⁹** |

This makes conventional pixel accuracy a poor optimization target. A model predicting background everywhere could exceed 99.6% pixel accuracy while producing an IoU of zero.

The project therefore focuses on:

- Mean IoU
- Precision
- Recall
- Hit Rate
- Miss Rate
- AP@IoU 0
- Multi-scale IoU (MIoU)

---

## Training Strategy

### Why Native-Resolution Crops?

A straightforward approach would resize the full **2048 × 2048** solar disk to **512 × 512**.

This was rejected because such downsampling can erase ultra-thin filament barbs before the image reaches the segmentation network.

Instead, training uses **native-resolution 512 × 512 crops** extracted directly from the original 2048 × 2048 observations.

### Centroid-Biased Sampling

The crop sampler uses a 50/50 strategy:

#### 50% — Filament-Centered Crops

An annotated filament is selected, its polygon centroid is calculated, and the crop is positioned around that location:

```text
x̄ = (1/N) Σ xi
ȳ = (1/N) Σ yi
```

The crop is clipped to valid image boundaries.

This guarantees regular exposure to rare foreground structures.

#### 50% — Background Crops

The remaining samples are selected uniformly across the disk to expose the network to:

- Quiet-Sun granulation
- Limb artifacts
- Empty photosphere
- Dark active regions
- Non-filament structures

This creates a deliberate balance between **foreground preservation** and **background discrimination**.

---

## Data Augmentation

Solar filaments do not have a fixed canonical orientation in image space.

The training pipeline therefore applies tensor-level geometric augmentation:

- Horizontal flip with probability **0.5**
- Vertical flip with probability **0.5**

These transformations are implemented directly inside the PyTorch dataset pipeline.

Validation images remain **unaugmented** and are evaluated at native resolution to maintain benchmark reproducibility.

---

## Dataset Split

The development split is approximately:

```text
80% Training
20% Validation
```

The split is stratified using filament frequency per image with histogram bins approximately corresponding to:

```text
[3, 7, 12, 18]
```

Development-stage mapping:

```text
Training
923 mapped entries
565 physical images

Validation
231 mapped entries
142 physical images
```

The final competition evaluation was performed on **231 full-disk validation images**.

---

# Model Architecture

The architecture evolved through multiple stages before arriving at the final transfer-learning configuration.

## Phase 1 — Plain U-Net

The initial baseline used:

- Four-stage encoder-decoder
- 3 × 3 convolutions
- Batch Normalization
- ReLU activations

The baseline showed limitations in:

- Effective receptive field
- Thin-structure preservation
- Convergence speed
- False-negative rate

---

## Phase 2 — MORDEN-Lite

The second architecture incorporated selected ideas inspired by the MORDEN approach.

### Larger Decoder Kernels

Decoder convolutions were upgraded from:

```text
3 × 3
```

to:

```text
5 × 5
```

with padding = 2.

The larger kernels increase the local receptive field and help trace elongated filament bodies.

### Instance Normalization

`BatchNorm2d` was replaced with:

```python
InstanceNorm2d
```

This is better suited to H-alpha observations with substantial image-to-image differences in:

- Global illumination
- Limb darkening
- Exposure
- Contrast

### LeakyReLU

ReLU was replaced with:

```text
LeakyReLU
negative_slope = 0.01
```

to preserve non-zero gradients for negative activations in the highly sparse foreground setting.

---

# Final Architecture — ResNet34-U-Net

The final model combines an **ImageNet-pretrained ResNet34 encoder** with a U-Net decoder.

```text
ImageNet-pretrained ResNet34
              +
          U-Net Decoder
              ↓
     Single-channel mask
```

This provides pretrained visual representations while retaining U-Net skip connections for spatial reconstruction.

---

## One-Channel ResNet Adaptation

Standard ResNet34 expects a three-channel RGB input, while the solar observations are single-channel H-alpha images.

Instead of duplicating the grayscale input into three channels, the first convolution was adapted from:

```text
64 × 3 × 7 × 7
```

to:

```text
64 × 1 × 7 × 7
```

by collapsing the pretrained channel weights:

\[
W_{\text{new}}
=
\sum_{c=1}^{3}
W_{\text{RGB}}[:,c,:,:]
\]

This allows the pretrained filters to operate directly on single-channel H-alpha imagery while avoiding unnecessary channel duplication under the available VRAM budget.

---

## Decoder

The final decoder progressively reconstructs the full-resolution mask.

```text
Stage 4:
512 → 256
      +
256-channel skip
      ↓
256-channel decoder

Stage 3:
256 → 128
      +
128-channel skip
      ↓
128-channel decoder

Stage 2:
128 → 64
      +
64-channel skip
      ↓
64-channel decoder

Stage 1:
64 → 64
      +
64-channel skip
      ↓
64-channel decoder

Final:
64 → 32
32 → 16
16 → 1
```

Decoder blocks consistently use:

- **5 × 5 convolutions**
- **InstanceNorm2d**
- **LeakyReLU**

---

# Loss Function Evolution

The loss function was progressively modified to handle the extreme foreground sparsity and false-positive behavior.

## BCE + Dice

The initial baseline used:

\[
\mathcal{L}
=
0.5\mathcal{L}_{\text{BCE}}
+
0.5\mathcal{L}_{\text{Dice}}
\]

However, BCE was strongly influenced by the overwhelming number of background pixels.

---

## Focal-Dice

Focal weighting was introduced to reduce the influence of easy background examples.

Parameters:

```text
α = 0.66
γ = 2.0
```

The Focal term was combined with Dice loss.

---

## Final — Balanced Tversky Loss

Transfer learning increased Recall but also produced false positives in:

- Quiet-Sun granulation
- Active regions
- Other dark non-filament structures

The final training objective therefore uses **Tversky Loss** with:

```text
α = 0.4
β = 0.6
```

The Tversky formulation explicitly controls the relative penalty assigned to false negatives and false positives.

An earlier configuration:

```text
α = 0.3
β = 0.7
```

increased Precision but reduced Recall too aggressively.

The final:

```text
α = 0.4
β = 0.6
```

configuration produced a more balanced precision-recall operating point and improved IoU.

---

# Inference Pipeline

The final inference system operates directly on the complete:

```text
2048 × 2048 × 1
```

full-disk image.

Because the U-Net is fully convolutional, no tile stitching is required.

This avoids:

- Tile boundary artifacts
- Stitching inconsistencies
- Duplicate processing of overlapping regions

---

## Test-Time Augmentation

The final inference pipeline uses a three-pass augmentation ensemble:

```text
Original
Horizontal Flip
Vertical Flip
```

The predictions from flipped inputs are transformed back into the original coordinate system before averaging.

```text
                 2048 × 2048 Image
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       Original       H-Flip          V-Flip
          │              │              │
       Model           Model           Model
          │              │              │
          └──── Inverse Transform ─────┘
                         │
                Average Probability Map
                         │
                      p > 0.5
                         │
                 Binary Segmentation
```

TTA improves prediction consistency and reduces orientation-specific artifacts without introducing additional trainable parameters.

---

# Post-Processing

## Connected Components

DenseCRF was considered but rejected because of its computational cost and undesirable behavior around visually similar solar structures.

Reported DenseCRF latency:

```text
~6.5 seconds / 2048 × 2048 image
```

The final pipeline uses OpenCV:

```python
cv2.connectedComponentsWithStats
```

with eight-way connectivity.

Components smaller than:

```text
64 pixels
```

are removed.

This eliminates isolated foreground speckles while preserving sufficiently large coherent filament components.

---

# End-to-End Pipeline

```text
                  MAGFiLO Hα Image
                      2048 × 2048
                           │
                           ▼
                Dataset / Annotation Triage
                           │
                           ▼
              Native-resolution 512 × 512
                  Biased Training Crops
                           │
               ┌───────────┴───────────┐
               │                       │
       50% Filament Crops        50% Background Crops
       (Centroid-Biased)           (Uniform)
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                    ResNet34-U-Net
                           │
                  5 × 5 Decoder Blocks
                 InstanceNorm + LeakyReLU
                           │
                           ▼
                    Tversky Loss
                   α = 0.4, β = 0.6
                           │
                           ▼
                 Full-Disk Inference
                    2048 × 2048
                           │
                           ▼
                         D4 TTA
                           │
                           ▼
                       p > 0.5
                           │
                           ▼
                 Connected Components
                      Area ≥ 64 px
                           │
                           ▼
                   Final Filament Mask
```

---

# Experimental Progression

| Phase | Architecture / Configuration | Resolution | Mean IoU | Precision | Recall |
|---|---|---:|---:|---:|---:|
| 1 | Plain U-Net, 3 × 3, BatchNorm, BCE + Dice | 512 × 512 resized | 0.3139 | — | — |
| 2 | MORDEN-Lite, 5 × 5, InstanceNorm, Focal-Dice | 512 × 512 biased crops | 0.5157 | — | — |
| 3 | MORDEN-Lite + OpenCV area filter | 2048 × 2048 | 0.3815 | 0.5483 | 0.5931 |
| 4 | ResNet34-U-Net + collapsed Conv1 + Focal-Dice | 2048 × 2048 | 0.3880 | 0.4631 | 0.7612 |
| 5 | ResNet34-U-Net + Tversky (α = 0.3, β = 0.7) | 2048 × 2048 | 0.4179 | 0.6822 | 0.5496 |
| 6 | ResNet34-U-Net + Tversky (α = 0.4, β = 0.6) | 2048 × 2048 | 0.4700 | 0.6846 | 0.6229 |
| **Final** | **ResNet34-U-Net + Balanced Tversky + TTA** | **2048 × 2048** | **0.4722** | **0.7215** | **0.6057** |

---

# Final Evaluation

The final configuration was evaluated natively on the **231 full-disk validation images**.

| Metric | Final Result |
|---|---:|
| **Mean IoU** | **0.4722** |
| **Global Pixel Precision** | **0.7215** |
| **Global Pixel Recall** | **0.6057** |
| **Hit Rate** | **0.9075** |
| **Miss Rate** | **0.0925** |
| **AP@IoU 0** | **0.6928** |
| **Multi-scale IoU (MIoU)** | **0.4729** |
| **Inference Latency** | **~0.31 s / full disk** |
| **Throughput** | **~3.23 images/s** |

### Object Detection Statistics

```text
Ground Truth objects : 1,633
Detected objects     : 1,482
Missed objects       :   151
```

Therefore:

```text
Hit Rate  = 1,482 / 1,633 = 0.9075
Miss Rate =   151 / 1,633 = 0.0925
```

For AP@IoU 0:

```text
Valid detections      : 1,838
Total predicted comps : 2,653
AP@IoU 0              : 0.6928
```

Multi-scale IoU was evaluated across:

```text
1×
0.5×
0.25×
```

---

# Ablation Studies

## Native Resolution vs. Downsampled Evaluation

Evaluating the model at 1024 × 1024 instead of native 2048 × 2048 resulted in:

```text
Mean IoU:
0.4700 → 0.3523

Recall:
0.6229 → 0.4456
```

This confirms the sensitivity of the learned representation to the physical scale of thin filament structures.

Since training used native-resolution crops, reducing the full-disk resolution also reduces apparent filament thickness and makes fine structures harder to activate and reconstruct.

---

## Pseudo-Labeling Failure

High-confidence pseudo-labeling was investigated using:

```text
p > 0.85
```

as a candidate threshold.

The approach was rejected because dark quiet-Sun fibrils and other non-filament structures could become incorrect pseudo-labels. Subsequent training then reinforced these errors, destabilizing precision.

---

## Hit Rate vs. IoU

The final model achieves:

```text
Hit Rate : 90.75%
Mean IoU  : 47.22%
```

These metrics measure different properties.

The high Hit Rate indicates that the majority of filament objects are successfully located.

The lower IoU is influenced by:

- Extremely sparse foreground coverage
- Thin filament geometry
- Small boundary displacements

A 1–2 pixel boundary offset can significantly change overlap metrics even when the filament remains correctly localized as an object.

---

# Computational Efficiency

The complete system was designed to run on consumer-grade hardware.

### Hardware

```text
GPU  : NVIDIA GeForce RTX 4050 Laptop GPU
VRAM : ~6 GB
```

### Final Runtime

```text
~0.31 seconds / full 2048 × 2048 disk
~3.23 images / second
```

The pipeline achieves this without DenseCRF-based refinement, instead relying on optimized OpenCV connected-component filtering.

---

# Repository Structure

The repository separates data management, experimentation, model definitions, training, evaluation, and reports.

```text
.
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   ├── processed/
│   └── manifests/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_model_learning.ipynb
│   └── ...
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── annotations.py
│   │   ├── cropping.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── unet.py
│   │   ├── resnet34_unet.py
│   │   └── blocks.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── train.py
│   │   └── utils.py
│   │
│   └── evaluation/
│       ├── inference.py
│       ├── postprocess.py
│       └── metrics.py
│
├── configs/
│   └── ...
│
├── checkpoints/
│   └── ...
│
├── experiments/
│   └── ...
│
├── reports/
│   └── ReportTask_A.pdf
│
├── requirements.txt
├── README.md
└── ...
```

> The exact filenames may evolve with the implementation; the structure above reflects the modular organization of the pipeline.

---

# Environment

The project was developed around a consumer-grade GPU environment:

```text
Operating System : Fedora Linux / KDE Plasma
Python           : 3.12
Framework        : PyTorch 2.14.0+cu126
GPU              : NVIDIA GeForce RTX 4050 Laptop GPU
VRAM             : ~6 GB
```

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Reproducing the Pipeline

The complete workflow is:

```text
1. Prepare MAGFiLO dataset
        ↓
2. Parse and group annotations
        ↓
3. Rasterize segmentation masks
        ↓
4. Create stratified train/validation split
        ↓
5. Generate native-resolution biased crops
        ↓
6. Train ResNet34-U-Net
        ↓
7. Evaluate on native 2048 × 2048 full disks
        ↓
8. Apply horizontal / vertical TTA
        ↓
9. Threshold probability map
        ↓
10. Apply connected-component filtering
        ↓
11. Compute competition metrics
```

Example entry points, depending on the final repository implementation:

```bash
python -m src.data.prepare
python -m src.training.train
python -m src.evaluation.inference
python -m src.evaluation.metrics
```

---

# Design Principles

### Preserve Physical Scale

Avoid destructive resizing when the target structures are only a few pixels wide.

### Use Transfer Learning

Leverage pretrained visual representations rather than learning every feature from scratch.

### Treat Class Imbalance Explicitly

Foreground sparsity should influence both the sampling strategy and the loss function.

### Balance Precision and Recall

A segmentation system should not maximize detection by simply predicting large amounts of visually plausible background.

### Prefer Lightweight Post-Processing

Connected-component filtering provides a fast alternative to computationally expensive CRF refinement.

### Design for Real Hardware

The pipeline was designed around a **6 GB VRAM consumer GPU**, rather than assuming access to large data-center accelerators.

---

# Technical Report

The complete methodology, experiments, ablations, and quantitative analysis are documented in the technical report:

**[Task A Technical Report](./reports/ReportTask_A.pdf)**

---

# Video Demonstration

A short demonstration of the complete pipeline can be linked here:

**[Watch the Task A Demo](ADD_VIDEO_LINK_HERE)**

A useful demonstration can show:

```text
Input H-alpha image
        ↓
Model prediction
        ↓
TTA prediction
        ↓
Post-processed segmentation
        ↓
Final mask + inference latency
```

---

# Supplementary Material

The project references the supplementary problem-statement file:

```text
solarfragmentsCV
```

---

# References

1. Ahmadzadeh, A., et al. (2024). *A dataset of manually annotated filaments from H-alpha observations (MAGFiLO).* Scientific Data.  
   https://doi.org/10.1038/s41597-024-03876-y

2. Hu, et al. (2026). *A Modern ConvNet for Solar Filament Detection (MORDEN).* The Astrophysical Journal.

3. Global Oscillation Network Group (GONG). *Full-disk H-alpha solar observations.* National Solar Observatory.

4. Inter IIT Tech Meet 15.0. *solarfragmentsCV — Computer Vision Prepathon Problem Statement Supplementary File.*

---

# Final Results at a Glance

| Metric | Result |
|---|---:|
| **Mean IoU** | **0.4722** |
| **Precision** | **0.7215** |
| **Recall** | **0.6057** |
| **Hit Rate** | **0.9075** |
| **Miss Rate** | **0.0925** |
| **AP@IoU 0** | **0.6928** |
| **MIoU** | **0.4729** |
| **Latency** | **~0.31 s / image** |
| **Throughput** | **~3.23 images/s** |
| **Resolution** | **2048 × 2048 native full disk** |
| **Hardware** | **RTX 4050, 6 GB VRAM** |

---

## Task A

**Automated Segmentation of Solar Filaments**

The final system combines native-resolution training, biased foreground sampling, ResNet34 transfer learning, MORDEN-inspired decoder design, balanced Tversky optimization, full-disk inference, test-time augmentation, and lightweight connected-component post-processing to produce an efficient solar-filament segmentation pipeline.
