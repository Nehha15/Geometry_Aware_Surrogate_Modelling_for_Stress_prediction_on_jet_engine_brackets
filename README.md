# Geometry-Aware Stress Surrogate — DeepJEB Bracket Dataset

A PointNet++ surrogate model predicting Von Mises stress in jet engine bracket
geometries from 3D point clouds, trained on real nonlinear FEA results from the
Point-DeepONet / DeepJEB dataset (2,139 brackets, 4 load directions, Ti-6Al-4V).

---

## What This Project Does

Every jet engine bracket must be validated under multiple load directions before
manufacture. FEA takes minutes to hours per geometry. This surrogate predicts Von
Mises stress in under 10 ms after training on 8,556 real FEA samples.

---

## Dataset

Source: Point-DeepONet / DeepJEB (Hong et al., 2025, Journal of Mechanical Design)
Material: Ti-6Al-4V (E=113.8 GPa, nu=0.342, yield=227.6 MPa, nonlinear hardening)
FEA solver: Altair OptiStruct, nonlinear static, 2 mm second-order tetra elements

Place these files in data/raw/:
  bracket_labels.csv   — scalar FEA results per bracket per load direction
  xyzdmlc.npz          — 3D point cloud coordinates per bracket
  targets.npz          — full-field arrays (optional for scalar surrogate)

Dataset stats:
  2,139 brackets x 4 load directions = 8,556 samples
  Von Mises stress: 235 to 2,277 MPa (mean 650 MPa)
  Bracket mass: 0.56 to 2.41 kg

---

## Quick Start

  pip install -r requirements.txt

  # Demo — no PyTorch or xyzdmlc.npz needed:
  python run_demo.py

  # Step 1: Process data
  python scripts/prepare_dataset.py           # real xyzdmlc.npz
  python scripts/prepare_dataset.py --fallback  # geometry proxy fallback

  # Step 2: Train
  python scripts/train.py

  # Step 3: Evaluate + all 7 plots
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

  # Step 4: Predict on new geometry
  python scripts/predict.py --pc data/processed/00001_pc.npy --dir ver

---

## Model: PointNet++ with Load-Direction Conditioning

  Input (B, 2048, 3)
    -> T-Net (3x3 rotation alignment)
    -> Set Abstraction 1: 512 centroids, r=0.2, 32 neighbors -> (B,512,128)
    -> Set Abstraction 2: 128 centroids, r=0.4, 64 neighbors -> (B,128,256)
    -> Set Abstraction 3: global -> (B,1,1024)
    -> Flatten + concat one-hot load direction [ver/hor/dia/tor]
    -> FC(1028->512) + BN + ReLU + Dropout(0.4)
    -> FC(512->256)  + BN + ReLU + Dropout(0.2)
    -> FC(256->128)  + BN + ReLU
    -> FC(128->1) -> log1p(Von Mises MPa)
    -> expm1() -> Von Mises stress (MPa)

The one-hot load direction is essential: the same bracket geometry has different
peak stress under vertical vs torsional loading.

---

## Training

  Loss:           Huber (delta=1.0) + physics regularization (neg prediction penalty)
  Optimizer:      Adam, lr=1e-3, weight_decay=1e-4
  LR schedule:    Cosine annealing over 100 epochs
  Early stopping: patience=20
  Augmentation:   jitter sigma=0.005 + Z-rotation + scale+-5%
  Batch size:     16
  T-Net reg:      0.001 * ||I - A*A^T||^2

---

## Visualization Outputs (outputs/)

  1_dataset_overview.png     Stress violin/scatter/histogram, sample counts
  2_training_curves.png      Loss curves (log scale) + LR schedule
  3_predicted_vs_actual.png  Scatter plot + per-direction R2/MAPE bars
  4_error_distribution.png   Relative error histograms + CDF per direction
  5_failure_analysis.png     >10% error cases + failure rate by direction
  6_safety_factor_dist.png   FoS scatter + yield classification accuracy
  7_stress_vs_geometry.png   Stress vs mass/volume/nodes/frequency

---

## Expected Performance (with real xyzdmlc.npz)

  R2           > 0.95
  MAPE         < 8%
  RMSE         < 60 MPa
  Within 10%   > 88% of samples
  Inference    < 10 ms (CPU)

---

## Tests

  python tests/test_pipeline.py

  84 total: 65 pass without PyTorch, 19 additional with PyTorch.

---

## Citation

  Hong, S., Kwon, Y., Shin, D., Park, J., Kang, N. (2025).
  DeepJEB: 3D Deep Learning-based Synthetic Jet Engine Bracket Dataset.
  Journal of Mechanical Design, 147(4).
