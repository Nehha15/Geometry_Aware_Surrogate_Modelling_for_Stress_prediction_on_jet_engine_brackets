"""
test_pipeline.py
================
Complete test suite for the DeepJEB stress surrogate project.

Covers:
  1. Data loading & preprocessing  (no torch required)
  2. Dataset statistics & splits
  3. Metrics correctness
  4. Visualisation outputs
  5. Model architecture            (torch required)
  6. Training components           (torch required)
  7. End-to-end inference          (torch required)

Run:
    python tests/test_pipeline.py

All tests in sections 1-4 run without PyTorch.
Sections 5-7 are skipped gracefully when torch is not installed.
"""

import os, sys, tempfile, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─── tiny test harness ────────────────────────────────────────────────────────
PASS = 0; FAIL = 0; SKIP = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {msg}")

def skip(name, reason="torch not available"):
    global SKIP
    SKIP += 1; print(f"  SKIP  {name}  ({reason})")

def section(title):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")


# ═══════════════════════════════════════════════════════════════
# 1. DEEPJEB LOADER
# ═══════════════════════════════════════════════════════════════
section("1. DeepJEB data loader")

from src.data.deepjeb_loader import (
    normalize_pointcloud, normalize_stress_mpa, denormalize_stress,
    fps_subsample, random_subsample,
    STRESS_COLUMNS, YIELD_STRESS_MPA, LOAD_DIRECTION_IDX,
    load_deepjeb_fallback,
)

# Normalization: unit sphere
pc = np.random.randn(2048, 3).astype(np.float32) * 5 + np.array([10, -3, 7])
pn = normalize_pointcloud(pc)
check("PC centroid at origin (x)", abs(pn[:,0].mean()) < 1e-5)
check("PC centroid at origin (y)", abs(pn[:,1].mean()) < 1e-5)
check("PC centroid at origin (z)", abs(pn[:,2].mean()) < 1e-5)
check("PC max dist = 1.0", abs(np.max(np.linalg.norm(pn, axis=1)) - 1.0) < 1e-5)
check("PC dtype float32", pn.dtype == np.float32)
check("PC shape preserved", pn.shape == (2048, 3))

# Stress normalization roundtrip
for vm in [10.0, 100.0, 227.6, 500.0, 1500.0, 2277.0]:
    yn = normalize_stress_mpa(vm)
    yr = denormalize_stress(yn)
    check(f"Stress roundtrip {vm:.1f} MPa", abs(yr - vm) / vm < 1e-10,
          f"got {yr:.4f}")

# Monotonicity: higher stress → higher normalized value
vals = [10.0, 50.0, 200.0, 500.0, 1000.0]
norm = [normalize_stress_mpa(v) for v in vals]
check("Stress normalization monotone",
      all(norm[i] < norm[i+1] for i in range(len(norm)-1)))

# Constants
check("YIELD_STRESS_MPA = 227.6", abs(YIELD_STRESS_MPA - 227.6) < 0.01)
check("4 load direction indices",
      set(LOAD_DIRECTION_IDX.values()) == {0, 1, 2, 3})
check("4 stress columns defined",
      set(STRESS_COLUMNS.keys()) == {"ver","hor","dia","tor"})

# Subsampling
big_pc = np.random.randn(10000, 3).astype(np.float32)
rng    = np.random.default_rng(0)
sub_r  = random_subsample(big_pc, 2048, rng)
sub_f  = fps_subsample(big_pc, 512, rng)
check("Random subsample shape",   sub_r.shape == (2048, 3))
check("FPS subsample shape",      sub_f.shape == (512, 3))
check("Random subsample float32", sub_r.dtype == np.float32)
check("FPS: all selected points are from original",
      all(np.any(np.all(sub_f[i] == big_pc, axis=1)) for i in range(0, 512, 50)))

# Padding when pc smaller than n_points
small_pc = np.random.randn(100, 3).astype(np.float32)
padded   = random_subsample(small_pc, 2048, rng)
check("Padding small PC to target size", padded.shape == (2048, 3))


# ═══════════════════════════════════════════════════════════════
# 2. FALLBACK LOADER WITH REAL CSV
# ═══════════════════════════════════════════════════════════════
section("2. Fallback loader — real bracket_labels.csv")

CSV_PATH = "data/raw/bracket_labels.csv"
if not os.path.exists(CSV_PATH):
    print(f"  SKIP: {CSV_PATH} not found — place CSV in data/raw/")
else:
    tmpdir = tempfile.mkdtemp()
    try:
        stats = load_deepjeb_fallback(
            labels_csv=CSV_PATH, out_dir=tmpdir,
            n_points=256, load_directions=("ver","hor","dia","tor"),
            seed=42, verbose=False,
        )
        n = stats["total_samples"]
        check("Fallback: total samples > 0",  n > 0,  f"got {n}")
        check("Fallback: 4 directions",
              all(k in stats["per_direction"] for k in ("ver","hor","dia","tor")))

        # Each direction should have equal counts
        counts = [stats["per_direction"][d]["count"]
                  for d in ("ver","hor","dia","tor")]
        check("Fallback: balanced direction counts",
              max(counts) - min(counts) <= 1)

        # Verify a few files on disk
        files = [f for f in os.listdir(tmpdir) if f.endswith("_pc.npy")]
        check("Fallback: correct number of PC files",
              len(files) == n, f"{len(files)} vs {n}")

        # Spot-check one sample
        if files:
            base = files[0].replace("_pc.npy","")
            pc   = np.load(os.path.join(tmpdir, f"{base}_pc.npy"))
            y    = float(np.load(os.path.join(tmpdir, f"{base}_y.npy")))
            meta = np.load(os.path.join(tmpdir, f"{base}_meta.npy"),
                           allow_pickle=True).item()
            check("Fallback: PC shape (N,3)", pc.shape[1] == 3)
            check("Fallback: PC unit sphere",
                  abs(np.max(np.linalg.norm(pc,axis=1)) - 1.0) < 0.01)
            check("Fallback: y is positive",  y > 0)
            check("Fallback: meta has von_mises_mpa", "von_mises_mpa" in meta)
            check("Fallback: meta has load_direction", "load_direction" in meta)
            check("Fallback: meta has safety_factor",  "safety_factor" in meta)
            check("Fallback: meta has mass_kg",         "mass_kg" in meta)
            vm  = meta["von_mises_mpa"]
            fos = meta["safety_factor"]
            check("Fallback: stress > 0", vm > 0)
            check("Fallback: FoS consistent",
                  abs(fos - YIELD_STRESS_MPA / vm) < 0.01)

        # Stress range from real data
        all_vm = [stats["per_direction"][d]["vm_max"]
                  for d in stats["per_direction"] if stats["per_direction"][d]["count"] > 0]
        check("Fallback: stress max > 200 MPa (real brackets)",
              max(all_vm) > 200.0, f"max={max(all_vm):.1f}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 3. METRICS
# ═══════════════════════════════════════════════════════════════
section("3. Evaluation metrics")

from src.evaluation.metrics import (
    rmse, mae, r2_score, mape, compute_all_metrics,
    denormalize_stress as dm
)

# Perfect predictions
t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
check("RMSE perfect = 0",   abs(rmse(t, t)) < 1e-10)
check("MAE perfect = 0",    abs(mae(t, t))  < 1e-10)
check("R² perfect = 1",     abs(r2_score(t, t) - 1.0) < 1e-10)
check("MAPE perfect = 0",   abs(mape(t, t)) < 1e-6)

# Known values
t   = np.array([3.0, 3.0, 3.0, 3.0])
p   = np.array([4.0, 2.0, 4.0, 2.0])
check("RMSE known = 1.0",   abs(rmse(p, t) - 1.0) < 1e-10)
check("MAE known = 1.0",    abs(mae(p, t)  - 1.0) < 1e-10)
t_r2 = np.array([1.0, 2.0, 3.0, 4.0])
check("R² = 0 (mean pred)",
      abs(r2_score(t_r2.mean() * np.ones_like(t_r2), t_r2)) < 1e-6)

# R² negative for worse-than-mean
bad = t + 5.0
check("R² < 0 for bad preds", r2_score(bad, t) < 0)

# MAPE: 50% and 100% errors
t2  = np.array([100.0, 200.0])
p2  = np.array([150.0, 100.0])
exp = 100 * (50/100 + 100/200) / 2   # 37.5%
check("MAPE 37.5%", abs(mape(p2, t2) - exp) < 0.01)

# compute_all_metrics end-to-end
rng  = np.random.default_rng(1)
true_log = rng.uniform(2.0, 7.0, 200)
pred_log = true_log + rng.normal(0, 0.1, 200)   # small noise
metrics  = compute_all_metrics(pred_log, true_log)
check("compute_all_metrics: r2 in result",         "r2"          in metrics)
check("compute_all_metrics: rmse_mpa in result",   "rmse_mpa"    in metrics)
check("compute_all_metrics: mape_pct in result",   "mape_pct"    in metrics)
check("compute_all_metrics: max_err_mpa in result","max_err_mpa" in metrics)
check("compute_all_metrics: rel_err dict",         "rel_err"     in metrics)
check("compute_all_metrics: R² high for low noise",metrics["r2"] > 0.95,
      f"R²={metrics['r2']:.4f}")
check("compute_all_metrics: MAPE < 20%",
      metrics["mape_pct"] < 20.0, f"MAPE={metrics['mape_pct']:.2f}%")
check("compute_all_metrics: within_5pct key",
      "within_5pct" in metrics["rel_err"])
check("compute_all_metrics: within_10pct key",
      "within_10pct" in metrics["rel_err"])

# denormalize is inverse of normalize
for vm in [1.0, 50.0, 500.0, 2000.0]:
    yn = normalize_stress_mpa(vm)
    yr = dm(yn)
    check(f"dm∘normalize roundtrip {vm:.0f} MPa", abs(yr - vm)/vm < 1e-10)


# ═══════════════════════════════════════════════════════════════
# 4. VISUALIZATIONS (numpy/matplotlib only)
# ═══════════════════════════════════════════════════════════════
section("4. Visualization outputs")

import matplotlib
matplotlib.use("Agg")

from src.evaluation.visualize import (
    plot_dataset_overview, plot_training_curves,
    plot_predicted_vs_actual, plot_error_distribution,
    plot_failure_analysis, plot_safety_factor_distribution,
    plot_stress_vs_geometry, VisualizationSuite,
)

tmpviz = tempfile.mkdtemp()

# Synthetic meta list (mimics real DeepJEB metadata)
rng  = np.random.default_rng(42)
meta_list = []
dirs = ["ver","hor","dia","tor"]
for i in range(80):
    d  = dirs[i % 4]
    vm = float(rng.uniform(250, 2000))
    meta_list.append({
        "load_direction":  d,
        "load_dir_idx":    dirs.index(d),
        "von_mises_mpa":   vm,
        "safety_factor":   227.6 / vm,
        "yielded":         vm > 227.6,
        "mass_kg":         float(rng.uniform(0.5, 2.5)),
        "volume_mm3":      float(rng.uniform(100000, 400000)),
        "num_nodes":       int(rng.integers(120000, 380000)),
        "freq_1st_hz":     float(rng.uniform(1000, 6000)),
        "disp_x_mm":       float(rng.uniform(0, 0.5)),
        "disp_y_mm":       float(rng.uniform(0, 0.5)),
        "disp_z_mm":       float(rng.uniform(0, 0.5)),
        "disp_mag_mm":     float(rng.uniform(0, 0.7)),
    })

# Synthetic report (mimics validate() output)
n = len(meta_list)
true_log = np.array([normalize_stress_mpa(m["von_mises_mpa"]) for m in meta_list])
pred_log = true_log + rng.normal(0, 0.15, n)
from src.evaluation.metrics import compute_all_metrics
metrics  = compute_all_metrics(pred_log, true_log)
per_dir  = {}
d_arr    = np.array([m["load_direction"] for m in meta_list])
for d in dirs:
    mask = d_arr == d
    if mask.sum() > 0:
        per_dir[d] = compute_all_metrics(pred_log[mask], true_log[mask])

report = {
    "split":    "test",
    "n":        n,
    "metrics":  metrics,
    "per_dir":  per_dir,
    "pred_log": pred_log.tolist(),
    "true_log": true_log.tolist(),
    "dirs":     d_arr.tolist(),
}

history = {
    "train_loss": list(np.exp(-np.linspace(0, 2, 30)) * 0.5 + 0.05),
    "val_loss":   list(np.exp(-np.linspace(0, 1.8, 30)) * 0.55 + 0.08),
    "lr":         list(np.cos(np.linspace(0, np.pi, 30)) * 5e-4 + 5e-4),
}

def plot_test(name, fn, *args):
    path = os.path.join(tmpviz, f"{name}.png")
    try:
        fn(*args, out_path=path)
        exists = os.path.exists(path) and os.path.getsize(path) > 1000
        check(f"plot_{name}: file created and non-empty", exists)
    except Exception as e:
        check(f"plot_{name}: no exception", False, str(e)[:80])

plot_test("dataset_overview",    plot_dataset_overview, meta_list)
plot_test("training_curves",     plot_training_curves,  history)
plot_test("predicted_vs_actual", plot_predicted_vs_actual, report)
plot_test("error_distribution",  plot_error_distribution,  report)
plot_test("failure_analysis",    plot_failure_analysis,    report)
plot_test("safety_factor",       plot_safety_factor_distribution, report, meta_list)
plot_test("stress_vs_geometry",  plot_stress_vs_geometry, report, meta_list)

# VisualizationSuite
try:
    suite = VisualizationSuite(plots_dir=tmpviz)
    suite.plot_all(report, history, meta_list)
    pngs  = [f for f in os.listdir(tmpviz) if f.endswith(".png")]
    check("VisualizationSuite: >= 6 plots generated", len(pngs) >= 6,
          f"got {len(pngs)}")
except Exception as e:
    check("VisualizationSuite.plot_all: no exception", False, str(e)[:80])

shutil.rmtree(tmpviz, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 5. MODEL ARCHITECTURE  (torch required)
# ═══════════════════════════════════════════════════════════════
section("5. Model architecture")

if not HAS_TORCH:
    for t in ["TNet output shape", "TNet reg loss scalar",
              "TNet identity init near I", "TNet reg loss at identity = 0",
              "PointNet++ output shape (B,1)", "T-Net trans shape (B,3,3)",
              "No T-Net: trans is None", "Eval mode deterministic",
              "Grad flows through all layers",
              "Load dir one-hot conditioning",
              "Parameter count > 100k"]:
        skip(t)
else:
    from src.models.tnet import TNet, tnet_regularization_loss
    from src.models.pointnet_plus import PointNetPlusPlus, _fps, _index_points, _ball_query

    # T-Net
    tnet  = TNet(k=3)
    x_in  = torch.randn(4, 3, 512)
    trans = tnet(x_in)
    check("TNet output shape (B,3,3)", trans.shape == (4, 3, 3))

    loss_t = tnet_regularization_loss(trans)
    check("TNet reg loss scalar", loss_t.shape == torch.Size([]))

    # Near-identity at init
    tnet2   = TNet(k=3)
    x_zero  = torch.zeros(2, 3, 128)
    t2      = tnet2(x_zero)
    I       = torch.eye(3).unsqueeze(0).expand(2, -1, -1)
    check("TNet identity init near I",
          (t2 - I).abs().max().item() < 1.0)

    # Exact identity gives zero reg loss
    I_batch = torch.eye(3).unsqueeze(0).repeat(4,1,1)
    check("TNet reg loss at identity = 0",
          tnet_regularization_loss(I_batch).item() < 1e-10)

    # PointNet++
    B, N = 2, 2048
    xyz  = torch.randn(B, N, 3)
    lc   = torch.tensor([0, 2], dtype=torch.long)

    model = PointNetPlusPlus(use_tnet=True, dropout=0.0, n_load_directions=4)
    model.eval()
    with torch.no_grad():
        pred, trans = model(xyz, lc)

    check("PointNet++ output shape (B,1)", pred.shape == (B, 1))
    check("T-Net trans shape (B,3,3)",     trans.shape == (B, 3, 3))

    model_no_tnet = PointNetPlusPlus(use_tnet=False, dropout=0.0)
    model_no_tnet.eval()
    with torch.no_grad():
        p2, t2 = model_no_tnet(xyz)
    check("No T-Net: trans is None",       t2 is None)
    check("No T-Net: output shape correct", p2.shape == (B, 1))

    # Deterministic in eval mode
    with torch.no_grad():
        p1, _ = model(xyz, lc)
        p2, _ = model(xyz, lc)
    check("Eval mode deterministic", torch.allclose(p1, p2))

    # Gradient flow
    model_train = PointNetPlusPlus(use_tnet=True, dropout=0.0, n_load_directions=4)
    model_train.train()
    xyz_r = torch.randn(2, 2048, 3)
    lc_r  = torch.tensor([1, 3], dtype=torch.long)
    p_r, trans_r = model_train(xyz_r, lc_r)
    loss = p_r.mean() + 0.001 * tnet_regularization_loss(trans_r)
    loss.backward()
    nan_grads = [(n, p.grad) for n, p in model_train.named_parameters()
                 if p.grad is not None and torch.isnan(p.grad).any()]
    check("Grad flows: no NaN gradients", len(nan_grads) == 0,
          f"NaN in: {[n for n,_ in nan_grads[:3]]}")

    # Load direction conditioning
    xyz1  = torch.randn(1, 2048, 3)
    model_lc = PointNetPlusPlus(use_tnet=False, dropout=0.0, n_load_directions=4)
    model_lc.eval()
    outs = []
    with torch.no_grad():
        for d in range(4):
            lc_d = torch.tensor([d])
            o, _ = model_lc(xyz1, lc_d)
            outs.append(o.item())
    check("Load dir conditioning: 4 different outputs",
          len(set(f"{v:.4f}" for v in outs)) == 4,
          f"outs={[f'{v:.4f}' for v in outs]}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("Parameter count > 100k", n_params > 100_000, f"{n_params:,}")


# ═══════════════════════════════════════════════════════════════
# 6. TRAINING COMPONENTS  (torch required)
# ═══════════════════════════════════════════════════════════════
section("6. Training components")

if not HAS_TORCH:
    for t in ["StressLoss output is scalar", "StressLoss dict keys",
              "StressLoss penalises negative preds",
              "EarlyStopping fires after patience",
              "EarlyStopping resets on improvement"]:
        skip(t)
else:
    from src.training.losses import StressLoss
    from src.training.trainer import EarlyStopping

    loss_fn = StressLoss(primary="huber", delta=1.0, physics_weight=0.05)
    pred    = torch.tensor([[2.0],[3.0],[4.0]])
    target  = torch.tensor([[2.1],[2.9],[4.2]])
    total, d = loss_fn(pred, target)
    check("StressLoss output is scalar", total.shape == torch.Size([]))
    check("StressLoss dict keys",
          all(k in d for k in ("loss_primary","loss_physics","loss_total")))
    check("StressLoss total >= primary",
          d["loss_total"] >= d["loss_primary"] - 1e-6)

    # Physics penalty for negative predictions
    neg_pred = torch.tensor([[-1.0],[-2.0]])
    pos_pred = torch.tensor([[1.0],[2.0]])
    target2  = torch.tensor([[1.0],[2.0]])
    l_neg, d_neg = loss_fn(neg_pred, target2)
    l_pos, d_pos = loss_fn(pos_pred, target2)
    check("StressLoss penalises negative preds",
          d_neg["loss_physics"] > d_pos["loss_physics"])

    # EarlyStopping
    es = EarlyStopping(patience=3, min_delta=1e-4)
    results = [es.step(v) for v in [1.0, 0.9, 0.85, 0.84, 0.84, 0.84]]
    check("EarlyStopping fires after patience",
          results[-1] is True, f"results={results}")

    es2 = EarlyStopping(patience=3)
    es2.step(1.0); es2.step(0.9); es2.step(0.85)
    check("EarlyStopping counter resets on improvement",
          es2.counter == 0)

    # Constant loss → fires after patience
    es3 = EarlyStopping(patience=2)
    r = [es3.step(0.5) for _ in range(5)]
    check("EarlyStopping fires on plateau", any(r))


# ═══════════════════════════════════════════════════════════════
# 7. END-TO-END INFERENCE  (torch required + processed data)
# ═══════════════════════════════════════════════════════════════
section("7. End-to-end inference (no training needed)")

if not HAS_TORCH:
    skip("End-to-end predict_one")
    skip("Predict: FoS consistent with stress")
    skip("Predict: all 4 directions give different stress")
elif not os.path.exists("data/processed"):
    skip("End-to-end predict_one", "data/processed not found")
else:
    from src.models.pointnet_plus import PointNetPlusPlus
    from src.data.deepjeb_loader import (
        normalize_pointcloud, denormalize_stress,
        LOAD_DIRECTION_IDX, YIELD_STRESS_MPA, random_subsample
    )

    device = torch.device("cpu")

    # Untrained model — just tests the forward pass shape and denormalize
    model_inf = PointNetPlusPlus(use_tnet=True, dropout=0.0, n_load_directions=4)
    model_inf.eval()

    pc_raw  = np.random.randn(5000, 3).astype(np.float32)
    pc_sub  = random_subsample(pc_raw, 2048)
    pc_norm = normalize_pointcloud(pc_sub)
    pc_t    = torch.tensor(pc_norm).unsqueeze(0)

    outs = []
    for d_name, d_idx in LOAD_DIRECTION_IDX.items():
        lc_t = torch.tensor([d_idx])
        with torch.no_grad():
            pred_log, _ = model_inf(pc_t, lc_t)
        vm_mpa = denormalize_stress(pred_log.item())
        fos    = YIELD_STRESS_MPA / max(vm_mpa, 1e-6)
        outs.append((d_name, vm_mpa, fos))

    check("Predict: outputs for all 4 directions", len(outs) == 4)
    check("Predict: all vm_mpa values positive",
          all(vm > 0 for _, vm, _ in outs))
    check("Predict: FoS consistent with stress",
          all(abs(fos - YIELD_STRESS_MPA/vm) < 0.01 for _, vm, fos in outs))
    check("Predict: 4 directions give different stress",
          len(set(f"{vm:.4f}" for _, vm, _ in outs)) == 4)

    # Read a real processed sample
    files = [f for f in os.listdir("data/processed") if f.endswith("_pc.npy")]
    if files:
        base = files[0].replace("_pc.npy","")
        pc   = np.load(f"data/processed/{base}_pc.npy")
        pc_t = torch.tensor(pc).unsqueeze(0)
        lc_t = torch.tensor([0])
        with torch.no_grad():
            pred_log, _ = model_inf(pc_t, lc_t)
        vm   = denormalize_stress(pred_log.item())
        check("Predict on real processed sample: vm > 0", vm > 0, f"vm={vm:.2f}")
        check("Predict on real processed sample: pc unit sphere",
              abs(np.max(np.linalg.norm(pc, axis=1)) - 1.0) < 0.01)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
section("Test summary")
print(f"  Passed : {PASS}")
print(f"  Failed : {FAIL}")
print(f"  Skipped: {SKIP}  (torch not installed)")
print(f"  Total  : {PASS + FAIL + SKIP}")
if FAIL == 0:
    print(f"\n  All tests passed.")
else:
    print(f"\n  {FAIL} test(s) FAILED — see above.")
    sys.exit(1)
