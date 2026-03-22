"""
config_patch_v3.py
==================
Run once to update configs/config.yaml for v3.
"""
import yaml

path = "configs/config.yaml"
with open(path) as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("model", {}).update({
    "n_load_directions": 3,
    "n_phys_features":  13,
})
cfg.setdefault("training", {}).update({
    "epochs":                     150,
    "scheduler_T_max":            150,
    "early_stopping_patience":    999,   # effectively disabled — let it run
    "early_stopping_min_delta":   0.0001,
    "early_stopping_ema_alpha":   0.3,
    "save_every":                 10,
})

with open(path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"Patched {path}")
print(f"  n_load_directions : {cfg['model']['n_load_directions']}")
print(f"  n_phys_features   : {cfg['model']['n_phys_features']}")
print(f"  epochs            : {cfg['training']['epochs']}")
print(f"  early_stop patience: {cfg['training']['early_stopping_patience']} (disabled)")