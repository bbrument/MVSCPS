# MVSCPS — Multi-View Self-Calibrated Photometric Stereo

## Communication
- Language: French
- User: bb
- Skill level: intermediate researcher

## What this repo does

Neural inverse rendering from multi-view OLAT (one-light-at-a-time) images. Jointly recovers:
- **Geometry** (neural SDF → mesh via marching cubes)
- **SVBRDF** (spatially varying reflectance via neural network)
- **Lighting** (per-light direction or position + intensity, learned from scratch)
- **Shadows** (shadow mapping MLP)

Paper: "Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues" (ICCV 2025).

## Architecture

```
launch.py (Hydra entry point)
├── configs/conf/{diligentmv,lucesmv,lucesmv_native,mvscps}.yaml
├── dataloader/
│   ├── dataset_{train,val,test}.py     — IterableDataset, ray sampling
│   ├── load_fn.py                      — DiligentMV camera/image loaders
│   ├── load_fn_lucesmv.py              — LUCES-MV native loaders
│   └── load_fn_idr.py                  — IDR format loaders (cameras.npz)
├── models/
│   ├── mvscps.py                       — NeuSModel: SDF + BRDF + lighting + shadow
│   ├── geometry.py                     — HashGrid SDF network
│   ├── texture.py                      — Neural BRDF
│   └── lighting.py                     — LightingParameters (dir or point)
├── systems/
│   ├── system_mvscps.py                — PL LightningModule (train/val/test/predict)
│   └── criterions.py                   — Losses (weighted L1, eikonal, Chamfer)
└── slurm/                              — SLURM job scripts
```

## Scene Normalization (O2W) — Critical Convention

```
obj_space = (world_mm - O2W_translation) / O2W_scale
```

- **O2W_translation** = object center (mm), found by triangulating mask centers-of-mass across views
- **O2W_scale** = object bounding sphere radius (mm), from mask area + projection geometry
- Computed by `data/data_utils.py::scene_normalization(P_list, mask_list, fg_area_ratio=5)`
- SDF initialized as sphere of radius 0.5 in obj_space
- GT mesh max radius in obj_space: 0.48–0.84 (similar range for DiligentMV and LUCES-MV)
- Mesh export: `v_world = v_obj * O2W_scale + O2W_translation`

### Two normalization sources for LUCES-MV

| Source | Config | O2W from | Camera dist CV |
|--------|--------|----------|---------------|
| **Native (mask-based)** | `lucesmv_native.yaml` | `scene_normalization()` → `camera_params.json` | 17.2% |
| **IDR (scale_mat)** | `lucesmv.yaml` | `cameras.npz` scale_mat from sdmunips | 0.5% |

Both are correct. Native is the principled approach (mask triangulation). IDR is more consistent because sdmunips calibrates a fixed-ratio unit sphere. **Current experiments use native normalization** (validated April 2025: roundtrip < 1e-14, all objects fit unit sphere).

### Light position formula (point light)
```python
# In models/mvscps.py:
light_pos_world = R_c2w @ light_pos + origin   # origin = camera center in obj_space
# So: light_pos = Lpos_cam_mm / O2W_scale
```

## Datasets

### DiligentMV
- 5 objects: bear, buddha, cow, pot2, reading
- 20 views × 96 directional lights (view-aligned OLAT)
- Data: `data/DiLiGenT-MV/` (symlink → `/projects/m25115/DiLiGenT-MV/`)
- Config: `configs/conf/diligentmv.yaml`

### LUCES-MV
- 10 objects: Bowl, Buddha, Bunny, Cup, Die, Hippo, House, Owl, Queen, Squirrel
- 12 views × 15 camera-mounted point lights (LEDs)
- 2 cameras: cam1 (views 0-36), cam2 (views 36-66)
- 16-bit PNG, 1552×2080
- Data: `data/LucesMV_processed/` (symlink → `/projects/m25115/LucesMV_processed/`)
- Config: `configs/conf/lucesmv_native.yaml` (native loader, mask-based O2W)

## Data Locations (symlinks from repo root)

| Symlink | Target | Content |
|---------|--------|---------|
| `data/DiLiGenT-MV` | `/projects/m25115/DiLiGenT-MV` | Raw DiligentMV (5 objects) |
| `data/LucesMV` | `/projects/m25115/LucesMV` | Raw LUCES-MV + GT meshes |
| `data/LucesMV_processed` | `/projects/m25115/LucesMV_processed` | Preprocessed native format |
| `exp` | `/projects/m25115/exp` | All experiment checkpoints |
| `eval` | `/projects/m25115/eval_3d_datasets` | Eval results (dlmv + lucesmv) |
| `eval_pipeline` | `~/dev/eval_dataset/eval_pipeline` | Eval pipeline code |

## Running Experiments

### Training
```bash
# LUCES-MV: 4 variants (dir/point × 1/15 lights)
bash slurm/submit_lucesmv_idr_all.sh          # submits 40 jobs
# DiligentMV
DLMV_NUM_LIGHTS=96 sbatch slurm/run_dlmv.sh bear
DLMV_NUM_LIGHTS=1  sbatch slurm/run_dlmv.sh bear
```

Key env vars for `run_lucesmv.sh`:
- `LUCESMV_LIGHT_TYPE` = directional | point
- `LUCESMV_NUM_LIGHTS` = 1 | 15

### Mesh re-export (from checkpoint, no training)
```bash
EXP_ROOT=/projects/m25115/exp/lucesmv_dir_15l LIGHT_TYPE=directional VL_INDEX=lucesmv_view_12_light_15 \
  sbatch slurm/reexport_mesh.sh Bowl
```

### Evaluation
```bash
source ~/dev/eval_dataset/eval_pipeline/venv/bin/activate
# Auto-detect meshes and submit cleanup+evaluate jobs:
eval-pipeline -c ~/dev/eval_dataset/eval_pipeline/config/lucesmv.yaml watch --scan-mode auto
# DiligentMV:
eval-pipeline -c ~/dev/eval_dataset/eval_pipeline/config/dlmv.yaml watch --scan-mode auto
```

Eval pipeline flow: `results_raw/mesh.ply` → cleanup (visibility filtering) → evaluate (Chamfer + F-score vs GT)

## Latest Results (March 2025)

### DiligentMV — Chamfer Distance (mm)

| Object | Dir 96L | Dir 1L |
|--------|---------|--------|
| bear | 0.219 | 0.289 |
| buddha | 0.166 | 0.238 |
| cow | 0.102 | 0.164 |
| pot2 | 0.174 | 0.182 |
| reading | 0.312 | 0.279 |
| **Mean** | **0.195** | **0.230** |

### LUCES-MV — Chamfer Distance (mm)

| Object | Dir 1L | Dir 15L | Point 1L | Point 15L |
|--------|--------|---------|----------|-----------|
| Bowl | 0.374 | **0.313** | 0.405 | 0.536 |
| Buddha | 0.337 | 0.372 | **0.316** | 0.341 |
| Bunny | **0.207** | 0.241 | 0.228 | 0.327 |
| Cup | 0.548 | **0.505** | 0.563 | 0.588 |
| Die | 0.313 | **0.238** | 0.565 | 0.252 |
| Hippo | **0.233** | 0.245 | 0.281 | 0.296 |
| House | 0.573 | 0.555 | 0.559 | **0.555** |
| Owl | 0.299 | 0.282 | **0.273** | 0.299 |
| Queen | 0.345 | 0.291 | 0.436 | **0.269** |
| Squirrel | 0.421 | **0.349** | 0.361 | 0.376 |
| **Mean** | 0.365 | **0.339** | 0.399 | 0.384 |

Dir-15L wins on average (4/10 objects). Point light helps on specular objects (Queen, Buddha).

## Experiment Storage

```
/projects/m25115/exp/
├── diligentmv_dir_96l/{bear,...}/@{timestamp}/ckpt/last.ckpt
├── diligentmv_dir_1l/{bear,...}/@{timestamp}/...
├── lucesmv_dir_15l/{Bowl,...}/@{timestamp}/...
├── lucesmv_dir_1l/{Bowl,...}/@{timestamp}/...
├── lucesmv_point_15l/{Bowl,...}/@{timestamp}/...
└── lucesmv_point_1l/{Bowl,...}/@{timestamp}/...
```

Eval structure:
```
/projects/m25115/eval_3d_datasets/lucesmv/eval/{object}/
├── Groundtruth/gt_pcd.npy
└── mvscps-{dir,point}-{1,15}l/nbv-12/nbl-{1,15}/nbit-20000/
    ├── results_raw/mesh.ply → symlink to exp mesh
    ├── results_cleaned/mesh.ply
    └── eval_results/metrics.json
```

## Key Files

| File | Role |
|------|------|
| `launch.py` | Hydra entry point, trial naming, phase dispatch |
| `data/data_utils.py:110` | `scene_normalization()` — mask-based O2W |
| `dataloader/dataset_train.py:73-75` | O2W loading, `obj_space = (world - t) / s` |
| `models/mvscps.py` | NeuSModel: ray marching, SDF, BRDF, lighting, shadow |
| `models/lighting.py` | LightingParameters: dir or point, learnable |
| `systems/system_mvscps.py:642` | Mesh export: `v_world = v_obj * s + t` |
| `slurm/run_lucesmv.sh` | LUCES-MV training (configurable light type/count) |
| `slurm/run_dlmv.sh` | DiligentMV training (configurable light count) |
| `slurm/reexport_mesh.sh` | Mesh re-export from checkpoint |
| `slurm/copy_results_to_eval.sh` | Symlink mesh to eval directory |

## Known Issues

- **Ray-mesh intersection hang**: `system_mvscps.py:710-735` hangs with large GT meshes (Buddha 79MB) at full resolution. Workaround: `gt_mesh_fpath=None` during predict phase.
- **PyVista crash on SLURM**: No X11 on compute nodes → core dump after mesh save. Non-fatal.
- **cam1 vs cam2 LED positions**: `prepare_data_lucesmv_idr.py` uses cam1 LEDs for all views. cam2 has a different LED arrangement (most frontal = L03 vs L00). Approximation for now.
- **predict_mesh not in default predict_targets**: `lucesmv_native.yaml` has `predict_targets: ["predict_brdf", "predict_relighting"]`. Must add `predict_mesh` via hydra override or config edit.
