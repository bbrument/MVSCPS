#!/bin/bash
# Submit all 8 Suzanne ablation runs.
#
# Matrix:
#   Dataset:  dir (directional) | point (fixed world-space)
#   Light:    gt (GT frozen)    | learn (learned from init)
#   BRDF:     albfix (fixed=1)  | alblearn (LambertianBRDF learned)
#   Shadow:   off (all runs)
#   Lights:   1 (L00 only)

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p slurm/logs

# Priority: deprioritize existing pending jobs so suzanne runs first
if [[ "${SUZANNE_SKIP_REPRIO:-}" != "1" ]]; then
  PENDING_JOBS=$(squeue -u "$USER" -h -t PD -o "%i" 2>/dev/null | tr '\n' ' ')
  if [[ -n "${PENDING_JOBS}" ]]; then
    echo "Deprioritizing existing pending jobs: ${PENDING_JOBS}"
    for jid in ${PENDING_JOBS}; do
      scontrol update JobId="${jid}" Nice=100 2>/dev/null || true
    done
  fi
fi

# ── Common overrides (all 8 runs) ────────────────────────────────────
COMMON=(
  conf.model.use_shadow=false
  conf.exp.test_after_train=false
  "'conf.dataset.predict_targets=[\"predict_mesh\"]'"
)

# ── BRDF presets ─────────────────────────────────────────────────────
BRDF_ALBFIX=(
  conf.model.brdf.name=lambertian_brdf
  +conf.model.brdf.albedo_mode=fixed
  "'+conf.model.brdf.albedo_value=[1.0,1.0,1.0]'"
)

BRDF_ALBLEARN=(
  conf.model.brdf.name=lambertian_brdf
  +conf.model.brdf.albedo_mode=learned
  conf.model.brdf.mlp_network_config.output_activation=none
)

# ── Light presets ────────────────────────────────────────────────────
LIGHT_GT=(
  conf.model.light.use_gt_light=true
  conf.model.light.freeze_light=true
)

LIGHT_LEARN_DIR=(
  conf.model.light.use_gt_light=false
  conf.model.light.freeze_light=false
  conf.model.light.light_dir_file=None
)

LIGHT_LEARN_POINT=(
  conf.model.light.use_gt_light=false
  conf.model.light.freeze_light=false
  conf.model.light.light_pos_file=None
)

# ── Per-dataset index overrides (1 light = L00 only) ────────────────
idx_overrides() {
  local ds="$1"  # synthetic_suzanne_dir or synthetic_suzanne_point
  echo "conf.dataset.train.view_light_index_file=data/${ds}/view_light_idx_1l_train.txt"
  echo "conf.dataset.val.view_light_index_file=data/${ds}/view_light_idx_1l_val.txt"
  echo "conf.dataset.test.view_light_index_file=data/${ds}/view_light_idx_1l_test.txt"
  echo "conf.dataset.predict_mesh.view_light_index_file=data/${ds}/view_light_idx_1l_val.txt"
}

# ── Submit helper ────────────────────────────────────────────────────
submit() {
  local run_id="$1"; shift
  local config="$1"; shift
  local tag="$1"; shift
  local exp_path="$1"; shift

  echo ""
  echo "── Run ${run_id}: ${tag} ──"
  local cmd="sbatch --job-name=suz_${run_id} slurm/run_suzanne.sh ${config}"
  cmd+=" conf.exp.exp_path=${exp_path}"
  cmd+=" conf.exp.tag=_${tag}"

  for arg in "${COMMON[@]}"; do cmd+=" ${arg}"; done
  for arg in "$@"; do cmd+=" ${arg}"; done

  echo "  ${cmd}"
  eval "${cmd}"
}

echo "Submitting 8 Suzanne ablation runs..."

# ── DIR dataset ──────────────────────────────────────────────────────
DIR_IDX=($(idx_overrides synthetic_suzanne_dir))

# 1) dir + GT light + albedo fix
submit 1 suzanne_dir "1l_gtlight_albfix" "exp/suzanne_dir" \
  "${DIR_IDX[@]}" "${LIGHT_GT[@]}" "${BRDF_ALBFIX[@]}"

# 2) dir + GT light + albedo learned
submit 2 suzanne_dir "1l_gtlight_alblearn" "exp/suzanne_dir" \
  "${DIR_IDX[@]}" "${LIGHT_GT[@]}" "${BRDF_ALBLEARN[@]}"

# 5) dir + learned light + albedo fix
submit 5 suzanne_dir "1l_learnlight_albfix" "exp/suzanne_dir" \
  "${DIR_IDX[@]}" "${LIGHT_LEARN_DIR[@]}" "${BRDF_ALBFIX[@]}"

# 6) dir + learned light + albedo learned
submit 6 suzanne_dir "1l_learnlight_alblearn" "exp/suzanne_dir" \
  "${DIR_IDX[@]}" "${LIGHT_LEARN_DIR[@]}" "${BRDF_ALBLEARN[@]}"

# ── POINT dataset ────────────────────────────────────────────────────
POINT_IDX=($(idx_overrides synthetic_suzanne_point))

# 3) point + GT light + albedo fix
submit 3 suzanne_point "1l_gtlight_albfix" "exp/suzanne_point" \
  "${POINT_IDX[@]}" "${LIGHT_GT[@]}" "${BRDF_ALBFIX[@]}"

# 4) point + GT light + albedo learned
submit 4 suzanne_point "1l_gtlight_alblearn" "exp/suzanne_point" \
  "${POINT_IDX[@]}" "${LIGHT_GT[@]}" "${BRDF_ALBLEARN[@]}"

# 7) point + learned light + albedo fix
submit 7 suzanne_point "1l_learnlight_albfix" "exp/suzanne_point" \
  "${POINT_IDX[@]}" "${LIGHT_LEARN_POINT[@]}" "${BRDF_ALBFIX[@]}"

# 8) point + learned light + albedo learned
submit 8 suzanne_point "1l_learnlight_alblearn" "exp/suzanne_point" \
  "${POINT_IDX[@]}" "${LIGHT_LEARN_POINT[@]}" "${BRDF_ALBLEARN[@]}"

echo ""
echo "All 8 runs submitted."
