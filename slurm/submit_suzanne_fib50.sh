#!/bin/bash
# ============================================================================
# submit_suzanne_fib50.sh
#
# Submit 12 training runs for synthetic_suzanne_point_colored_shadows_fib50:
#   6 strategies × 2 light models (point neuralBRDF + dir neuralBRDF)
#   All with shadows enabled.
#
# Strategies:
#   maxgray   – per view: light maximising mean gray in mask
#   mingray   – per view: light minimising mean gray (grazing angle)
#   medgray   – per view: light closest to median gray (balanced)
#   maxsobel  – per view: light maximising Sobel contrast (best for geometry)
#   1l_front  – fixed L00 for all views (frontal, low relief)
#   1l_maxvar – fixed Lk maximising cross-view gray variance
#
# Slurm policy:
#   1. Hold all currently pending jobs (they keep their queue positions)
#   2. Submit the 12 training jobs → they go straight to pending
#   3. Cancel currently running jobs to free GPU slots now
#   4. Sleep briefly, verify new jobs started
#   5. Resubmit cancelled jobs (they'll re-enter the queue)
#   6. Release all held jobs
#
# Usage:
#   bash slurm/submit_suzanne_fib50.sh
#   bash slurm/submit_suzanne_fib50.sh --dry-run   (only prints commands)
# ============================================================================

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p slurm/logs

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  echo "[DRY-RUN] No jobs will be submitted/cancelled."
fi

run() {
  if [[ $DRY_RUN -eq 1 ]]; then echo "  [dry] $*"; else eval "$@"; fi
}

DATASET="data/synthetic_suzanne_point_colored_shadows_fib50"
EXP_PATH="exp/suzanne_point_colored_shadows_fib50"

STRATEGIES=(maxgray mingray medgray maxsobel 1l_front 1l_maxvar)

# ── Common overrides ─────────────────────────────────────────────────────────
COMMON=(
  conf.model.use_shadow=true
  conf.exp.test_after_train=false
  "conf.dataset.predict_targets=[predict_mesh]"
)

BRDF_NEURAL=(
  conf.model.brdf.name=neural_brdf
)

LIGHT_LEARN_POINT=(
  conf.model.light.use_gt_light=false
  conf.model.light.freeze_light=false
  conf.model.light.light_pos_file=None
)

LIGHT_LEARN_DIR=(
  conf.model.light.use_gt_light=false
  conf.model.light.freeze_light=false
  conf.model.light.light_dir_file=None
)

# ── per_image_light: one independent light param per image (not shared by index)
# Activated for per-view strategies (maxgray/mingray/medgray/maxsobel);
# NOT for fixed-light strategies (1l_front/1l_maxvar) where all images
# intentionally share the same light.
PER_IMAGE_LIGHT=(
  conf.dataset.per_image_light=true
  conf.model.light.per_image_light=true
)

# ── Index file overrides for a given strategy ─────────────────────────────────
idx_overrides() {
  local strat="$1"
  echo "conf.dataset.train.view_light_index_file=${DATASET}/view_light_idx_${strat}_train.txt"
  echo "conf.dataset.val.view_light_index_file=${DATASET}/view_light_idx_${strat}_val.txt"
  echo "conf.dataset.test.view_light_index_file=${DATASET}/view_light_idx_${strat}_test.txt"
  echo "conf.dataset.predict_mesh.view_light_index_file=${DATASET}/view_light_idx_${strat}_val.txt"
}

# ── Submit helper ─────────────────────────────────────────────────────────────
SUBMITTED_IDS=()
submit() {
  local config="$1"; shift
  local tag="$1"; shift

  local cmd="sbatch --parsable slurm/run_suzanne.sh ${config}"
  cmd+=" conf.exp.exp_path=${EXP_PATH}"
  cmd+=" conf.exp.tag=_${tag}"
  for arg in "${COMMON[@]}"; do cmd+=" ${arg}"; done
  for arg in "$@"; do cmd+=" ${arg}"; done

  echo "  sbatch: ${tag}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry] ${cmd}"
  else
    JID=$(eval "${cmd}")
    SUBMITTED_IDS+=("${JID}")
    echo "    → Job ID: ${JID}"
  fi
}

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Hold all currently pending jobs
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ STEP 1: Holding all pending jobs ═══"
PENDING_JOBS=$(squeue -u "$USER" -h -t PD -o "%i" 2>/dev/null | tr '\n' ' ')
if [[ -n "${PENDING_JOBS}" ]]; then
  echo "  Pending: ${PENDING_JOBS}"
  for jid in ${PENDING_JOBS}; do
    run scontrol hold "${jid}" 2>/dev/null || true
  done
  echo "  All pending jobs held."
else
  echo "  No pending jobs."
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Submit 12 training runs
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ STEP 2: Submitting 12 training runs ═══"

for strat in "${STRATEGIES[@]}"; do
  IDX=($(idx_overrides "${strat}"))

  # Per-view strategies need independent light params per image
  case "${strat}" in
    maxgray|mingray|medgray|maxsobel) EXTRA_LIGHT=("${PER_IMAGE_LIGHT[@]}") ;;
    *) EXTRA_LIGHT=() ;;
  esac

  # Point model
  submit suzanne_point_fib50 \
    "50v50l_learnlight_point_neuralbrdf_shadow_${strat}" \
    "${IDX[@]}" "${LIGHT_LEARN_POINT[@]}" "${BRDF_NEURAL[@]}" "${EXTRA_LIGHT[@]}"
  # Directional model (same dataset, different light model)
  submit suzanne_dir_fib50 \
    "50v50l_learnlight_dir_neuralbrdf_shadow_${strat}" \
    "${IDX[@]}" "${LIGHT_LEARN_DIR[@]}" "${BRDF_NEURAL[@]}" "${EXTRA_LIGHT[@]}"
done

echo ""
echo "  Submitted ${#SUBMITTED_IDS[@]} jobs: ${SUBMITTED_IDS[*]:-[dry-run]}"

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: Jobs are submitted and pending. Free slots manually if needed.
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ STEP 3: Jobs submitted. Free GPU slots manually if needed. ═══"
echo "  New jobs are pending. To prioritize them:"
echo "  - They will run as slots free up from held jobs being deprioritized"
echo "  - Or manually cancel specific running jobs to free slots"

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: Release all held jobs
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ STEP 6: Releasing held jobs ═══"
if [[ -n "${PENDING_JOBS}" ]]; then
  for jid in ${PENDING_JOBS}; do
    run scontrol release "${jid}" 2>/dev/null || true
  done
  echo "  Released: ${PENDING_JOBS}"
else
  echo "  Nothing to release."
fi

# ════════════════════════════════════════════════════════════════════════════
# Done
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ All done ═══"
echo "  Submitted IDs : ${SUBMITTED_IDS[*]:-[dry-run]}"
echo ""
echo "  To chain eval after all training jobs complete, run:"
echo "    DEP=\$(echo ${SUBMITTED_IDS[*]:-JOBID1 JOBID2} | tr ' ' ':')"
echo "    sbatch --dependency=afterok:\${DEP} slurm/run_eval_fib50.sh"
echo ""
echo "  Monitor: squeue -u $USER"
