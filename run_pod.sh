#!/usr/bin/env bash
# One M4 run on a RunPod GPU pod (PyTorch template, volume at /workspace).
#   LOSS=focal SEED=7 HF_TOKEN=... HF_REPO=user/m4-runs bash run_pod.sh
# Stages, each skipped if already done, so a restarted pod resumes:
#   1. install TF   2. download HAM10000 (once, under a lock)   3. train + evaluate
# The augmented set (~1.3 GB) goes to the pod's local disk. Keras BackupAndRestore resumes training
# mid-run. A budget guard stops after MAX_TRAIN_HOURS.
# If HF_REPO is set, logs and results are pushed to that (private) Hugging Face dataset every
# SYNC_EVERY seconds and once more on exit, under <loss>_seed<seed>/, so they can be read without
# reaching the pod itself. Model weights are pushed only at the end.
set -euo pipefail
: "${LOSS:?set LOSS=ce or LOSS=focal}" "${SEED:?set SEED}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-8}"
SYNC_EVERY="${SYNC_EVERY:-600}"
REPO=/workspace/repo
DATA=/workspace/data/ham10000
WORK=/workspace/m4
TAG="${LOSS}_seed${SEED}$([ "${QUICK_CHECK:-0}" = 1 ] && echo _quick || true)"
cd "$REPO"
mkdir -p "$WORK" "$(dirname "$DATA")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WORK/pod_${TAG}.log"; }

sync_hf() {  # $1 = "final" to include the model weights
  [ -n "${HF_REPO:-}" ] || return 0
  python - "$1" <<EOF || true
import sys
from huggingface_hub import HfApi
ignore = ["**/backup/**"] + ([] if sys.argv[1] == "final" else ["**/*.h5"])
HfApi().upload_folder(repo_id="$HF_REPO", repo_type="dataset", folder_path="$WORK",
                      path_in_repo="$TAG", ignore_patterns=ignore, commit_message="$TAG ${1}")
EOF
}

if [ -n "${HF_REPO:-}" ]; then
  pip install -q huggingface_hub
  (while true; do sleep "$SYNC_EVERY"; sync_hf periodic; done) &
  trap 'log "pod script exiting"; sync_hf final' EXIT
fi

if ! python -c "import tensorflow" 2>/dev/null; then
  log "installing TensorFlow"
  pip install -q --upgrade pip
  pip install -q "tensorflow[and-cuda]==2.20.*" opencv-python-headless pandas scikit-learn matplotlib kagglehub
fi
python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); assert g, 'no GPU visible'" 2>&1 | tee -a "$WORK/pod_${TAG}.log"
sync_hf periodic

# several pods may share one volume: one downloads, the others wait on the lock
(
  flock 9
  if [ ! -f "$DATA/HAM10000_metadata.csv" ]; then
    log "downloading HAM10000"
    python - "$DATA" <<'EOF'
import kagglehub, shutil, sys
src = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
shutil.copytree(src, sys.argv[1], dirs_exist_ok=True)
EOF
  fi
) 9>"$(dirname "$DATA")/.download.lock"
log "dataset ready: $(find "$DATA" -name '*.jpg' | wc -l) jpg files"

log "training loss=$LOSS seed=$SEED (guard: ${MAX_TRAIN_HOURS}h)"
timeout "$((MAX_TRAIN_HOURS * 3600))" python train.py --loss "$LOSS" --seed "$SEED" \
  --data-dir "$DATA" --work-dir "$WORK" --aug-dir /root 2>&1 | tee -a "$WORK/train_${TAG}.log" || \
  log "training stopped (timeout or error) -- check the log"

log "DONE -- results in $WORK/runs/; STOP the pod"
