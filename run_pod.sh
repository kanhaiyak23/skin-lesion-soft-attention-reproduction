#!/usr/bin/env bash
# One M4 run on a RunPod GPU pod (PyTorch template, network volume at /workspace).
#   LOSS=focal SEED=7 bash run_pod.sh
# Stages, each skipped if already done, so a restarted pod resumes:
#   1. install TF   2. download HAM10000 once to the shared volume   3. train + evaluate
# The augmented set (~1.3 GB) goes to the pod's local disk, which is much faster than the network volume.
# Keras BackupAndRestore resumes training mid-run. A budget guard stops after MAX_TRAIN_HOURS.
set -euo pipefail
: "${LOSS:?set LOSS=ce or LOSS=focal}" "${SEED:?set SEED}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-8}"
REPO=/workspace/repo
DATA=/workspace/data/ham10000
WORK=/workspace/m4
cd "$REPO"
mkdir -p "$WORK" "$(dirname "$DATA")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WORK/pod_${LOSS}_seed${SEED}.log"; }

if ! python -c "import tensorflow" 2>/dev/null; then
  log "installing TensorFlow"
  pip install -q --upgrade pip
  pip install -q "tensorflow[and-cuda]==2.20.*" opencv-python-headless pandas scikit-learn matplotlib kagglehub
fi
python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); assert g, 'no GPU visible'"

# several pods start at once: one downloads, the others wait on the lock
(
  flock 9
  if [ ! -f "$DATA/HAM10000_metadata.csv" ]; then
    log "downloading HAM10000 to the shared volume"
    python - "$DATA" <<'EOF'
import kagglehub, shutil, sys
src = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
shutil.copytree(src, sys.argv[1], dirs_exist_ok=True)
EOF
  fi
) 9>"$(dirname "$DATA")/.download.lock"

log "training loss=$LOSS seed=$SEED (guard: ${MAX_TRAIN_HOURS}h)"
timeout "$((MAX_TRAIN_HOURS * 3600))" python train.py --loss "$LOSS" --seed "$SEED" \
  --data-dir "$DATA" --work-dir "$WORK" --aug-dir /root 2>&1 | tee -a "$WORK/train_${LOSS}_seed${SEED}.log" || \
  log "training stopped (timeout or error) -- check the log"

log "DONE -- results in $WORK/runs/irv2_${LOSS}_paper_seed${SEED}; download them, then STOP the pod"
