#!/usr/bin/env bash
# ============================================================
# B200 Master Benchmark Script
# Runs all phases sequentially, commits results to git after each.
# Usage:
#   ./run_all.sh [--hf-token TOKEN] [--model MODEL] [--quick] [--skip-inference]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_TOKEN=""
MODEL="auto"
QUICK=""
SKIP_INFERENCE=0
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --model)    MODEL="$2";    shift 2 ;;
        --quick)    QUICK="--quick"; shift ;;
        --skip-inference) SKIP_INFERENCE=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
commit_results() {
    local msg="$1"
    cd "$REPO_ROOT"
    git add -A
    git diff --cached --quiet && echo "  (nothing to commit)" && return
    git commit -m "B200 benchmark: $msg [$(date '+%Y-%m-%d %H:%M')]"
    git push origin main || echo "  (push failed — check credentials)"
    log "Committed & pushed: $msg"
}

cd "$REPO_ROOT"

# ── Phase 0: Environment verification ────────────────────────────────────────
log "Phase 0: Environment Verification"
python3 -c "
import torch, pynvml, vllm
pynvml.nvmlInit()
n = pynvml.nvmlDeviceGetCount()
print(f'  torch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPUs: {n}  |  vLLM: {vllm.__version__}')
for i in range(n):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f'  GPU {i}: {name}  |  {mem.total/1e9:.0f} GB')
"

# ── Phase 1: System Profiling ─────────────────────────────────────────────────
log "Phase 1: System Microbenchmarks"
python3 "$SCRIPT_DIR/01_system_profile.py"
commit_results "Phase 1 system profile complete"

# ── Phase 2: LLM Inference Sweep ─────────────────────────────────────────────
if [[ $SKIP_INFERENCE -eq 0 ]]; then
    log "Phase 2: LLM Inference Sweep (1x GPU)"
    HF_ARG=""
    [[ -n "$HF_TOKEN" ]] && HF_ARG="--hf-token $HF_TOKEN"
    python3 "$SCRIPT_DIR/03_llm_inference_sweep.py" \
        --model "$MODEL" $HF_ARG --num-gpus 1 $QUICK
    commit_results "Phase 2 inference sweep 1x GPU"

    if [[ $NUM_GPUS -ge 2 ]]; then
        log "Phase 2b: LLM Inference Sweep (2x GPU)"
        python3 "$SCRIPT_DIR/03_llm_inference_sweep.py" \
            --model "$MODEL" $HF_ARG --num-gpus 2 $QUICK
        commit_results "Phase 2b inference sweep 2x GPU"
    fi
else
    log "Skipping Phase 2 (--skip-inference)"
fi

# ── Phase 3: Training Sweep ───────────────────────────────────────────────────
log "Phase 3: LLM Training Sweep"
python3 "$SCRIPT_DIR/04_llm_training_sweep.py" --num-gpus 1 $QUICK
commit_results "Phase 3 training sweep 1x GPU"

if [[ $NUM_GPUS -ge 2 ]]; then
    log "Phase 3b: Training Sweep (2x GPU)"
    python3 "$SCRIPT_DIR/04_llm_training_sweep.py" --num-gpus 2 $QUICK
    commit_results "Phase 3b training sweep 2x GPU"
fi

# ── Phase 4: Tier-1 Telemetry ────────────────────────────────────────────────
log "Phase 4: Tier-1 NVML Telemetry Tests"
python3 "$SCRIPT_DIR/02_tier1_telemetry.py"
commit_results "Phase 4 tier-1 telemetry complete"

# ── Phase 5: NVLink Comparison ───────────────────────────────────────────────
if [[ $NUM_GPUS -ge 2 ]]; then
    log "Phase 5: NVLink vs 1x GPU Comparison"
    python3 "$SCRIPT_DIR/05_nvlink_comparison.py"
    commit_results "Phase 5 NVLink comparison complete"
fi

# ── Phase 6: Generate Reports ─────────────────────────────────────────────────
log "Phase 6: Generating Markdown Reports"
python3 "$SCRIPT_DIR/06_generate_reports.py"
commit_results "Phase 6 all reports generated"

log "ALL PHASES COMPLETE"
echo "Results: $REPO_ROOT/results/"
echo "Graphs:  $REPO_ROOT/graphs/"
echo "Reports: $REPO_ROOT/reports/"
