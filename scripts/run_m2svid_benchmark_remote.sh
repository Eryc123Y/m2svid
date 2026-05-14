#!/usr/bin/env bash
set -euo pipefail

# One-command Lightning runner for the M2SVid benchmark.
# Run from repo root on Lightning:
#   ./scripts/run_m2svid_benchmark_remote.sh smoke
#   ./scripts/run_m2svid_benchmark_remote.sh pilot
#   ./scripts/run_m2svid_benchmark_remote.sh scaling
#   ./scripts/run_m2svid_benchmark_remote.sh all

MODE="${1:-pilot}"
ROOT="${M2SVID_ROOT:-/teamspace/studios/this_studio}"
cd "$ROOT"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "Repo root: $ROOT"
log "Mode: $MODE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; GPU metrics will be unavailable"
fi

log "Checking system tools"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y ffmpeg
fi

log "Checking benchmark clips"
if [ ! -d experiments/source_datasets/intel_sample_videos/.git ]; then
  mkdir -p experiments/source_datasets
  git clone --depth 1 https://github.com/intel-iot-devkit/sample-videos.git experiments/source_datasets/intel_sample_videos
fi
mkdir -p experiments/clips
find experiments/clips -maxdepth 1 -type f -name '*.mp4' -delete || true
find experiments/clips -maxdepth 1 -type l -name '*.mp4' -delete || true

# Curated small sample set from Intel sample-videos: varied but lightweight.
# These are not stereo-GT clips; they are for resource/throughput/failure-mode benchmarking.
while IFS='|' read -r out src; do
  [ -z "$out" ] && continue
  ln -sf "../source_datasets/intel_sample_videos/$src" "experiments/clips/$out"
done <<'EOF'
close_object__bottle-detection.mp4|bottle-detection.mp4
driving__car-detection.mp4|car-detection.mp4
human_motion__people-detection.mp4|people-detection.mp4
mixed_motion__person-bicycle-car-detection.mp4|person-bicycle-car-detection.mp4
indoor_aisle__store-aisle-detection.mp4|store-aisle-detection.mp4
workspace__worker-zone-detection.mp4|worker-zone-detection.mp4
indoor_classroom__classroom.mp4|classroom.mp4
objects_texture__fruit-and-vegetable-detection.mp4|fruit-and-vegetable-detection.mp4
EOF

log "Clip manifest"
for f in experiments/clips/*.mp4; do
  printf '%s ' "$f"
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration,r_frame_rate -of csv=p=0 "$f" || true
done | tee experiments/m2svid_benchmark_clip_manifest.txt

log "Checking Python scripts"
python3 -m py_compile scripts/benchmark_m2svid.py scripts/analyze_m2svid_benchmark.py

COMMON=(
  --input-dir experiments/clips
  --fps 8
  --disparity-perc 0.05
  --patch-xformers-fallbacks
  --use-blackwell-config
)

case "$MODE" in
  smoke)
    log "Running smoke benchmark: 1 clip, 8 frames, 384p-ish, 5 depth steps"
    python3 scripts/benchmark_m2svid.py \
      "${COMMON[@]}" \
      --output-dir experiments/m2svid_benchmark/runs_smoke \
      --max-res 384 \
      --frames 8 \
      --depth-steps 5 \
      --only-first-n-runs 1
    ;;
  pilot)
    log "Running pilot benchmark: 8 clips, 24 frames, max_res 512"
    python3 scripts/benchmark_m2svid.py \
      "${COMMON[@]}" \
      --output-dir experiments/m2svid_benchmark/runs \
      --max-res 512 \
      --frames 24 \
      --depth-steps 25
    ;;
  scaling)
    log "Running scaling benchmark: first 3 clips × resolutions × frame counts"
    python3 scripts/benchmark_m2svid.py \
      "${COMMON[@]}" \
      --output-dir experiments/m2svid_benchmark/runs_scaling \
      --max-res 384,512,768 \
      --frames 8,16,24,32 \
      --depth-steps 25 \
      --limit-clips 3
    ;;
  all)
    "$0" smoke
    "$0" pilot
    "$0" scaling
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|scaling|all}" >&2
    exit 2
    ;;
esac

log "Benchmark command finished for mode=$MODE"
log "Results CSV files"
find experiments/m2svid_benchmark -maxdepth 3 -name 'results.csv' -print -exec tail -n 5 {} \; || true

log "Next: copy experiments/m2svid_benchmark back to local for report writing"
