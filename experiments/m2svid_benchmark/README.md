# M2SVid Benchmark Experiment Plan

## Goal

Measure the practical inference cost and failure modes of the M2SVid full pipeline:

```text
input video -> DepthCrafter depth -> depth-based warping -> M2SVid refinement -> stereo video
```

The first benchmark focuses on resource usage and throughput rather than paper-level quantitative quality.

## Recommended first run

Prepare 10-12 short clips under:

```text
experiments/clips/
```

Each clip can be arbitrary length/resolution; the benchmark script normalizes to a fixed FPS, frame count, and max resolution.

Run on Lightning from repo root:

```bash
python scripts/benchmark_m2svid.py \
  --input-dir experiments/clips \
  --output-dir experiments/m2svid_benchmark/runs \
  --max-res 512 \
  --frames 24 \
  --fps 8 \
  --depth-steps 25 \
  --disparity-perc 0.05 \
  --patch-xformers-fallbacks \
  --use-blackwell-config
```

For a quick smoke test:

```bash
python scripts/benchmark_m2svid.py \
  --input-dir experiments/clips \
  --output-dir experiments/m2svid_benchmark/runs_smoke \
  --max-res 384 \
  --frames 8 \
  --fps 8 \
  --depth-steps 5 \
  --patch-xformers-fallbacks \
  --use-blackwell-config \
  --only-first-n-runs 1
```

## Scaling sweep

After the pilot works, run a small scaling sweep on 3 representative clips:

```bash
python scripts/benchmark_m2svid.py \
  --input-dir experiments/clips \
  --output-dir experiments/m2svid_benchmark/runs_scaling \
  --max-res 384,512,768 \
  --frames 8,16,24,32 \
  --fps 8 \
  --depth-steps 25 \
  --disparity-perc 0.05 \
  --limit-clips 3 \
  --patch-xformers-fallbacks \
  --use-blackwell-config
```

This produces `3 clips × 3 resolutions × 4 frame counts = 36 runs`.

## Output structure

```text
experiments/m2svid_benchmark/
  results.csv
  runs/
    <run_id>/
      input.mp4
      gpu_monitor.csv
      metrics.json
      logs/
        00_preprocess.log
        01_depthcrafter.log
        02_warping.log
        03_m2svid_refine.log
      depthcrafter/*.npz
      reprojected/input_reprojected.mp4
      reprojected/input_reprojected_mask.mp4
      m2svid/*_generated.mp4
      m2svid/*_sbs.mp4
      m2svid/*_anaglyph.mp4
```

## Metrics captured

- Input metadata from `ffprobe`.
- Per-stage wall-clock time:
  - preprocess
  - DepthCrafter
  - warping
  - M2SVid refinement
  - total
- GPU monitor samples from `nvidia-smi`:
  - peak VRAM
  - mean VRAM
  - mean GPU utilization
  - peak power
  - peak temperature
- Success/failure and error message.

## Report outline

1. Introduction
   - Motivation: practical reproducibility and deployment cost of M2SVid.
2. Pipeline Overview
   - DepthCrafter estimates depth.
   - M2SVid `warping.py` creates coarse right view and mask.
   - M2SVid refinement generates final right view.
3. Experimental Setup
   - GPU, driver, CUDA, PyTorch, repo commit, checkpoints, xformers fallback config.
4. Benchmark Protocol
   - Dataset/clip selection, normalization, frame/resolution sweep.
5. Results
   - Runtime breakdown table.
   - Peak VRAM table.
   - Runtime vs resolution plot.
   - Runtime vs frame count plot.
   - Qualitative SBS examples.
6. Discussion
   - Bottlenecks, failure modes, xformers/Blackwell compatibility, cost estimate.
7. Limitations
   - No stereo GT in pilot; qualitative/proxy metrics only.
8. Future Work
   - KITTI/Sintel stereo quantitative evaluation, depth alternatives, persistent model loading optimization.

## Qualitative scoring rubric

For each successful run, optionally score 1-5:

| Metric | Description |
|---|---|
| Stereo plausibility | Does the generated right view create plausible depth? |
| Temporal consistency | Does it flicker across frames? |
| Boundary quality | Are foreground edges stable? |
| Disocclusion filling | Are holes repaired plausibly? |
| Artifact severity | Are distortions/ghosting severe? |

Save scores in a separate CSV, e.g. `qualitative_scores.csv`.
