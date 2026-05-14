#set document(
  title: "M2SVid Inference Benchmark Report",
  author: "Eric Yang",
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.2cm),
  numbering: "1",
)

#set text(size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show link: set text(fill: blue)

#let figdir = "../experiments/m2svid_benchmark/figures_pilot"

#align(center)[
  #text(size: 20pt, weight: "bold")[M2SVid Inference Benchmark Report]

  #v(0.4em)
  #text(size: 11pt)[Pilot resource and throughput evaluation for monocular-to-stereo video generation]

  #v(0.8em)
  Eric Yang \
  #datetime.today().display("[year]-[month]-[day]")
]

#v(1em)

= Executive Summary

This report evaluates the practical inference cost of the M2SVid full pipeline on a remote Lightning AI GPU instance. The benchmark runs the complete monocular-to-stereo video generation pipeline:

#align(center)[
  `input video -> DepthCrafter depth -> geometric warping -> M2SVid refinement -> stereo video`
]

The pilot benchmark used 8 short video clips, each normalized to 24 frames at 8 FPS and padded to a 512 × 512 canvas. All 8 runs completed successfully on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU.

Key findings:

- *Success rate:* 8 / 8 pilot clips completed successfully.
- *Runtime:* mean 54.96 s per 24-frame clip, with a narrow range of 54.76--55.37 s.
- *Throughput:* mean 0.437 processed FPS, or 2.29 s per frame.
- *Peak VRAM:* 24,847 MiB, approximately 24.3 GiB, stable across clips.
- *Main bottleneck:* M2SVid refinement, mean 33.04 s, approximately 60% of total runtime.
- *Second bottleneck:* DepthCrafter, mean 17.87 s, approximately 32.5% of total runtime.
- *Warping cost:* low, mean 3.86 s, approximately 7% of total runtime.

The results suggest that, for fixed frame count and resolution, runtime and memory are primarily determined by model architecture and input shape rather than scene content. The current script-based pipeline is reliable but not optimized for throughput because it repeatedly loads models and invokes each stage as a separate subprocess.

= Pipeline Overview

M2SVid's full inference pipeline consists of three computational stages after input normalization.

== Depth Estimation

DepthCrafter estimates per-frame monocular depth from the input video. In the benchmark, each clip is first normalized to a fixed frame count, FPS, and square padded canvas. DepthCrafter then writes a `.npz` file containing the relative depth sequence.

== Geometric Warping

The repository's `warping.py` uses the DepthCrafter depth map to perform depth-based horizontal reprojection. It produces:

- a coarse right-view video, `input_reprojected.mp4`; and
- an inpainting mask, `input_reprojected_mask.mp4`.

This stage is deterministic geometry rather than neural generation. Pixels from the left view are shifted according to the estimated disparity; holes caused by disocclusion become the mask passed to M2SVid.

== M2SVid Refinement

`inpaint_and_refine.py` receives the original left video, the coarse right view, and the mask. It then uses the M2SVid diffusion model to synthesize a refined right view and output side-by-side and anaglyph visualizations.

= Experimental Setup

== Hardware and Runtime

The pilot was run on a Lightning AI remote instance with:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU memory: approximately 97,887 MiB
- Driver: 580.142
- CUDA runtime reported by PyTorch: CUDA 13.0
- Python: 3.12.11 in the Lightning `cloudspace` environment
- PyTorch: 2.12.0+cu130

== Repository and Configuration

The benchmark used Eric's M2SVid fork:

#block(fill: luma(245), inset: 8pt, radius: 4pt)[
  `https://github.com/Eryc123Y/m2svid.git`
]

The canonical local repository path was:

#block(fill: luma(245), inset: 8pt, radius: 4pt)[
  `/Users/eric/GitHub/m2svid`
]

The Lightning source root was:

#block(fill: luma(245), inset: 8pt, radius: 4pt)[
  `/teamspace/studios/this_studio`
]

A Blackwell-compatible configuration was used:

#block(fill: luma(245), inset: 8pt, radius: 4pt)[
  `configs/m2svid_no_xformers_blackwell.yaml`
]

This configuration replaces `attn_type: vanilla-xformers` with `attn_type: vanilla` for the Hi3D autoencoder attention blocks. This avoids xformers kernel incompatibility on Blackwell GPUs.

== Compatibility Fixes

Several compatibility adjustments were necessary for the current Lightning environment:

- `torchvision.io.write_video` is unavailable in the installed torchvision version, so video saving was patched to use `imageio.mimwrite`.
- DepthCrafter's optional xformers attention path was patched to catch `NotImplementedError` and `RuntimeError`, falling back to attention slicing.
- Benchmark inputs are padded to square canvases, e.g. 512 × 512, to avoid U-Net skip-connection shape mismatches such as `Expected size 8 but got size 7`.
- The benchmark runner uses `/home/zeus/miniconda3/envs/cloudspace/bin/python` instead of system `python3`, because dependencies are installed in the Lightning cloudspace environment.

= Benchmark Protocol

== Dataset

The pilot used 8 clips from the Intel sample video collection. The selected clips cover a small range of content types: close objects, driving, human motion, mixed foreground objects, indoor aisle scenes, workspace scenes, classrooms, and texture-heavy object scenes.

#table(
  columns: (2.2fr, 2.2fr),
  inset: 5pt,
  stroke: 0.4pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*Clip ID*], [*Scene Type*],
  [`close_object__bottle-detection`], [close object],
  [`driving__car-detection`], [driving / outdoor motion],
  [`human_motion__people-detection`], [human motion],
  [`mixed_motion__person-bicycle-car-detection`], [mixed foreground objects],
  [`indoor_aisle__store-aisle-detection`], [indoor aisle / perspective depth],
  [`workspace__worker-zone-detection`], [workspace scene],
  [`indoor_classroom__classroom`], [indoor high-resolution scene],
  [`objects_texture__fruit-and-vegetable-detection`], [texture-heavy objects],
)

== Input Normalization

Each clip was normalized to:

- frame count: 24 frames
- FPS: 8
- duration after normalization: 3 seconds
- spatial canvas: 512 × 512
- depth denoising steps: 25
- disparity percentage: 0.05

The square canvas is intentionally used for stability. Earlier 16:9 preprocessing, e.g. 512 × 288 or 384 × 216, can create incompatible tensor sizes in the U-Net skip connections.

== Metrics

For each run, the benchmark records:

- per-stage wall-clock time;
- total runtime;
- peak GPU memory from `nvidia-smi` sampling;
- mean GPU utilization;
- peak power and temperature;
- output paths and success/failure status.

Derived metrics include processed FPS, seconds per frame, and GPU-hours per minute of input video.

= Results

== Success Rate

All 8 pilot runs completed successfully. No clip failed during DepthCrafter, warping, or M2SVid refinement.

#table(
  columns: (1.6fr, 1fr),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*Metric*], [*Value*],
  [Pilot runs], [8],
  [Successful runs], [8],
  [Failed runs], [0],
  [Input shape], [24 frames, 512 × 512, 8 FPS],
)

== Runtime Breakdown

#figure(
  image(figdir + "/runtime_breakdown_by_clip.png", width: 100%),
  caption: [Runtime breakdown by clip. M2SVid refinement dominates total runtime, followed by DepthCrafter. Warping is comparatively small.],
) <fig:runtime-breakdown>

The runtime is highly stable across clips. Mean total runtime is 54.96 s, with only approximately 0.61 s difference between the fastest and slowest run. This indicates that runtime is primarily controlled by the fixed input dimensions and model compute rather than scene content.

#figure(
  image(figdir + "/mean_stage_runtime.png", width: 80%),
  caption: [Mean runtime per pipeline stage. M2SVid refinement accounts for approximately 60% of total runtime.],
) <fig:mean-stage-runtime>

#table(
  columns: (1.7fr, 1fr, 1fr),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*Stage*], [*Mean Time*], [*Approx. Share*],
  [Preprocess], [0.10 s], [less than 1%],
  [DepthCrafter], [17.87 s], [32.5%],
  [Warping], [3.86 s], [7.0%],
  [M2SVid refinement], [33.04 s], [60.1%],
  [Total], [54.96 s], [100%],
)

== Throughput

The mean throughput is 0.437 processed FPS. Since each benchmark clip contains 24 frames, a 3-second input video takes approximately 55 seconds to process.

#table(
  columns: (1.8fr, 1fr),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*Metric*], [*Value*],
  [Mean total runtime], [54.96 s],
  [Mean seconds per frame], [2.29 s/frame],
  [Mean processed FPS], [0.437 FPS],
  [Runtime relative to real-time], [approximately 18.3× slower than real-time],
  [Mean GPU-hours per video minute], [0.305 GPU-hours/min],
)

== GPU Memory and Utilization

#figure(
  image(figdir + "/peak_vram_by_clip.png", width: 100%),
  caption: [Peak VRAM by clip. Peak memory is constant across pilot clips at approximately 24.3 GiB.],
) <fig:peak-vram>

Peak memory is identical across the 8 clips at 24,847 MiB. This suggests that for this setting memory is determined almost entirely by frame count, resolution, and model architecture. Content variation has little effect on the peak allocation.

#figure(
  image(figdir + "/mean_gpu_util_by_clip.png", width: 100%),
  caption: [Mean GPU utilization by clip. Utilization is relatively low, suggesting overhead from model loading, subprocess boundaries, CPU processing, and video I/O.],
) <fig:gpu-util>

Mean GPU utilization is only 16.81%. This does not mean the GPU is unnecessary; the peak memory and peak power are substantial. However, the current script-based pipeline is not throughput-optimized. It repeatedly loads models and uses separate Python processes for each stage.

#table(
  columns: (2fr, 1fr),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*GPU Metric*], [*Pilot Result*],
  [Peak VRAM], [24,847 MiB, approximately 24.3 GiB],
  [Mean GPU utilization], [16.81%],
  [Peak power], [approximately 555.85 W],
  [Peak temperature], [45 °C],
)

== Per-Clip Summary

#table(
  columns: (2.1fr, 0.8fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  inset: 4pt,
  stroke: 0.35pt,
  fill: (_, row) => if row == 0 { luma(235) },
  [*Clip*], [*Status*], [*Total*], [*Depth*], [*Warp*], [*M2SVid*],
  [`close_object`], [success], [55.37 s], [18.05 s], [3.91 s], [33.27 s],
  [`driving`], [success], [55.04 s], [17.82 s], [3.91 s], [33.16 s],
  [`human_motion`], [success], [54.92 s], [17.90 s], [3.80 s], [33.06 s],
  [`indoor_aisle`], [success], [54.82 s], [17.87 s], [3.85 s], [32.91 s],
  [`indoor_classroom`], [success], [54.82 s], [17.84 s], [3.84 s], [32.90 s],
  [`mixed_motion`], [success], [54.76 s], [17.83 s], [3.84 s], [32.94 s],
  [`objects_texture`], [success], [54.97 s], [17.88 s], [3.88 s], [33.04 s],
  [`workspace`], [success], [54.94 s], [17.79 s], [3.82 s], [33.06 s],
)

== Qualitative Outputs

#figure(
  image(figdir + "/qualitative_sbs_montage.png", width: 100%),
  caption: [Side-by-side qualitative examples extracted from the benchmark outputs. Each tile shows the generated stereo visualization for a representative clip.],
) <fig:qualitative>

The montage provides a quick qualitative sanity check. It should be used as a report figure for visual examples, not as a rigorous perceptual metric. A later study should add structured human scoring or stereo ground-truth metrics.

= Discussion

== Bottlenecks

The primary bottleneck is M2SVid refinement, averaging 33.04 s per 24-frame clip. DepthCrafter is the second-largest cost at 17.87 s. Warping is much cheaper at 3.86 s and is unlikely to be the main target for optimization.

A practical optimization path is therefore:

+ Keep M2SVid and DepthCrafter models resident in memory rather than invoking subprocesses for every clip.
+ Avoid repeated model initialization across clips.
+ Batch or pipeline video preprocessing and postprocessing.
+ Investigate whether lower DepthCrafter denoising steps preserve acceptable quality.
+ Run a resolution/frame-count scaling study to identify the best cost-quality trade-off.

== Memory Implications

The pilot requires approximately 24.3 GiB peak VRAM at 24 frames and 512 × 512. This explains earlier failures on smaller GPUs such as L4 23 GiB, where the official 512 × 512 demo is close to or beyond the practical limit. A 48 GiB L40S or larger GPU should be sufficient for this pilot configuration, while the 96 GiB RTX PRO 6000 Blackwell leaves substantial headroom.

== Content Dependence

All clips have very similar runtime and identical peak VRAM. This implies that, at fixed resolution and frame count, content type has minimal effect on compute cost. Content is still important for qualitative output quality, especially around disocclusions, thin structures, human boundaries, and low-texture regions.

== Environment Lessons

The reproduction required several environment-specific fixes. The most important lesson is that optional acceleration paths such as xformers should be treated as replaceable. On newer Blackwell GPUs, xformers kernels may not support the device capability or attention shape. Falling back to vanilla attention is slower or more memory-intensive but much more robust when sufficient VRAM is available.

= Limitations

This pilot benchmark has several limitations:

- The selected clips do not provide stereo ground truth, so this report focuses on resource and throughput rather than objective stereo quality.
- All pilot runs use the same frame count and resolution, so the report does not yet provide scaling curves.
- The benchmark uses a script-based pipeline with repeated model loading. A persistent service implementation would likely improve throughput.
- The qualitative montage is only a sanity check. It does not replace human evaluation or quantitative metrics such as LPIPS, SSIM, temporal consistency measures, or stereo reconstruction error.
- The Blackwell no-xformers configuration may not exactly match the original environment expected by the paper, although it is a pragmatic compatibility fix.

= Future Work

The next experimental steps are:

+ Run the scaling benchmark with several frame counts and resolutions:
  - frames: 8, 16, 24, 32
  - resolutions: 384, 512, 768
+ Add stereo or multi-view datasets with reference right views, such as KITTI stereo or selected synthetic data.
+ Evaluate quality using PSNR, SSIM, LPIPS, and temporal consistency where ground truth is available.
+ Compare DepthCrafter against alternative video depth estimators.
+ Measure the effect of DepthCrafter denoising steps on runtime and final stereo quality.
+ Implement a persistent inference runner to reduce model loading overhead.
+ Evaluate cost per video minute on L40S, A100, and RTX PRO 6000 class GPUs.

= Conclusion

The M2SVid full pipeline is now reproducible and benchmarkable on the Lightning GPU setup. For 24-frame, 512 × 512 clips, the pipeline consistently completes in approximately 55 seconds with a peak VRAM requirement of 24.3 GiB. The dominant cost is M2SVid refinement, followed by DepthCrafter depth estimation, while geometric warping is comparatively inexpensive.

The benchmark shows that M2SVid is feasible for offline monocular-to-stereo video generation, but it is not close to real-time in the current script-based setup. Future work should focus on scaling behavior, persistent model loading, and quality evaluation on stereo-ground-truth datasets.
