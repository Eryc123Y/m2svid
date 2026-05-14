#!/usr/bin/env python3
"""Batch benchmark runner for M2SVid full inference.

Runs the full pipeline for a directory of clips:

  input video -> DepthCrafter depth -> geometry warping -> M2SVid refinement

For each run it records wall-clock time, GPU monitor samples from nvidia-smi,
input metadata, output paths, success/failure status, and writes both per-run
metrics.json and an aggregate results.csv.

Typical Lightning usage from repo root:

  python scripts/benchmark_m2svid.py \
    --input-dir experiments/clips \
    --output-dir experiments/m2svid_benchmark/runs \
    --max-res 512 \
    --frames 24 \
    --fps 8 \
    --depth-steps 25 \
    --model-config configs/m2svid_no_xformers_blackwell.yaml \
    --patch-xformers-fallbacks

Scaling sweep example:

  python scripts/benchmark_m2svid.py \
    --input-dir experiments/clips \
    --output-dir experiments/m2svid_benchmark/runs \
    --max-res 384,512,768 \
    --frames 8,16,24,32 \
    --fps 8 \
    --limit-clips 3 \
    --patch-xformers-fallbacks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int
    elapsed_sec: float
    log_path: str


@dataclass
class RunSpec:
    clip_id: str
    source_path: str
    run_id: str
    max_res: int
    frames: int
    fps: float
    depth_steps: int
    disparity_perc: float
    model_config: str


def run(cmd: Sequence[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> CommandResult:
    """Run a subprocess, teeing stdout/stderr to a log file."""
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, cmd)) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
    return CommandResult(list(map(str, cmd)), rc, time.perf_counter() - start, str(log_path))


def ffprobe_json(video: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration,codec_name",
        "-of", "json", str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        return json.loads(out)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def parse_csv_list(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def discover_clips(input_dir: Path, limit: int | None = None) -> list[Path]:
    clips = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if limit is not None:
        clips = clips[:limit]
    return clips


def safe_slug(path: Path) -> str:
    stem = path.stem.lower()
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() or ch in "-_" else "-")
    slug = "".join(keep).strip("-") or "clip"
    return slug[:80]


def start_gpu_monitor(csv_path: Path, interval_ms: int = 500) -> subprocess.Popen | None:
    if shutil.which("nvidia-smi") is None:
        return None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
        "--format=csv,nounits",
        "-lms", str(interval_ms),
    ]
    f = csv_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    # Attach file object so it is not GC'd; close in stop_gpu_monitor.
    proc._m2svid_log_file = f  # type: ignore[attr-defined]
    return proc


def stop_gpu_monitor(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    f = getattr(proc, "_m2svid_log_file", None)
    if f:
        f.close()


def summarize_gpu_monitor(csv_path: Path) -> dict:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}
    rows = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:  # noqa: BLE001
        return {"gpu_monitor_error": repr(e)}
    if not rows:
        return {}

    def num(row: dict, key: str) -> float | None:
        # nvidia-smi headers may contain spaces, e.g. "memory.used [MiB]".
        for k, v in row.items():
            if k and key in k:
                try:
                    return float(str(v).strip())
                except Exception:
                    return None
        return None

    mem = [x for x in (num(r, "memory.used") for r in rows) if x is not None]
    util = [x for x in (num(r, "utilization.gpu") for r in rows) if x is not None]
    power = [x for x in (num(r, "power.draw") for r in rows) if x is not None]
    temp = [x for x in (num(r, "temperature.gpu") for r in rows) if x is not None]
    names = [r.get(" name") or r.get("name") or r.get(" name ") for r in rows]
    names = [n.strip() for n in names if n]
    return {
        "gpu_name": names[0] if names else None,
        "gpu_samples": len(rows),
        "gpu_peak_memory_mib": max(mem) if mem else None,
        "gpu_mean_memory_mib": sum(mem) / len(mem) if mem else None,
        "gpu_mean_util_percent": sum(util) / len(util) if util else None,
        "gpu_peak_power_w": max(power) if power else None,
        "gpu_peak_temp_c": max(temp) if temp else None,
    }


def patch_depthcrafter_xformers(repo_root: Path) -> bool:
    p = repo_root / "third_party/DepthCrafter/depthcrafter/inference.py"
    if not p.exists():
        return False
    s = p.read_text(encoding="utf-8")
    old = "except (ImportError, ModuleNotFoundError, AttributeError) as e:"
    new = "except (ImportError, ModuleNotFoundError, AttributeError, NotImplementedError, RuntimeError) as e:"
    if old in s:
        p.write_text(s.replace(old, new), encoding="utf-8")
        return True
    return False


def create_no_xformers_config(repo_root: Path, base_config: Path, out_config: Path) -> bool:
    if not base_config.is_absolute():
        base_config = repo_root / base_config
    if not out_config.is_absolute():
        out_config = repo_root / out_config
    if out_config.exists():
        return False
    s = base_config.read_text(encoding="utf-8")
    s = s.replace("attn_type: vanilla-xformers", "attn_type: vanilla")
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(s, encoding="utf-8")
    return True


def preprocess_clip(src: Path, dst: Path, fps: float, frames: int, max_res: int, cwd: Path, log_path: Path) -> CommandResult:
    # Keep aspect ratio, make dimensions even, cap max side to max_res.
    vf = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),{max_res},-2)':'if(gt(iw,ih),-2,{max_res})',"
        f"trim=end_frame={frames},setpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info",
        "-i", str(src),
        "-vf", vf,
        "-an", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        str(dst),
    ]
    return run(cmd, cwd=cwd, log_path=log_path)


def run_one(spec: RunSpec, repo_root: Path, run_dir: Path, args: argparse.Namespace) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    gpu_csv = run_dir / "gpu_monitor.csv"
    input_video = run_dir / "input.mp4"
    depth_dir = run_dir / "depthcrafter"
    reprojected_dir = run_dir / "reprojected"
    m2svid_dir = run_dir / "m2svid"
    depth_dir.mkdir(exist_ok=True)
    reprojected_dir.mkdir(exist_ok=True)
    m2svid_dir.mkdir(exist_ok=True)

    metrics: dict = {
        **asdict(spec),
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "status": "running",
    }

    mon = start_gpu_monitor(gpu_csv, args.gpu_monitor_interval_ms)
    t0 = time.perf_counter()
    try:
        # 1) Normalize/trim input.
        r = preprocess_clip(Path(spec.source_path), input_video, spec.fps, spec.frames, spec.max_res, repo_root, logs / "00_preprocess.log")
        metrics["t_preprocess_sec"] = r.elapsed_sec
        if r.returncode != 0:
            raise RuntimeError(f"preprocess failed rc={r.returncode}")
        metrics["input_probe"] = ffprobe_json(input_video)

        env = {
            "PYTHONPATH": f"{repo_root}:{repo_root / 'third_party/DepthCrafter'}:{repo_root / 'third_party/Hi3D-Official'}:{repo_root / 'third_party/pytorch-msssim'}:{os.environ.get('PYTHONPATH', '')}",
            "PYTORCH_CUDA_ALLOC_CONF": args.cuda_alloc_conf,
        }

        # 2) DepthCrafter.
        depth_cmd = [
            sys.executable, "third_party/DepthCrafter/run.py",
            "--video-path", str(input_video),
            "--save_folder", str(depth_dir),
            "--save_npz", "True",
            "--num_inference_steps", str(spec.depth_steps),
            "--max_res", str(spec.max_res),
        ]
        r = run(depth_cmd, cwd=repo_root, log_path=logs / "01_depthcrafter.log", env=env)
        metrics["t_depthcrafter_sec"] = r.elapsed_sec
        if r.returncode != 0:
            raise RuntimeError(f"DepthCrafter failed rc={r.returncode}")
        depth_candidates = sorted(depth_dir.glob("*.npz"))
        if not depth_candidates:
            raise RuntimeError(f"DepthCrafter produced no .npz under {depth_dir}")
        depth_npz = depth_candidates[0]
        metrics["depth_npz"] = str(depth_npz)

        # 3) Warping.
        reprojected = reprojected_dir / "input_reprojected.mp4"
        mask = reprojected_dir / "input_reprojected_mask.mp4"
        warp_cmd = [
            sys.executable, "warping.py",
            "--video_path", str(input_video),
            "--depth_path", str(depth_npz),
            "--output_path_reprojected", str(reprojected),
            "--output_path_mask", str(mask),
            "--disparity_perc", str(spec.disparity_perc),
        ]
        r = run(warp_cmd, cwd=repo_root, log_path=logs / "02_warping.log", env=env)
        metrics["t_warping_sec"] = r.elapsed_sec
        if r.returncode != 0:
            raise RuntimeError(f"warping failed rc={r.returncode}")
        metrics["reprojected_video"] = str(reprojected)
        metrics["mask_video"] = str(mask)

        # 4) M2SVid refinement.
        refine_cmd = [
            sys.executable, "inpaint_and_refine.py",
            "--mask_antialias", "0",
            "--model_config", spec.model_config,
            "--ckpt", args.ckpt,
            "--video_path", str(input_video),
            "--reprojected_path", str(reprojected),
            "--reprojected_mask_path", str(mask),
            "--output_folder", str(m2svid_dir),
        ]
        r = run(refine_cmd, cwd=repo_root, log_path=logs / "03_m2svid_refine.log", env=env)
        metrics["t_m2svid_sec"] = r.elapsed_sec
        if r.returncode != 0:
            raise RuntimeError(f"M2SVid refinement failed rc={r.returncode}")
        metrics["m2svid_output_dir"] = str(m2svid_dir)
        metrics["generated_video"] = str(next(iter(sorted(m2svid_dir.glob("*_generated.mp4"))), ""))
        metrics["sbs_video"] = str(next(iter(sorted(m2svid_dir.glob("*_sbs.mp4"))), ""))
        metrics["anaglyph_video"] = str(next(iter(sorted(m2svid_dir.glob("*_anaglyph.mp4"))), ""))
        metrics["status"] = "success"

    except Exception as e:  # noqa: BLE001
        metrics["status"] = "failed"
        metrics["error"] = repr(e)
        print(f"[FAILED] {spec.run_id}: {e}", file=sys.stderr)
    finally:
        metrics["t_total_sec"] = time.perf_counter() - t0
        stop_gpu_monitor(mon)
        metrics.update(summarize_gpu_monitor(gpu_csv))
        metrics["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def append_results(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Stable useful subset first, then remaining keys.
    preferred = [
        "run_id", "status", "clip_id", "source_path", "max_res", "frames", "fps",
        "depth_steps", "disparity_perc", "gpu_name", "gpu_peak_memory_mib",
        "gpu_mean_util_percent", "t_preprocess_sec", "t_depthcrafter_sec", "t_warping_sec",
        "t_m2svid_sec", "t_total_sec", "run_dir", "error",
    ]
    keys = []
    for k in preferred + sorted({k for row in rows for k in row.keys()}):
        if k not in keys:
            keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_specs(clips: list[Path], max_res_values: list[int], frame_values: list[int], fps: float, args: argparse.Namespace) -> list[RunSpec]:
    specs = []
    for clip in clips:
        clip_id = safe_slug(clip)
        for max_res in max_res_values:
            for frames in frame_values:
                run_id = f"{clip_id}_r{max_res}_f{frames}_fps{str(fps).replace('.', 'p')}"
                specs.append(RunSpec(
                    clip_id=clip_id,
                    source_path=str(clip.resolve()),
                    run_id=run_id,
                    max_res=max_res,
                    frames=frames,
                    fps=fps,
                    depth_steps=args.depth_steps,
                    disparity_perc=args.disparity_perc,
                    model_config=args.model_config,
                ))
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(), help="M2SVid repo root. Default: cwd")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory of input videos")
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/m2svid_benchmark/runs"), help="Output runs directory")
    ap.add_argument("--max-res", default="512", help="Comma-separated max resolutions, e.g. 384,512,768")
    ap.add_argument("--frames", default="24", help="Comma-separated frame counts after FPS normalization")
    ap.add_argument("--fps", type=float, default=8.0, help="Target FPS")
    ap.add_argument("--depth-steps", type=int, default=25)
    ap.add_argument("--disparity-perc", type=float, default=0.05)
    ap.add_argument("--model-config", default="configs/m2svid.yaml")
    ap.add_argument("--ckpt", default="ckpts/m2svid_weights.pt")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--only-first-n-runs", type=int, default=None, help="Debug: execute only first N expanded runs")
    ap.add_argument("--patch-xformers-fallbacks", action="store_true", help="Patch DepthCrafter xformers fallback and create no-xformers M2SVid config if needed")
    ap.add_argument("--blackwell-config", default="configs/m2svid_no_xformers_blackwell.yaml", help="Config generated by --patch-xformers-fallbacks")
    ap.add_argument("--use-blackwell-config", action="store_true", help="Use --blackwell-config as model_config")
    ap.add_argument("--gpu-monitor-interval-ms", type=int, default=500)
    ap.add_argument("--cuda-alloc-conf", default="expandable_segments:True")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    if not (repo_root / "inpaint_and_refine.py").exists():
        print(f"ERROR: {repo_root} does not look like M2SVid repo root", file=sys.stderr)
        return 2
    os.chdir(repo_root)

    if args.patch_xformers_fallbacks:
        patched = patch_depthcrafter_xformers(repo_root)
        created = create_no_xformers_config(repo_root, Path("configs/m2svid.yaml"), Path(args.blackwell_config))
        print(f"xformers fallback patch: DepthCrafter patched={patched}, blackwell_config_created={created}")
        if args.use_blackwell_config:
            args.model_config = args.blackwell_config

    clips = discover_clips(args.input_dir, args.limit_clips)
    if not clips:
        print(f"ERROR: no videos found under {args.input_dir}", file=sys.stderr)
        return 2
    specs = build_specs(clips, parse_csv_list(args.max_res, int), parse_csv_list(args.frames, int), args.fps, args)
    if args.only_first_n_runs is not None:
        specs = specs[: args.only_first_n_runs]

    print(f"Repo: {repo_root}")
    print(f"Clips: {len(clips)}; expanded runs: {len(specs)}")
    print(f"Output: {args.output_dir}")
    print(f"Model config: {args.model_config}")
    if args.dry_run:
        for spec in specs:
            print(json.dumps(asdict(spec), ensure_ascii=False))
        return 0

    all_rows = []
    for idx, spec in enumerate(specs, 1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(specs)}] {spec.run_id}")
        print("=" * 100)
        run_dir = args.output_dir / spec.run_id
        row = run_one(spec, repo_root, run_dir, args)
        all_rows.append(row)
        append_results(args.output_dir.parent / "results.csv", all_rows)
        if row.get("status") != "success":
            print(f"Run failed; continuing to next run. See {run_dir / 'metrics.json'}")

    append_results(args.output_dir.parent / "results.csv", all_rows)
    print(f"\nDone. Aggregate CSV: {args.output_dir.parent / 'results.csv'}")
    successes = sum(1 for r in all_rows if r.get("status") == "success")
    print(f"Success: {successes}/{len(all_rows)}")
    return 0 if successes == len(all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
