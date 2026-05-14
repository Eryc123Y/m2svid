#!/usr/bin/env python3
"""Create basic plots from M2SVid benchmark results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("experiments/m2svid_benchmark/results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/m2svid_benchmark/figures"))
    args = ap.parse_args()

    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        print(f"pandas unavailable: {e}. Install with `python -m pip install pandas matplotlib tabulate`.")
        return 2

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_markdown(args.out_dir / "results_table.md", index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable: {e}")
        return 0

    ok = df[df["status"] == "success"].copy()
    if ok.empty:
        print("No successful runs to plot")
        return 0

    def scatter(x: str, y: str, name: str, ylabel: str):
        if x not in ok or y not in ok:
            return
        plt.figure(figsize=(7, 4))
        for clip_id, g in ok.groupby("clip_id"):
            plt.plot(g[x], g[y], marker="o", linestyle="-", label=clip_id[:20])
        plt.xlabel(x)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if ok["clip_id"].nunique() <= 8:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_dir / name, dpi=180)
        plt.close()

    scatter("max_res", "t_total_sec", "runtime_vs_resolution.png", "Total runtime (s)")
    scatter("frames", "t_total_sec", "runtime_vs_frames.png", "Total runtime (s)")
    scatter("max_res", "gpu_peak_memory_mib", "memory_vs_resolution.png", "Peak VRAM (MiB)")
    scatter("frames", "gpu_peak_memory_mib", "memory_vs_frames.png", "Peak VRAM (MiB)")

    stage_cols = ["t_preprocess_sec", "t_depthcrafter_sec", "t_warping_sec", "t_m2svid_sec"]
    have = [c for c in stage_cols if c in ok]
    if have:
        summary = ok.groupby(["max_res", "frames"])[have].mean().reset_index()
        summary.to_csv(args.out_dir / "runtime_breakdown_mean.csv", index=False)
        # Plot first config's stage means as simple bar chart.
        row = summary.iloc[0]
        plt.figure(figsize=(7, 4))
        plt.bar([c.replace("t_", "").replace("_sec", "") for c in have], [row[c] for c in have])
        plt.ylabel("Runtime (s)")
        plt.title(f"Mean stage runtime, max_res={row['max_res']}, frames={row['frames']}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(args.out_dir / "runtime_breakdown_example.png", dpi=180)
        plt.close()

    print(f"Wrote figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
