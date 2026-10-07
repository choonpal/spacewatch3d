"""Render exported PanoVGGT PLY/JSON results without changing the source data."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_cloud(path):
    dtype = np.dtype([(name, "<f4") for name in ("x", "y", "z")]
                     + [(name, "u1") for name in ("red", "green", "blue")])
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Incomplete PLY header")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        assert header[0:2] == ["ply", "format binary_little_endian 1.0"]
        count = int(header[2].split()[-1])
        assert header[2] == f"element vertex {count}"
        assert header[3:-1] == [f"property float {k}" for k in ("x", "y", "z")] + [
            f"property uchar {k}" for k in ("red", "green", "blue")]
        data = np.fromfile(stream, dtype=dtype, count=count)
        assert len(data) == count and not stream.read(1)
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")])
    rgb = np.column_stack([data[k] for k in ("red", "green", "blue")])
    assert np.isfinite(xyz).all()
    return xyz, rgb


def display_coordinates(xyz):
    # Match the supplied Colab preview. This is not a gravity alignment.
    return xyz[:, [0, 2, 1]] * [1, 1, -1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.result_dir
    read_json = lambda name: json.loads((root / name).read_text())
    summary = read_json("summary.json")
    settings = read_json("settings.json")
    manifest = read_json("input_manifest.json")
    poses = read_json("camera_poses.json")
    diagnostics = read_json("alignment_diagnostics.json")
    timings = read_json("inference_timing.json")
    xyz, rgb = read_cloud(root / "panovggt_merged.ply")
    local_xyz, local_rgb = read_cloud(root / "first_window.ply")
    assert len(xyz) == summary["merged_points"]
    assert [p["frame"] for p in poses] == list(range(len(manifest["frames"])))
    cameras = np.array([p["camera_to_world"] for p in poses])
    assert cameras.shape == (len(poses), 4, 4) and np.isfinite(cameras).all()
    assert len(diagnostics) == summary["windows"] - 1
    assert len(timings) == summary["windows"] == len(settings["window_indices"])
    ratios = np.array([d["residual_over_median_depth"] for d in diagnostics])
    seconds = sum(t["seconds"] for t in timings.values())
    assert np.isclose(seconds, summary["inference_seconds"])
    assert np.isclose(ratios.max(), summary["max_alignment_residual_ratio"])

    selected = np.random.default_rng(7).choice(len(xyz), min(300000, len(xyz)), replace=False)
    point = display_coordinates(xyz[selected])
    color = rgb[selected] / 255.0
    local = display_coordinates(local_xyz)
    camera = display_coordinates(cameras[:, :3, 3])
    bounds = display_coordinates(xyz)
    low, high = bounds.min(0), bounds.max(0)
    del bounds
    bg, fg, muted, accent = "#101924", "#edf3fa", "#b9c7d8", "#ffbe4f"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": "#526176",
        "xtick.color": muted, "ytick.color": muted, "savefig.facecolor": bg,
        "axes.facecolor": bg, "figure.facecolor": bg,
    })

    fig = plt.figure(figsize=(15, 8))
    for index, (pts, colors, title, lower, upper) in enumerate([
        (local, local_rgb / 255.0, f"First 5 frames | {len(local):,} points", local.min(0), local.max(0)),
        (point, color, f"69 frames | {len(xyz):,} exported points", low, high),
    ]):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        ax.scatter(*pts.T, c=colors, s=.35, depthshade=False, linewidths=0)
        if index == 1:
            ax.plot(*camera.T, c=accent, lw=1.4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis.set_tick_params(labelsize=9, colors=muted)
        ax.set(xlabel="X", ylabel="Z", zlabel="-Y")
        span = np.maximum(upper - lower, 1e-3)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(span)
        ax.set_proj_type("ortho")
        ax.view_init(elev=42, azim=-66)
        ax.grid(False)
        ax.set_title(title, fontsize=13, pad=15)
    fig.text(.05, .95, "PanoVGGT | Local reconstruction and merged map", fontsize=22, weight="bold")
    fig.text(.05, .90, "R0010003 / actual exported XYZ + RGB / custom sequential Sim(3) fusion", color=muted)
    fig.text(.05, .075, f"Left: all {len(local):,} points. Right: deterministic {len(point):,}-point sample; full exported coordinate bounds.", color=muted, fontsize=10)
    fig.text(.05, .04, "Uncalibrated scale. Display axes (X, Z, -Y) are not aligned to gravity. Yellow: predicted camera path.", color=muted, fontsize=10)
    fig.subplots_adjust(left=.025, right=.94, top=.82, bottom=.16, wspace=.08)
    fig.savefig(args.output_dir / "pointcloud_overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, dims, title in zip(axes, [(0, 1), (0, 2)], ["X-Z projection", "X / -Y projection"]):
        a, b = dims
        ax.scatter(point[:, a], point[:, b], c=color, s=.32, linewidths=0)
        ax.plot(camera[:, a], camera[:, b], color=accent, lw=1.5, label="Camera path (69 poses)")
        ax.scatter(camera[0, a], camera[0, b], color="#47e2b1", s=60, edgecolors=bg, label="Start", zorder=4)
        ax.scatter(camera[-1, a], camera[-1, b], color="#ff7e9d", marker="X", s=75, edgecolors=bg, label="End", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(low[a] - .5, high[a] + .5)
        ax.set_ylim(low[b] - .5, high[b] + .5)
        ax.set_xlabel(["X", "Z", "-Y"][a] + " (relative units)")
        ax.set_ylabel(["X", "Z", "-Y"][b] + " (relative units)")
        ax.set_title(title, fontsize=15)
        ax.grid(alpha=.12)
        ax.legend(loc="best", facecolor=bg, edgecolor="#526176", fontsize=9)
    fig.text(.06, .95, "PanoVGGT | Structure and camera trajectory", fontsize=22, weight="bold")
    fig.text(.06, .90, "Two coordinate projections; no gravity alignment, scale calibration, or trajectory correction", color=muted)
    fig.text(.06, .035, "300,000 sampled points; full exported coordinate bounds. A near-return in projection is not a loop-closure accuracy test.", color=muted, fontsize=10)
    fig.subplots_adjust(left=.07, right=.96, top=.82, bottom=.13, wspace=.23)
    fig.savefig(args.output_dir / "pointcloud_projections.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    idx = np.array([d["window"] for d in diagnostics])
    axes[0].plot(idx, ratios * 100, "o-", color="#62b9ff", ms=5)
    axes[0].axhline(np.median(ratios) * 100, color=accent, ls="--", lw=1, label=f"Median {np.median(ratios) * 100:.2f}%")
    axes[0].set_ylabel("Residual / median depth (%)")
    axes[0].set_ylim(0, max(ratios * 100) * 1.2)
    axes[0].set_title("Shared-pixel agreement between adjacent windows", loc="left", fontsize=13)
    axes[1].plot(idx, [d["relative_scale"] for d in diagnostics], "o-", color="#62b9ff", label="Adjacent-window scale")
    axes[1].plot(idx, [d["cumulative_scale"] for d in diagnostics], "s-", color=accent, label="Cumulative scale to window 0")
    axes[1].axhline(1, color=muted, ls=":", lw=1)
    axes[1].set_ylabel("Sim(3) scale factor")
    axes[1].set_xlabel("Window index (0-based; window 0 anchors the map)")
    axes[1].set_xticks(np.arange(1, 23))
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(facecolor=bg, edgecolor="#526176", fontsize=10)
    fig.text(.09, .95, "PanoVGGT | Alignment diagnostics", fontsize=22, weight="bold")
    fig.text(.09, .90, "22 sequential alignments / 23 windows / final overlap uses 4 shared frames", color=muted)
    fig.text(.09, .035, "These values measure prediction consistency, not real-world accuracy. No loop closure or bundle adjustment.", color=muted, fontsize=10)
    fig.subplots_adjust(left=.1, right=.96, top=.8, bottom=.13, hspace=.38)
    fig.savefig(args.output_dir / "alignment_diagnostics.png", dpi=160)
    plt.close(fig)

    verification = {
        "first_window_points": len(local_xyz), "merged_points": len(xyz),
        "all_xyz_finite": True, "rgb_range": [int(rgb.min()), int(rgb.max())],
        "merged_xyz_min": xyz.min(0).tolist(), "merged_xyz_max": xyz.max(0).tolist(),
        "camera_poses": len(poses), "inference_seconds_sum": seconds,
        "peak_allocated_gpu_gb": max(t["peak_gpu_gb"] for t in timings.values()),
        "alignment_residual_ratio_min_median_max": [float(ratios.min()), float(np.median(ratios)), float(ratios.max())],
        "visualization": {"merged_sample_points": len(point), "seed": 7, "axis_crop": False,
                          "display_axes": ["X", "Z", "-Y"], "gravity_aligned": False},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ["panovggt_merged.ply", "first_window.ply", "summary.json", "settings.json",
                                       "camera_poses.json", "input_manifest.json", "alignment_diagnostics.json", "inference_timing.json"]},
    }
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
