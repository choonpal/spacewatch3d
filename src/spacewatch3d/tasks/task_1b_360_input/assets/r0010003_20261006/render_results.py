"""Render README figures from the exported ASCII PLY and TUM trajectory."""

import argparse
import io
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.result_dir / "dense.ply").open() as stream:
        header = []
        for line in stream:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        assert "format ascii 1.0" in header, "Expected exported ASCII PLY."
        vertices = int(next(line.split()[-1] for line in header if line.startswith("element vertex ")))
        data = np.loadtxt(stream)
    assert data.shape == (vertices, 6) and np.isfinite(data).all()
    xyz, rgb = data[:, :3], data[:, 3:] / 255.0
    assert np.all((rgb >= 0) & (rgb <= 1))

    trajectory_path = args.result_dir / "trajectory/frame_trajectory.txt"
    if trajectory_path.is_file():
        trajectory = np.loadtxt(trajectory_path)
    else:
        archives = sorted(args.result_dir.glob("R0010003_*.zip"))
        if len(archives) != 1:
            raise ValueError("Expected one result ZIP or an extracted trajectory directory.")
        with zipfile.ZipFile(archives[0]) as archive:
            trajectory = np.loadtxt(io.BytesIO(archive.read("trajectory/frame_trajectory.txt")))
    assert trajectory.ndim == 2 and trajectory.shape[1] == 8
    assert np.isfinite(trajectory).all()
    camera = trajectory[:, 1:4]

    # Display the same X, Z, -Y coordinates as the exported Plotly preview.
    plot_xyz = xyz[:, [0, 2, 1]] * [1, 1, -1]
    plot_camera = camera[:, [0, 2, 1]] * [1, 1, -1]
    lower, upper = plot_xyz.min(axis=0), plot_xyz.max(axis=0)
    span = upper - lower
    background, foreground, accent = "#101924", "#edf3fa", "#ffbe4f"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": foreground, "axes.labelcolor": foreground,
        "xtick.color": "#b9c7d8", "ytick.color": "#b9c7d8",
        "axes.edgecolor": "#526176", "savefig.facecolor": background,
    })

    fig = plt.figure(figsize=(12, 8), facecolor=background)
    ax = fig.add_subplot(111, projection="3d", facecolor=background)
    ax.scatter(*plot_xyz.T, c=rgb, s=0.65, depthshade=False, linewidths=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.06, 0.10, 0.15, 0.0))
        axis.set_tick_params(labelsize=9, colors="#b9c7d8")
    ax.set(xlabel="X", ylabel="Z", zlabel="-Y (up)")
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(span, zoom=1.18)
    ax.set_proj_type("ortho")
    ax.view_init(elev=55, azim=-65)
    ax.grid(False)
    fig.text(0.07, 0.93, "R0010003 | 3D point cloud", fontsize=22, weight="bold")
    fig.text(0.07, 0.89, f"{vertices:,} RGB points · stella_vslam_dense", fontsize=13, color="#b9c7d8")
    fig.text(0.07, 0.065, "All exported points shown. No filtering or surface reconstruction.", fontsize=11, color="#b9c7d8")
    fig.text(0.07, 0.035, "Axes use uncalibrated monocular units, not meters.", fontsize=11, color="#b9c7d8")
    fig.subplots_adjust(left=0.04, right=0.94, bottom=0.1, top=0.9)
    fig.savefig(args.output_dir / "pointcloud_overview.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10), facecolor=background)
    ax.set_facecolor(background)
    # Draw higher points last, approximating a view from above; retain all points.
    order = np.argsort(plot_xyz[:, 2])
    ax.scatter(plot_xyz[order, 0], plot_xyz[order, 1], c=rgb[order], s=0.8, linewidths=0)
    ax.plot(plot_camera[:, 0], plot_camera[:, 1], color=accent, linewidth=1.8,
            label=f"Camera trajectory ({len(camera)} poses)", zorder=3)
    ax.scatter(*plot_camera[0, :2], c="#47e2b1", marker="o", s=80,
               edgecolors=background, linewidths=1.2, label="Start", zorder=4)
    ax.scatter(*plot_camera[-1, :2], c="#ff7e9d", marker="X", s=90,
               edgecolors=background, linewidths=1, label="End", zorder=5)
    ax.set_aspect("equal")
    ax.set(xlabel="X (uncalibrated units)", ylabel="Z (uncalibrated units)")
    ax.set_xlim(lower[0] - 2, upper[0] + 2)
    ax.set_ylim(lower[1] - 2, upper[1] + 2)
    ax.grid(alpha=0.12)
    legend = ax.legend(loc="lower right", facecolor=background, edgecolor="#526176", fontsize=10)
    for text in legend.get_texts():
        text.set_color(foreground)
    fig.text(0.1, 0.955, "R0010003 | Top view & camera trajectory", fontsize=20, weight="bold")
    fig.text(0.1, 0.917, f"X-Z projection · all {vertices:,} points · trajectory from exported TUM file", fontsize=11, color="#b9c7d8")
    fig.text(0.1, 0.025, "No metric scale calibration. Trajectory is overlaid for visibility.", fontsize=11, color="#b9c7d8")
    fig.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.095)
    fig.savefig(args.output_dir / "pointcloud_top_trajectory.png", dpi=160)
    plt.close(fig)
    print(f"Rendered {vertices:,} points and {len(camera)} camera poses into {args.output_dir}")


if __name__ == "__main__":
    main()
