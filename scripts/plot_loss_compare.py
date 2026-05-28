import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_loss_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "total": float(row["total"]),
                    "l1": float(row["l1"]),
                    "ssim": float(row["ssim"]),
                    "depth": float(row["depth"]),
                }
            )
    return rows


def moving_average(values, window):
    if window <= 1 or len(values) == 0:
        return values
    out = []
    acc = 0.0
    for idx, value in enumerate(values):
        acc += value
        if idx >= window:
            acc -= values[idx - window]
            out.append(acc / window)
        else:
            out.append(acc / (idx + 1))
    return out


def plot_compare(base_rows, merge_rows, out_path, smooth_window=20):
    b_step = [r["step"] for r in base_rows]
    m_step = [r["step"] for r in merge_rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for key, color in [("total", "tab:blue")]:
        axes[0].plot(
            b_step,
            moving_average([r[key] for r in base_rows], smooth_window),
            color=color,
            linewidth=1.4,
            label=f"baseline-{key}",
        )
        axes[0].plot(
            m_step,
            moving_average([r[key] for r in merge_rows], smooth_window),
            color="tab:orange",
            linewidth=1.4,
            label=f"merge-{key}",
        )

    axes[0].set_ylabel("Total Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    for key, color in [("l1", "tab:green"), ("ssim", "tab:red"), ("depth", "tab:purple")]:
        axes[1].plot(
            b_step,
            moving_average([r[key] for r in base_rows], smooth_window),
            color=color,
            linewidth=1.0,
            linestyle="-",
            label=f"baseline-{key}",
        )
        axes[1].plot(
            m_step,
            moving_average([r[key] for r in merge_rows], smooth_window),
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=f"merge-{key}",
        )

    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Component Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", ncols=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot baseline vs merge loss comparison")
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--merge_csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    base_rows = read_loss_csv(args.baseline_csv)
    merge_rows = read_loss_csv(args.merge_csv)
    plot_compare(base_rows, merge_rows, args.out, smooth_window=args.smooth)
    print(f"Saved comparison plot to {args.out}")
