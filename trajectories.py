import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


EPS = 1e-8


@dataclass
class Issue:
    severity: str
    title: str
    evidence: str
    likely_root_cause: str
    actionable_fix: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trajectory export + SLAM front-end diagnostics for On-The-Fly-NVS metadata"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.json in a results folder",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="trajectory_output",
        help="Output directory for trajectory and diagnosis reports",
    )
    parser.add_argument(
        "--assume_top_view",
        action="store_true",
        help="Enable top-view specific checks (planarity and forward-axis stability).",
    )
    parser.add_argument(
        "--jump_ratio_thresh",
        type=float,
        default=4.0,
        help="Step is considered a jump if step > jump_ratio_thresh * median_step.",
    )
    parser.add_argument(
        "--turn_angle_thresh_deg",
        type=float,
        default=70.0,
        help="Consecutive motion turn angle above this threshold is counted as unstable turn.",
    )
    parser.add_argument(
        "--rot_step_thresh_deg",
        type=float,
        default=20.0,
        help="Consecutive pose rotation above this threshold is counted as rotation spike.",
    )
    return parser.parse_args()


def read_optional_csv(csv_path: str) -> Optional[List[Dict[str, str]]]:
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def W2C_to_C2W(Rt):
    """Convert world-to-camera matrix [4,4] (or [3,4]) to camera-to-world."""
    Rt = np.asarray(Rt, dtype=np.float64)
    R_cw = Rt[:3, :3]
    t_cw = Rt[:3, 3]
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    return R_wc, t_wc


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """Quaternion in xyzw format."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def relative_rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    dR = R_b @ R_a.T
    c = (np.trace(dR) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, EPS, None)


def trajectory_metrics(centers: np.ndarray, Rwcs: np.ndarray, args) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if centers.shape[0] < 2:
        return metrics

    steps = centers[1:] - centers[:-1]
    step_norm = np.linalg.norm(steps, axis=1)
    med_step = float(np.median(step_norm))
    mean_step = float(np.mean(step_norm))
    std_step = float(np.std(step_norm))
    p95_step = float(np.percentile(step_norm, 95))
    max_step = float(np.max(step_norm))

    jump_mask = step_norm > args.jump_ratio_thresh * max(med_step, EPS)
    stagnant_mask = step_norm < 0.15 * max(med_step, EPS)

    metrics["num_keyframes"] = int(centers.shape[0])
    metrics["num_steps"] = int(step_norm.shape[0])
    metrics["step_median"] = med_step
    metrics["step_mean"] = mean_step
    metrics["step_std"] = std_step
    metrics["step_cv"] = std_step / max(mean_step, EPS)
    metrics["step_p95"] = p95_step
    metrics["step_max"] = max_step
    metrics["jump_count"] = int(jump_mask.sum())
    metrics["jump_ratio"] = float(jump_mask.mean())
    metrics["stagnant_count"] = int(stagnant_mask.sum())
    metrics["stagnant_ratio"] = float(stagnant_mask.mean())

    if steps.shape[0] >= 2:
        dirs = unit(steps)
        dots = np.sum(dirs[1:] * dirs[:-1], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        turn_deg = np.degrees(np.arccos(dots))
        metrics["turn_deg_median"] = float(np.median(turn_deg))
        metrics["turn_deg_p95"] = float(np.percentile(turn_deg, 95))
        turn_spike_mask = turn_deg > args.turn_angle_thresh_deg
        metrics["turn_spike_count"] = int(turn_spike_mask.sum())
        metrics["turn_spike_ratio"] = float(turn_spike_mask.mean())
    else:
        metrics["turn_deg_median"] = 0.0
        metrics["turn_deg_p95"] = 0.0
        metrics["turn_spike_count"] = 0
        metrics["turn_spike_ratio"] = 0.0

    if Rwcs.shape[0] >= 2:
        rot_steps = [
            relative_rotation_angle_deg(Rwcs[i - 1], Rwcs[i]) for i in range(1, Rwcs.shape[0])
        ]
        rot_steps = np.asarray(rot_steps, dtype=np.float64)
        metrics["rot_step_deg_median"] = float(np.median(rot_steps))
        metrics["rot_step_deg_p95"] = float(np.percentile(rot_steps, 95))
        metrics["rot_step_deg_max"] = float(np.max(rot_steps))
        rot_spike_mask = rot_steps > args.rot_step_thresh_deg
        metrics["rot_spike_count"] = int(rot_spike_mask.sum())
        metrics["rot_spike_ratio"] = float(rot_spike_mask.mean())
    else:
        metrics["rot_step_deg_median"] = 0.0
        metrics["rot_step_deg_p95"] = 0.0
        metrics["rot_step_deg_max"] = 0.0
        metrics["rot_spike_count"] = 0
        metrics["rot_spike_ratio"] = 0.0

    # Planarity check: large-scale top-view path should mainly lie on one plane.
    C = centers - centers.mean(axis=0, keepdims=True)
    cov = np.cov(C.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    e0, e1, e2 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    metrics["pca_var_0"] = e0
    metrics["pca_var_1"] = e1
    metrics["pca_var_2"] = e2
    metrics["out_of_plane_ratio"] = e2 / max(e1, EPS)

    # Forward direction consistency (camera z-axis in world).
    forwards = Rwcs[:, :, 2]
    mean_fwd = unit(np.mean(forwards, axis=0, keepdims=True))[0]
    cosang = np.clip(np.sum(unit(forwards) * mean_fwd[None], axis=1), -1.0, 1.0)
    fwd_dev = np.degrees(np.arccos(cosang))
    metrics["forward_dev_deg_median"] = float(np.median(fwd_dev))
    metrics["forward_dev_deg_p95"] = float(np.percentile(fwd_dev, 95))

    return metrics


def analyze_failure_log(rows: Optional[List[Dict[str, str]]], n_keyframes: int) -> Dict[str, float]:
    if not rows:
        return {
            "failure_entries": 0,
            "failure_rate_per_keyframe": 0.0,
        }
    miniba_inliers = [float(r["miniba_inliers"]) for r in rows]
    residuals = [float(r["residual"]) for r in rows]
    return {
        "failure_entries": int(len(rows)),
        "failure_rate_per_keyframe": float(len(rows) / max(n_keyframes, 1)),
        "failure_miniba_inliers_median": float(np.median(miniba_inliers)),
        "failure_residual_median": float(np.median(residuals)),
    }


def analyze_gaussian_debug(rows: Optional[List[Dict[str, str]]]) -> Dict[str, float]:
    if not rows:
        return {
            "gauss_debug_entries": 0,
            "mvs_reliable_ratio_mean": 0.0,
            "mono_fallback_ratio_mean": 0.0,
        }

    def f(row: Dict[str, str], key: str) -> float:
        return float(row.get(key, "0") or 0.0)

    reliable_ratios = []
    fallback_ratios = []
    occlusion_keep = []
    for row in rows:
        n_valid = max(f(row, "valid_after_depth_conf"), 1.0)
        n_sample = max(f(row, "sampled_uv"), 1.0)
        n_occ = f(row, "after_occlusion")
        reliable_ratios.append(f(row, "reliable_mvs") / n_valid)
        fallback_ratios.append(f(row, "fallback_mono") / n_valid)
        occlusion_keep.append(n_occ / n_sample)

    return {
        "gauss_debug_entries": int(len(rows)),
        "mvs_reliable_ratio_mean": float(np.mean(reliable_ratios)),
        "mono_fallback_ratio_mean": float(np.mean(fallback_ratios)),
        "occlusion_keep_ratio_mean": float(np.mean(occlusion_keep)),
    }


def build_issues(metrics: Dict[str, float], assume_top_view: bool) -> List[Issue]:
    issues: List[Issue] = []

    if metrics.get("num_keyframes", 0) < 15:
        issues.append(
            Issue(
                severity="low",
                title="樣本長度偏短，診斷可信度有限",
                evidence=f"keyframes={int(metrics.get('num_keyframes', 0))}",
                likely_root_cause="可用軌跡段太短，統計量容易被局部段落主導。",
                actionable_fix="先跑更長序列（至少 100+ keyframes）再對比診斷報告。",
            )
        )

    if metrics.get("jump_ratio", 0.0) > 0.08:
        issues.append(
            Issue(
                severity="high",
                title="軌跡出現明顯跳點（pose jump）",
                evidence=(
                    f"jump_ratio={metrics.get('jump_ratio', 0.0):.3f}, "
                    f"step_max/step_median={metrics.get('step_max', 0.0) / max(metrics.get('step_median', 1e-8), 1e-8):.2f}"
                ),
                likely_root_cause="2D-3D對應包含錯配仍被PnP/MiniBA接受，或低inlier fallback門檻過寬。",
                actionable_fix=(
                    "提高 `--min_num_inliers`、收緊低inlier接受條件，並在 `initialize_incremental()` 加入"
                    "『若與軌跡先驗差異過大則拒絕該pose』的硬閥值。"
                ),
            )
        )

    if metrics.get("turn_spike_ratio", 0.0) > 0.25:
        issues.append(
            Issue(
                severity="medium",
                title="相機方向變化呈鋸齒型（zig-zag）",
                evidence=(
                    f"turn_spike_ratio={metrics.get('turn_spike_ratio', 0.0):.3f}, "
                    f"turn_p95={metrics.get('turn_deg_p95', 0.0):.1f}deg"
                ),
                likely_root_cause="相鄰幀匹配幾何約束不足，增量位姿在局部最小值間震盪。",
                actionable_fix=(
                    "在前端加入短窗平滑先驗（constant velocity/acceleration）並對大角度轉折施加懲罰；"
                    "同時提高 match 幾何一致性檢查。"
                ),
            )
        )

    if metrics.get("rot_spike_ratio", 0.0) > 0.10:
        issues.append(
            Issue(
                severity="high",
                title="相鄰幀旋轉突增",
                evidence=(
                    f"rot_spike_ratio={metrics.get('rot_spike_ratio', 0.0):.3f}, "
                    f"rot_step_max={metrics.get('rot_step_deg_max', 0.0):.1f}deg"
                ),
                likely_root_cause="PnP初值偏離 + MiniBA 局部收斂失敗，或特徵退化區域（重複紋理/低紋理）。",
                actionable_fix=(
                    "對 PnP 結果增加旋轉變化上限檢查；若超限則回退到外推pose或延後註冊該幀。"
                ),
            )
        )

    if assume_top_view and metrics.get("out_of_plane_ratio", 0.0) > 0.35:
        issues.append(
            Issue(
                severity="high",
                title="俯視場景下 out-of-plane 漂移偏大",
                evidence=f"out_of_plane_ratio={metrics.get('out_of_plane_ratio', 0.0):.3f}",
                likely_root_cause="目前SLAM前端缺少平面/高度先驗，large-scale頂視角容易累積高度漂移。",
                actionable_fix=(
                    "在前端加入軟平面約束（例如對相機中心法向分量做L2正則），"
                    "或在MiniBA加『高度變化懲罰』支援俯視路徑。"
                ),
            )
        )

    if assume_top_view and metrics.get("forward_dev_deg_p95", 0.0) > 25.0:
        issues.append(
            Issue(
                severity="medium",
                title="俯視相機光軸不穩定",
                evidence=f"forward_dev_deg_p95={metrics.get('forward_dev_deg_p95', 0.0):.1f}deg",
                likely_root_cause="位姿初始化對 pitch/roll 的幾何約束不足，導致姿態在弱紋理區抖動。",
                actionable_fix=(
                    "在初始化與優化加入俯視姿態先驗（限制 pitch/roll），"
                    "先驗可做成可開關參數避免影響一般場景。"
                ),
            )
        )

    if metrics.get("failure_rate_per_keyframe", 0.0) > 0.20:
        issues.append(
            Issue(
                severity="high",
                title="增量位姿初始化失敗率高",
                evidence=(
                    f"failure_rate_per_keyframe={metrics.get('failure_rate_per_keyframe', 0.0):.3f}, "
                    f"failure_entries={int(metrics.get('failure_entries', 0))}"
                ),
                likely_root_cause="2D-3D對應品質不穩，或 `min_num_inliers` 與資料場景尺度不匹配。",
                actionable_fix=(
                    "調整前端閥值：增加 `--num_prev_keyframes_check`、提高特徵品質過濾，"
                    "並把失敗幀標記為待重試而不是直接丟棄。"
                ),
            )
        )

    if metrics.get("mvs_reliable_ratio_mean", 0.0) < 0.35 and metrics.get("gauss_debug_entries", 0) > 0:
        issues.append(
            Issue(
                severity="medium",
                title="MVS深度可靠比例偏低",
                evidence=(
                    f"mvs_reliable_ratio_mean={metrics.get('mvs_reliable_ratio_mean', 0.0):.3f}, "
                    f"mono_fallback_ratio_mean={metrics.get('mono_fallback_ratio_mean', 0.0):.3f}"
                ),
                likely_root_cause="large-scale頂視角基線/視差不均，純MVS在遠處或低紋理區不穩。",
                actionable_fix=(
                    "提高引導深度與幾何一致性融合權重，對遠距區域採更保守的高斯初始化密度。"
                ),
            )
        )

    if not issues:
        issues.append(
            Issue(
                severity="info",
                title="未偵測到明顯前端失穩訊號",
                evidence="所有主要穩定性指標在預設閾值內。",
                likely_root_cause="N/A",
                actionable_fix="若視覺上仍有問題，建議提高資料量並開啟更多debug欄位（匹配內點率、PnP殘差曲線）。",
            )
        )

    return issues


def save_issue_reports(out_dir: str, issues: List[Issue]):
    csv_path = os.path.join(out_dir, "slam_issue_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["severity", "title", "evidence", "likely_root_cause", "actionable_fix"],
        )
        writer.writeheader()
        for it in issues:
            writer.writerow(
                {
                    "severity": it.severity,
                    "title": it.title,
                    "evidence": it.evidence,
                    "likely_root_cause": it.likely_root_cause,
                    "actionable_fix": it.actionable_fix,
                }
            )

    md_path = os.path.join(out_dir, "slam_issue_report.md")
    with open(md_path, "w") as f:
        f.write("# SLAM Front-End Issue Report\n\n")
        for idx, it in enumerate(issues, start=1):
            f.write(f"## {idx}. [{it.severity.upper()}] {it.title}\n")
            f.write(f"- Evidence: {it.evidence}\n")
            f.write(f"- Likely Root Cause: {it.likely_root_cause}\n")
            f.write(f"- Actionable Fix: {it.actionable_fix}\n\n")

    return csv_path, md_path


def save_trajectory_outputs(out_dir: str, centers: np.ndarray, Rwcs: np.ndarray):
    tum_lines = []
    for i in range(centers.shape[0]):
        qx, qy, qz, qw = rotation_matrix_to_quaternion_xyzw(Rwcs[i])
        t = centers[i]
        tum_lines.append(f"{i} {t[0]} {t[1]} {t[2]} {qx} {qy} {qz} {qw}")

    tum_path = os.path.join(out_dir, "trajectory_tum.txt")
    with open(tum_path, "w") as f:
        f.write("\n".join(tum_lines))

    plot_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            marker="o",
            markersize=2,
            linestyle="-",
            color="b",
            label="Camera Path",
        )
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], color="g", s=80, label="Start")
        ax.scatter(centers[-1, 0], centers[-1, 1], centers[-1, 2], color="r", s=80, label="End")
        ax.set_title("SLAM Camera Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plot_path = os.path.join(out_dir, "trajectory_plot.png")
        plt.savefig(plot_path, dpi=250)
        plt.close(fig)
    except Exception:
        plot_path = None
    return tum_path, plot_path


def main():
    args = parse_args()

    if not os.path.exists(args.metadata):
        print(f"Error: Could not find {args.metadata}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.metadata, "r") as f:
        data = json.load(f)

    if "keyframes" not in data:
        print("Error: metadata.json does not contain keyframes.")
        return

    keyframes = data["keyframes"]
    if len(keyframes) == 0:
        print("Error: metadata.json has 0 keyframes.")
        return

    Rwcs = []
    centers = []
    for kf in keyframes:
        Rt = kf["Rt"]
        R_wc, t_wc = W2C_to_C2W(Rt)
        Rwcs.append(R_wc)
        centers.append(t_wc)
    Rwcs = np.asarray(Rwcs, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # Optional logs (same results directory as metadata)
    result_dir = os.path.dirname(os.path.abspath(args.metadata))
    failure_rows = read_optional_csv(os.path.join(result_dir, "failure_log.csv"))
    gauss_debug_rows = read_optional_csv(
        os.path.join(result_dir, "debug", "gaussian_init_debug.csv")
    )

    metrics = trajectory_metrics(centers, Rwcs, args)
    metrics.update(analyze_failure_log(failure_rows, metrics.get("num_keyframes", 0)))
    metrics.update(analyze_gaussian_debug(gauss_debug_rows))

    issues = build_issues(metrics, assume_top_view=args.assume_top_view)

    metrics_path = os.path.join(args.out_dir, "slam_diagnostics_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    issue_csv, issue_md = save_issue_reports(args.out_dir, issues)
    tum_path, plot_path = save_trajectory_outputs(args.out_dir, centers, Rwcs)

    print(f"Found {len(keyframes)} keyframes in metadata.")
    print(f"Saved diagnostics metrics to {metrics_path}")
    print(f"Saved issue report CSV to {issue_csv}")
    print(f"Saved issue report markdown to {issue_md}")
    print(f"Saved TUM trajectory to {tum_path}")
    if plot_path is not None:
        print(f"Saved trajectory plot to {plot_path}")
    else:
        print("Skipped trajectory plot (matplotlib unavailable).")


if __name__ == "__main__":
    main()
