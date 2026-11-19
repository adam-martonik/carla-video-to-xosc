#!/usr/bin/env python
"""Phase 5: Quantitative before/after evaluation & visualization across all demo videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import video_to_xosc as v2x
from xosc_pipeline.config import load_config
from xosc_pipeline.metrics import evaluate_xosc_file


EVAL_DIR = ROOT / "reports" / "evaluation"
XSD_PATH = ROOT / "scenario_runner" / "srunner" / "openscenario" / "OpenSCENARIO.xsd"

VIDEO_TO_BASELINE_XOSC = {
    "dashcam_input": "dashcam_reconstructed.xosc",
    "brake": "scenar_brake.xosc",
    "jelen": "scenar_jelen.xosc",
    "stiahnute_video": "moj_scenar.xosc",
}


def run_all_after() -> dict:
    cfg = load_config()
    baseline_data = json.loads((EVAL_DIR / "baseline_audit.json").read_text(encoding="utf-8"))
    base_xosc_map = {item["file"]: item for item in baseline_data["existing_xosc"]}
    base_vid_map = {Path(item["video"]).stem: item for item in baseline_data["baseline_video_runs"]}

    results = {}
    projected_map = {}

    for stem, base_xosc_name in VIDEO_TO_BASELINE_XOSC.items():
        vid_path = ROOT / "demo" / f"{stem}.mp4"
        if not vid_path.exists():
            continue

        out_xosc = EVAL_DIR / f"after_{stem}.xosc"
        out_xodr = EVAL_DIR / f"after_{stem}.xodr"
        out_json = EVAL_DIR / f"after_{stem}.json"

        projected = v2x.run_pipeline(
            video_path=vid_path,
            output_xosc=out_xosc,
            output_json=out_json,
            output_xodr=out_xodr,
            xsd_path=XSD_PATH if XSD_PATH.exists() else None,
            town="Town04_Opt",
            offline=True,
            cfg=cfg,
            save_annotated=True,
        )
        projected_map[stem] = projected
        after_metrics = evaluate_xosc_file(out_xosc, xsd_path=XSD_PATH if XSD_PATH.exists() else None)

        base_xosc = base_xosc_map.get(base_xosc_name, {})
        base_vid = base_vid_map.get(stem, {})

        # Compute worst baseline metrics from existing .xosc
        bx_trajs = base_xosc.get("trajectories", {})
        before_max_acc = max((m["max_abs_acc"] for m in bx_trajs.values()), default=0.0)
        before_max_jerk = max((m["max_abs_jerk"] for m in bx_trajs.values()), default=0.0)
        before_max_yaw = max((m["max_yaw_rate"] for m in bx_trajs.values()), default=0.0)
        before_overlaps = base_xosc.get("overlap_frames_lt_3_5m", 0)

        results[stem] = {
            "video": vid_path.name,
            "baseline_xosc": base_xosc_name,
            "before": {
                "xsd_valid": base_xosc.get("xsd_valid"),
                "param_count": base_xosc.get("param_count", 0),
                "max_perf_acc_xml": 200.0,
                "worst_max_abs_acc": round(before_max_acc, 3),
                "worst_max_abs_jerk": round(before_max_jerk, 3),
                "worst_max_yaw_rate": round(before_max_yaw, 4),
                "overlap_steps": before_overlaps,
                "raw_ego_max_abs_acc": base_vid.get("raw_ego_max_abs_acc"),
                "has_xodr": False,
            },
            "after": {
                **after_metrics,
                "has_xodr": out_xodr.exists(),
                "road_geometry": projected.get("road_geometry", {}),
            },
        }

    (EVAL_DIR / "before_after_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown_table(results)
    _plot_kinematics_comparison(results)
    _plot_bev_trajectories(projected_map)
    _plot_velocity_profiles(projected_map)
    _plot_video_overlays(list(VIDEO_TO_BASELINE_XOSC.keys()))
    return results


def _write_markdown_table(results: dict) -> None:
    lines = [
        "# Pred / Po kvantitatívne vyhodnotenie (`reports/evaluation/before_after_table.md`)",
        "",
        "| Video | Max |a| Pred → Po (m/s²) | Max |jerk| Pred → Po (m/s³) | Max |yaw_rate| Pred → Po (rad/s) | OBB kolízie Pred → Po | Parametre Pred → Po | XODR | XSD Valid |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for stem, r in results.items():
        b, a = r["before"], r["after"]
        lines.append(
            f"| `{r['video']}` | {b['worst_max_abs_acc']:.2f} → **{a['worst_max_abs_acc']:.2f}** "
            f"| {b['worst_max_abs_jerk']:.2f} → **{a['worst_max_abs_jerk']:.2f}** "
            f"| {b['worst_max_yaw_rate']:.3f} → **{a['worst_max_yaw_rate']:.3f}** "
            f"| {b['overlap_steps']} → **{a['obb_collision_steps']}** "
            f"| {b['param_count']} → **{a['param_count']}** "
            f"| {'Áno' if a['has_xodr'] else 'Nie'} | {'PASS' if a['xsd_valid'] else 'FAIL'} |"
        )
    (EVAL_DIR / "before_after_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_kinematics_comparison(results: dict) -> None:
    stems = list(results.keys())
    labels = [results[s]["video"].replace(".mp4", "") for s in stems]
    x = np.arange(len(stems))
    w = 0.36

    b_acc = [results[s]["before"]["worst_max_abs_acc"] for s in stems]
    a_acc = [results[s]["after"]["worst_max_abs_acc"] for s in stems]
    b_jrk = [results[s]["before"]["worst_max_abs_jerk"] for s in stems]
    a_jrk = [results[s]["after"]["worst_max_abs_jerk"] for s in stems]
    b_yaw = [results[s]["before"]["worst_max_yaw_rate"] for s in stems]
    a_yaw = [results[s]["after"]["worst_max_yaw_rate"] for s in stems]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Panel 1: Max Acceleration
    axes[0].bar(x - w / 2, b_acc, w, label="Pred (baseline)", color="#d9534f", alpha=0.88)
    axes[0].bar(x + w / 2, a_acc, w, label="Po (nový pipeline)", color="#2b7bba", alpha=0.92)
    axes[0].axhline(9.5, color="#555", linestyle="--", linewidth=1.0, label="Fyzikálny limit (9.5 m/s²)")
    axes[0].set_title("Max. |zrýchlenie| (m/s²)", fontsize=10.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[0].legend(fontsize=8)

    # Panel 2: Max Jerk
    axes[1].bar(x - w / 2, b_jrk, w, label="Pred (baseline)", color="#d9534f", alpha=0.88)
    axes[1].bar(x + w / 2, a_jrk, w, label="Po (nový pipeline)", color="#2b7bba", alpha=0.92)
    axes[1].axhline(8.0, color="#555", linestyle="--", linewidth=1.0, label="Limit jerku (8.0 m/s³)")
    axes[1].set_title("Max. |jerk| (m/s³)", fontsize=10.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[1].legend(fontsize=8)

    # Panel 3: Max Yaw Rate
    axes[2].bar(x - w / 2, b_yaw, w, label="Pred (baseline)", color="#d9534f", alpha=0.88)
    axes[2].bar(x + w / 2, a_yaw, w, label="Po (nový pipeline)", color="#2b7bba", alpha=0.92)
    axes[2].axhline(0.6, color="#555", linestyle="--", linewidth=1.0, label="Realistický limit (0.6 rad/s)")
    axes[2].set_title("Max. |uhlová rýchlosť yaw| (rad/s)", fontsize=10.5, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15, fontsize=9)
    axes[2].legend(fontsize=8)

    fig.suptitle("Porovnanie kinematickej vierohodnosti pred a po úpravách", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "fig1_kinematics_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _plot_bev_trajectories(projected_map: dict) -> None:
    stems = list(projected_map.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=180)
    axes = axes.ravel()
    colors = ["#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    for ax, stem in zip(axes, stems):
        proj = projected_map[stem]
        ego = proj["ego"]
        ax.plot(ego["s"], ego["d"], color="#1f77b4", linewidth=2.4, label="Ego (hero)")
        ax.scatter(ego["s"][0], ego["d"][0], color="#1f77b4", s=40, zorder=5)

        for idx, act in enumerate(proj["actors"]):
            c = colors[idx % len(colors)]
            cat = act.get("category", "car")
            ax.plot(act["s"], act["d"], color=c, linewidth=1.9, linestyle="-", label=f"{act['role_name']} ({cat})")
            ax.scatter(act["s"][0], act["d"][0], color=c, s=35, zorder=5)

        # Draw lane lines every 3.5m
        for lane_y in [-5.25, -1.75, 1.75, 5.25]:
            ax.axhline(lane_y, color="#999999", linestyle=":", linewidth=0.9)

        ax.set_title(f"BEV Frenet trajektórie: {stem}.mp4", fontsize=10, fontweight="bold")
        ax.set_xlabel("Pozdĺžna súradnica s (m)", fontsize=8.5)
        ax.set_ylabel("Priečna súradnica d (m)", fontsize=8.5)
        ax.invert_yaxis()  # d > 0 to the right (downwards in top-down view)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(True, alpha=0.35)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "fig2_bev_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def _plot_velocity_profiles(projected_map: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), dpi=180)
    target_stems = ["dashcam_input", "brake"]
    for row, stem in enumerate(target_stems):
        if stem not in projected_map:
            continue
        proj = projected_map[stem]
        t = np.asarray(proj["times"], dtype=float)
        ego_v = np.asarray(proj["ego"]["v"], dtype=float)
        ego_a = np.gradient(ego_v, t)

        ax_v = axes[row, 0]
        ax_a = axes[row, 1]
        ax_v.plot(t, ego_v, label="Ego (hero)", linewidth=2.2, color="#1f77b4")
        ax_a.plot(t, ego_a, label="Ego (hero)", linewidth=2.0, color="#1f77b4")

        for act in proj["actors"][:3]:
            va = np.asarray(act["v"], dtype=float)
            aa = np.gradient(va, t)
            ax_v.plot(t, va, label=f"{act['role_name']} ({act.get('category', 'car')})", linewidth=1.6)
            ax_a.plot(t, aa, label=act["role_name"], linewidth=1.4, alpha=0.85)

        ax_v.set_title(f"{stem}.mp4 — Priebeh rýchlosti v(t)", fontsize=10, fontweight="bold")
        ax_v.set_xlabel("Čas t (s)", fontsize=8.5)
        ax_v.set_ylabel("Rýchlosť (m/s)", fontsize=8.5)
        ax_v.legend(fontsize=7.5)
        ax_v.grid(True, alpha=0.35)

        ax_a.axhline(4.5, color="#888", linestyle="--", linewidth=0.9)
        ax_a.axhline(-9.5, color="#888", linestyle="--", linewidth=0.9)
        ax_a.set_title(f"{stem}.mp4 — Priebeh zrýchlenia a(t)", fontsize=10, fontweight="bold")
        ax_a.set_xlabel("Čas t (s)", fontsize=8.5)
        ax_a.set_ylabel("Zrýchlenie (m/s²)", fontsize=8.5)
        ax_a.legend(fontsize=7.5)
        ax_a.grid(True, alpha=0.35)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "fig3_velocity_accel_profiles.png", bbox_inches="tight")
    plt.close(fig)


def _plot_video_overlays(stems: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), dpi=160)
    axes = axes.ravel()
    for ax, stem in zip(axes, stems):
        ann_vid = ROOT / "demo" / f"{stem}_tracked_annotated.mp4"
        if not ann_vid.exists():
            ax.axis("off")
            continue
        cap = cv2.VideoCapture(str(ann_vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.4))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(f"Anotovaný záznam: {stem}.mp4 (BoT-SORT + horizont + 3D)", fontsize=9.5, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "fig4_video_overlay.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summary = run_all_after()
    print(json.dumps(summary, indent=2))
