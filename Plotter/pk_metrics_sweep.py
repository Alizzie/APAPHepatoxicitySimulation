"""
PK Metrics Sweep: E_H, C_hep_max, AUC_hep
-------------------------------------------
For each uptake OR efflux scenario, computes:

  E_H        — Hepatic Extraction Ratio = mass_metabolized / dose
                (first-pass only; true E_H would need C_in/C_out at steady state,
                 but for a bolus single-pass: E_H ≈ metab / dose)

  C_hep_max  — Peak mean hepatocyte concentration during the simulation (µmol/L)

  AUC_hep    — Area under the mean hepatocyte concentration–time curve (µmol/L·s)
                computed via the trapezoidal rule

Outputs
-------
  - Console summary table
  - 3-panel figure: E_H bar, C_hep_max bar, AUC_hep bar, all per scenario
  - Optional: overlay of mean C_hep(t) traces for all scenarios

Usage: set SWEEP_PARAM = "uptake" or "efflux" at the top of the script.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantABM import LobuleQuadrant
from config import Config

IMAGE_DIR = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── User setting ──────────────────────────────────────────────────────────────
SWEEP_PARAM = "uptake"  # "uptake" | "efflux"
# ─────────────────────────────────────────────────────────────────────────────


def _is_duplicate() -> bool:
    return LobuleQuadrant.__module__ == "LobuleQuadrantDuplicate"


def decide_scenarios(sweep: str) -> dict[str, float]:
    """Return label → multiplier/value for the chosen sweep parameter."""
    if sweep == "uptake":
        if _is_duplicate():
            return {
                "No Uptake": 0.0,
                "Low Uptake": 0.003,
                "Normal Uptake": 0.006,
                "High Uptake": 0.012,
            }
        return {
            "No Uptake": 0.0,
            "Low Uptake": 0.5,
            "Normal Uptake": 1.0,
            "High Uptake": 2.0,
        }
    else:  # efflux
        if _is_duplicate():
            return {
                "No Efflux": 0.0,
                "Low Efflux": 0.0005,
                "Normal Efflux": 0.002,
                "High Efflux": 0.004,
            }
        return {
            "No Efflux": 0.0,
            "Low Efflux": 0.5,
            "Normal Efflux": 1.0,
            "High Efflux": 5.0,
        }


def _build_quadrant(
    quadrant_mass: float, sweep: str, multiplier: float
) -> tuple[LobuleQuadrant, Config]:
    config = Config()
    if _is_duplicate():
        kwargs = (
            {"base_uptake_pct": multiplier}
            if sweep == "uptake"
            else {"base_efflux_pct": multiplier}
        )
        q = LobuleQuadrant(dose=quadrant_mass, exchange_on=True, **kwargs)
    else:
        if sweep == "uptake":
            config.CL_INFLUX *= multiplier
        else:
            config.CL_EFFLUX *= multiplier
        q = LobuleQuadrant(dose=quadrant_mass, exchange_on=True, config_override=config)
    return q, config


# ── Core simulation ───────────────────────────────────────────────────────────


def run_pk_sweep(sweep: str = SWEEP_PARAM) -> tuple[dict, float]:
    """
    Run one simulation per scenario; record mean C_hep at every step.
    Returns results dict and quadrant_mass.
    """
    config = Config()
    quadrant_mass = config.DOSE / 4
    scenarios = decide_scenarios(sweep)
    results: dict[str, dict] = {}

    for label, multiplier in scenarios.items():
        print(f"\n[{sweep.upper()} SWEEP] {label}  (value={multiplier})")
        q, cfg = _build_quadrant(quadrant_mass, sweep, multiplier)

        time_pts: list[float] = []
        chep_mean: list[float] = []  # mean concentration over alive hep pixels

        step = 0
        stopping_threshold = quadrant_mass * 1e-3

        while True:
            q.compute_flux()
            sin_mass = np.sum(q.mass_grid * q.sin_mask)

            # Record every step for accurate AUC
            q.record(save_frame=False)
            time_pts.append(q.current_time)

            alive_hep = q.hep_mask
            n_px = alive_hep.sum()
            chep_mean.append(float(q.mass_grid[alive_hep].mean()) if n_px > 0 else 0.0)

            if sin_mass < stopping_threshold:
                print(f"  Stopped  step={step} | sin_mass={sin_mass:.4e}")
                break
            if step > 50_000:
                print(f"  Max steps step={step}")
                break
            step += 1

        # ── PK metrics ────────────────────────────────────────────────────────
        # E_H: fraction of dose metabolised in this single pass
        E_H = q.total_mass_metab / quadrant_mass if quadrant_mass > 0 else 0.0

        # C_hep_max: peak mean hepatocyte concentration
        C_hep_max = float(np.max(chep_mean)) if chep_mean else 0.0

        # AUC_hep: trapezoidal integration of mean C_hep over time
        AUC_hep = float(np.trapezoid(chep_mean, time_pts)) if len(time_pts) > 1 else 0.0

        results[label] = {
            "time": time_pts,
            "chep_mean": chep_mean,
            "exited": q.exited_mass_history,
            "E_H": E_H,
            "C_hep_max": C_hep_max,
            "AUC_hep": AUC_hep,
            "metab": q.total_mass_metab,
        }

        print(
            f"  E_H={E_H:.4f} | C_hep_max={C_hep_max:.4e} µM | "
            f"AUC_hep={AUC_hep:.4e} µM·s"
        )

    return results, quadrant_mass


# ── Plotting ──────────────────────────────────────────────────────────────────


def print_summary_table(results: dict, quadrant_mass: float) -> None:
    print("\n" + "=" * 72)
    print(
        f"{'Scenario':<22} {'E_H':>8} {'F_H':>8} {'C_hep_max (µM)':>16} {'AUC_hep (µM·s)':>16}"
    )
    print("-" * 72)
    for label, d in results.items():
        print(
            f"{label:<22} {d['E_H']:>8.4f} {1-d['E_H']:>8.4f} "
            f"{d['C_hep_max']:>16.4e} {d['AUC_hep']:>16.4e}"
        )
    print("=" * 72)
    print("E_H = hepatic extraction ratio (first-pass, bolus approximation)")
    print("F_H = hepatic bioavailability  = 1 - E_H")
    print("C_hep_max = peak mean hepatocyte concentration")
    print("AUC_hep   = area under hepatocyte conc-time curve (trapz)")
    print()


def plot_pk_metrics(results: dict, quadrant_mass: float, sweep: str) -> None:
    labels = list(results.keys())
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#F44336"][: len(labels)]

    E_H_vals = [results[l]["E_H"] for l in labels]
    F_H_vals = [1 - results[l]["E_H"] for l in labels]
    Cmax_vals = [results[l]["C_hep_max"] for l in labels]
    AUC_vals = [results[l]["AUC_hep"] for l in labels]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    def _bar(ax, vals, ylabel, title, ref_line=None):
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.3f}" if v < 10 else f"{v:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if ref_line is not None:
            ax.axhline(
                ref_line,
                color="black",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label=f"Ref = {ref_line}",
            )
            ax.legend(fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.tick_params(axis="x", labelsize=8, rotation=15)

    # Panel 1 — E_H
    _bar(
        fig.add_subplot(gs[0, 0]),
        E_H_vals,
        "E_H  (–)",
        "Hepatic Extraction Ratio\n(first-pass bolus approx.)",
        ref_line=0.3,
    )  # APAP literature upper bound

    # Panel 2 — F_H
    _bar(
        fig.add_subplot(gs[0, 1]),
        F_H_vals,
        "F_H  (–)",
        "Hepatic Bioavailability\nF_H = 1 – E_H",
        ref_line=0.7,
    )

    # Panel 3 — C_hep_max
    _bar(
        fig.add_subplot(gs[0, 2]),
        Cmax_vals,
        "C_hep_max  (µM)",
        "Peak Mean Hepatocyte\nConcentration",
    )

    # Panel 4 — AUC_hep
    _bar(
        fig.add_subplot(gs[1, 0]),
        AUC_vals,
        "AUC_hep  (µM·s)",
        "Hepatocyte AUC\n(trapz over simulation)",
    )

    # Panel 5 — Mean C_hep(t) traces
    ax5 = fig.add_subplot(gs[1, 1])
    for (label, d), col in zip(results.items(), colors):
        ax5.plot(d["time"], d["chep_mean"], color=col, label=label, linewidth=1.5)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Mean C_hep (µM)")
    ax5.set_title("Hepatocyte Concentration–Time\nCurves (all scenarios)")
    ax5.legend(fontsize=7)
    ax5.grid(True, linestyle=":", alpha=0.6)

    # Panel 6 — E_H vs AUC scatter (uptake/efflux tradeoff)
    ax6 = fig.add_subplot(gs[1, 2])
    for (label, d), col in zip(results.items(), colors):
        ax6.scatter(
            d["E_H"],
            d["AUC_hep"],
            color=col,
            s=80,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=label,
        )
        ax6.annotate(
            label,
            (d["E_H"], d["AUC_hep"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    ax6.set_xlabel("E_H  (extraction ratio)")
    ax6.set_ylabel("AUC_hep  (µM·s)")
    ax6.set_title("E_H vs AUC_hep\n(extraction–exposure tradeoff)")
    ax6.grid(True, linestyle=":", alpha=0.6)

    sweep_label = sweep.capitalize()
    plt.suptitle(
        f"PK Metrics Sweep — {sweep_label} Rate\n"
        f"(E_H, F_H, C_hep_max, AUC_hep | first-pass bolus, single quadrant)",
        fontsize=11,
        y=1.01,
    )

    out_path = os.path.join(IMAGE_DIR, f"pk_metrics_{sweep}_sweep.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, quadrant_mass = run_pk_sweep(sweep=SWEEP_PARAM)
    print_summary_table(results, quadrant_mass)
    plot_pk_metrics(results, quadrant_mass, sweep=SWEEP_PARAM)
