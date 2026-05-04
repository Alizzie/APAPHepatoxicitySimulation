"""
Bioavailability vs Uptake Rate
-------------------------------
Sweeps base_uptake_pct across a range and computes:
  F_H  = exited / (exited + metabolized)   [hepatic bioavailability]
  E_H  = 1 - F_H                           [extraction ratio]

Target: F_H ≈ 0.80 for APAP (literature: ~80% oral bioavailability,
        first-pass hepatic extraction ~20%).

Also plots absorbed molecule count (in hepatocytes) at peak uptake time
vs uptake rate — linking the PK metric to the physical drug retention.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantABM import LobuleQuadrant
from config import Config

IMAGE_DIR = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Sweep settings ────────────────────────────────────────────────────────────
UPTAKE_VALUES = np.linspace(0.05, 0.35, 10)  # base_uptake_pct range
AVOGADRO = 6.022e23
MW_APAP = 151.163  # g/mol


def _molecules(mass_umol: float) -> float:
    """Convert µmol → number of molecules."""
    return mass_umol * 1e-6 * AVOGADRO


def run_sweep() -> dict:
    config = Config()
    target_concentration = config.DOSE / 5.7
    quadrant_mass = target_concentration * (config.V_PIXEL * (config.N_PIXELS**2))

    results = {}

    for uptake in UPTAKE_VALUES:
        label = f"{uptake:.4f}"
        print(f"\nUptake = {uptake:.4f}")

        q = LobuleQuadrant(
            dose=quadrant_mass,
            exchange_on=True,
            base_uptake_pct=uptake,
            base_efflux_pct=0.08,
        )

        step = 0
        stopping_threshold = quadrant_mass * 1e-2
        peak_hep_mass = 0.0
        peak_hep_molecules = 0.0

        while True:
            q.compute_flux()
            q.record(save_frame=False)

            m_hep = float(np.sum(q.mass_grid * q.hep_mask))
            if m_hep > peak_hep_mass:
                peak_hep_mass = m_hep
                peak_hep_molecules = _molecules(m_hep)

            sin_mass = float(np.sum(q.mass_grid * q.sin_mask))
            if sin_mass < stopping_threshold or step > 50_000:
                break
            step += 1

        total_out = (
            q.total_mass_exited + q.total_mass_metab + q.total_mass_lost_to_necrosis
        )
        F_H = q.total_mass_exited / total_out if total_out > 0 else 0.0
        E_H = 1.0 - F_H

        results[label] = {
            "uptake": uptake,
            "F_H": F_H,
            "E_H": E_H,
            "exited": q.total_mass_exited,
            "metabolized": q.total_mass_metab,
            "necrosis": q.total_mass_lost_to_necrosis,
            "peak_hep_mass": peak_hep_mass,
            "peak_hep_mol": peak_hep_molecules,
        }
        print(f"  F_H={F_H:.4f}  E_H={E_H:.4f}  peak_hep={peak_hep_mass:.3e} µmol")

    return results


def plot_results(results: dict) -> None:
    uptake_vals = [d["uptake"] for d in results.values()]
    F_H_vals = [d["F_H"] for d in results.values()]
    E_H_vals = [d["E_H"] for d in results.values()]
    peak_mol = [d["peak_hep_mol"] for d in results.values()]
    peak_mass = [d["peak_hep_mass"] for d in results.values()]

    # Find uptake that gives F_H closest to 0.80
    best_idx = int(np.argmin(np.abs(np.array(F_H_vals) - 0.80)))
    best_uptake = uptake_vals[best_idx]
    best_FH = F_H_vals[best_idx]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: F_H vs uptake ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(uptake_vals, F_H_vals, "o-", color="#2196F3", linewidth=1, markersize=3)
    ax1.axhline(
        0.80,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Target F_H = 0.80 (APAP literature)",
    )
    ax1.axhline(
        0.70, color="orange", linestyle=":", linewidth=1, label="Lower bound F_H = 0.70"
    )
    ax1.axvline(
        best_uptake,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Best fit uptake = {best_uptake:.4f}\n(F_H = {best_FH:.3f})",
    )
    ax1.set_xlabel("Base Uptake Rate (fraction/step)")
    ax1.set_ylabel("Hepatic Bioavailability F_H")
    ax1.set_title("Bioavailability vs Uptake Rate\n(F_H = exited / total removed)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 2: E_H vs uptake ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(uptake_vals, E_H_vals, "o-", color="#F44336", linewidth=1, markersize=3)
    ax2.axhline(
        0.20, color="green", linestyle="--", linewidth=1.5, label="Target E_H = 0.20"
    )
    ax2.axvline(
        best_uptake,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Best fit = {best_uptake:.4f}",
    )
    ax2.set_xlabel("Base Uptake Rate (fraction/step)")
    ax2.set_ylabel("Hepatic Extraction Ratio E_H")
    ax2.set_title("Extraction Ratio vs Uptake Rate\nE_H = 1 − F_H")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 3: Absorbed molecules vs uptake ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(uptake_vals, peak_mol, "s-", color="#9C27B0", linewidth=2, markersize=6)
    ax3.axvline(
        best_uptake,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Best fit uptake = {best_uptake:.4f}",
    )
    ax3.set_xlabel("Base Uptake Rate (fraction/step)")
    ax3.set_ylabel("Peak Hepatocyte Drug Molecules (#)")
    ax3.set_title("Absorption Rate vs Drug Molecules\n(peak hepatocyte load)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2e}"))
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 4: Mass breakdown stacked bar ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    exited_vals = [d["exited"] for d in results.values()]
    metab_vals = [d["metabolized"] for d in results.values()]
    necro_vals = [d["necrosis"] for d in results.values()]

    ax4.bar(
        uptake_vals,
        exited_vals,
        width=0.001,
        label="Exited (bioavailable)",
        color="#4CAF50",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.bar(
        uptake_vals,
        metab_vals,
        width=0.001,
        bottom=exited_vals,
        label="Metabolized",
        color="#FF9800",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.bar(
        uptake_vals,
        necro_vals,
        width=0.001,
        bottom=[e + m for e, m in zip(exited_vals, metab_vals)],
        label="Lost to necrosis",
        color="#F44336",
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_xlabel("Base Uptake Rate (fraction/step)")
    ax4.set_ylabel("Mass (µmol)")
    ax4.set_title(
        "Mass Fate Breakdown vs Uptake\n(exited + metabolized + necrosis = dose)"
    )
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.suptitle(
        f"Hepatic Bioavailability Sweep — APAP First Pass\n"
        f"Literature target: F_H ≈ 0.80 | Best fit uptake = {best_uptake:.4f} "
        f"(F_H = {best_FH:.3f})",
        fontsize=11,
        y=1.01,
    )

    out = os.path.join(IMAGE_DIR, "bioavailability_uptake.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.show()

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Uptake':>10} {'F_H':>8} {'E_H':>8} {'Peak mol':>14} {'Peak µmol':>12}")
    print("-" * 70)
    for d in results.values():
        print(
            f"{d['uptake']:>10.4f} {d['F_H']:>8.4f} {d['E_H']:>8.4f} "
            f"{d['peak_hep_mol']:>14.3e} {d['peak_hep_mass']:>12.4e}"
        )
    print("=" * 70)
    print(f"\nBest fit uptake for F_H = 0.80: {best_uptake:.4f} → F_H = {best_FH:.4f}")


if __name__ == "__main__":
    results = run_sweep()
    save_path = os.path.join(IMAGE_DIR, "bioavailability_uptake_data.npy")
    np.save(save_path, results)
    print(f"\nRaw results saved → {save_path}")
    plot_results(results)
