"""
Dose vs Toxicity
-----------------
Sweeps input dose from therapeutic (1g APAP) to overdose (10g+) and tracks:
  - Dead cell fraction per zone at end of simulation
  - Peak toxicity_field value per zone
  - Total mass metabolized + lost to necrosis
  - Zone 3 death fraction as the primary hepatotoxicity readout

APAP reference doses (MW = 151.163 g/mol):
  Therapeutic : 1g   →  6,614 µmol
  Max daily   : 4g   → 26,456 µmol  (your Config.DOSE)
  Toxic       : 7.5g → 49,605 µmol
  Overdose    : 10g  → 66,141 µmol

The dose at which zone 3 dead fraction crosses a threshold
is your in-silico NOAEL proxy.
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

MW_APAP = 151.163  # g/mol
NOAEL_FRAC = 0.10  # zone 3 dead fraction threshold for NOAEL proxy

# Doses in µmol (quadrant = whole dose / 4)
DOSE_GRAMS = [0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 10.0, 12.0]
DOSE_UMOL = [g / MW_APAP * 1e6 for g in DOSE_GRAMS]  # whole liver µmol


def run_dose_sweep():
    config = Config()
    results = {}

    for g, dose_total in zip(DOSE_GRAMS, DOSE_UMOL):
        quadrant_mass = dose_total / 4
        print(
            f"\nDose = {g:.1f}g ({dose_total:.2e} µmol total | {quadrant_mass:.2e} µmol/quadrant)"
        )

        q = LobuleQuadrant(
            dose=quadrant_mass,
            exchange_on=True,
        )

        step = 0
        stopping_threshold = quadrant_mass * 1e-2

        # Track peak toxicity per zone over time
        peak_tox = {1: 0.0, 2: 0.0, 3: 0.0}

        while True:
            q.compute_flux()
            q.record(save_frame=False)

            for z in (1, 2, 3):
                mask = (q.zonation == z) & q.hep_mask
                if mask.any():
                    zt = float(q.toxicity_field[mask].mean())
                    if zt > peak_tox[z]:
                        peak_tox[z] = zt

            sin_mass = float(np.sum(q.mass_grid * q.sin_mask))
            if sin_mass < stopping_threshold or step > 50_000:
                break
            step += 1

        # Dead cell fractions per zone
        dead_frac = {}
        for z in (1, 2, 3):
            total_z = int(((q.zonation == z) & (q.hep_mask | q.is_cell_dead)).sum())
            dead_z = int(((q.zonation == z) & q.is_cell_dead).sum())
            dead_frac[z] = dead_z / total_z if total_z > 0 else 0.0

        total_out = (
            q.total_mass_metab + q.total_mass_lost_to_necrosis + q.total_mass_exited
        )
        E_H = (
            (q.total_mass_metab + q.total_mass_lost_to_necrosis) / total_out
            if total_out > 0
            else 0
        )

        results[g] = {
            "dose_g": g,
            "dose_umol": dose_total,
            "dead_frac": dead_frac,
            "peak_tox": peak_tox,
            "E_H": E_H,
            "metab": q.total_mass_metab,
            "necrosis": q.total_mass_lost_to_necrosis,
            "exited": q.total_mass_exited,
        }

        print(
            f"  Dead Z1={dead_frac[1]:.3f} Z2={dead_frac[2]:.3f} Z3={dead_frac[3]:.3f} | "
            f"E_H={E_H:.3f}"
        )

    return results


def plot_results(results: dict):
    doses = [d["dose_g"] for d in results.values()]
    dead_z1 = [d["dead_frac"][1] for d in results.values()]
    dead_z2 = [d["dead_frac"][2] for d in results.values()]
    dead_z3 = [d["dead_frac"][3] for d in results.values()]
    peak_t1 = [d["peak_tox"][1] for d in results.values()]
    peak_t2 = [d["peak_tox"][2] for d in results.values()]
    peak_t3 = [d["peak_tox"][3] for d in results.values()]
    E_H_vals = [d["E_H"] for d in results.values()]
    metab_v = [d["metab"] for d in results.values()]
    necro_v = [d["necrosis"] for d in results.values()]
    exited_v = [d["exited"] for d in results.values()]

    # NOAEL proxy: lowest dose where zone 3 dead fraction > NOAEL_FRAC
    noael_dose = None
    for d, z3 in zip(doses, dead_z3):
        if z3 >= NOAEL_FRAC:
            noael_dose = d
            break

    zone_colors = {1: "#2196F3", 2: "#FF9800", 3: "#F44336"}

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Dead cell fraction per zone vs dose ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        doses,
        dead_z1,
        "o-",
        color=zone_colors[1],
        linewidth=2,
        markersize=7,
        label="Zone 1 (periportal)",
    )
    ax1.plot(
        doses,
        dead_z2,
        "s-",
        color=zone_colors[2],
        linewidth=2,
        markersize=7,
        label="Zone 2 (midzonal)",
    )
    ax1.plot(
        doses,
        dead_z3,
        "^-",
        color=zone_colors[3],
        linewidth=2,
        markersize=7,
        label="Zone 3 (pericentral)",
    )
    ax1.axhline(
        NOAEL_FRAC,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"NOAEL proxy ({NOAEL_FRAC*100:.0f}% dead)",
    )
    if noael_dose:
        ax1.axvline(
            noael_dose,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"In silico NOAEL ≈ {noael_dose:.1f}g",
        )
    # Reference dose lines
    for ref_g, ref_label, ref_col in [
        (1.0, "1g therapeutic", "green"),
        (4.0, "4g max daily", "orange"),
        (7.5, "7.5g toxic", "red"),
    ]:
        if ref_g in doses:
            ax1.axvline(ref_g, color=ref_col, linestyle="--", alpha=0.4, linewidth=1)
            ax1.text(
                ref_g + 0.05,
                0.02,
                ref_label,
                color=ref_col,
                fontsize=7,
                rotation=90,
                va="bottom",
            )
    ax1.set_xlabel("APAP Dose (g)")
    ax1.set_ylabel("Dead Hepatocyte Fraction")
    ax1.set_title("Dose vs Dead Cell Fraction\nper Hepatic Zone")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 2: Peak toxicity field per zone ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        doses,
        peak_t1,
        "o-",
        color=zone_colors[1],
        linewidth=2,
        markersize=7,
        label="Zone 1",
    )
    ax2.plot(
        doses,
        peak_t2,
        "s-",
        color=zone_colors[2],
        linewidth=2,
        markersize=7,
        label="Zone 2",
    )
    ax2.plot(
        doses,
        peak_t3,
        "^-",
        color=zone_colors[3],
        linewidth=2,
        markersize=7,
        label="Zone 3",
    )
    ax2.set_xlabel("APAP Dose (g)")
    ax2.set_ylabel("Peak Mean Toxicity Field (µM·steps)")
    ax2.set_title("Dose vs Peak Toxicity Field\nper Zone (cumulative exposure)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 3: E_H vs dose ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(doses, E_H_vals, "D-", color="#607D8B", linewidth=2, markersize=7)
    ax3.axhline(
        0.20, color="green", linestyle="--", linewidth=1.5, label="Target E_H = 0.20"
    )
    ax3.set_xlabel("APAP Dose (g)")
    ax3.set_ylabel("Hepatic Extraction Ratio E_H")
    ax3.set_title("Dose vs Extraction Ratio\n(nonlinearity = saturation / necrosis)")
    ax3.set_ylim(0, 1.0)
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 4: Mass fate stacked area ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.stackplot(
        doses,
        exited_v,
        metab_v,
        necro_v,
        labels=["Exited (bioavailable)", "Metabolized", "Lost to necrosis"],
        colors=["#4CAF50", "#FF9800", "#F44336"],
        alpha=0.85,
    )
    ax4.set_xlabel("APAP Dose (g)")
    ax4.set_ylabel("Mass (µmol)")
    ax4.set_title("Mass Fate vs Dose\n(stacked: exited + metab + necrosis = dose)")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 5: Zone 3 dead fraction — toxicity dose-response curve ──────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.fill_between(doses, dead_z3, alpha=0.25, color="#F44336")
    ax5.plot(doses, dead_z3, "^-", color="#F44336", linewidth=2.5, markersize=8)
    ax5.axhline(
        NOAEL_FRAC,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"NOAEL threshold ({NOAEL_FRAC*100:.0f}%)",
    )
    if noael_dose:
        ax5.axvline(
            noael_dose,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            label=f"In silico NOAEL ≈ {noael_dose:.1f}g",
        )
    ax5.set_xlabel("APAP Dose (g)")
    ax5.set_ylabel("Zone 3 Dead Cell Fraction")
    ax5.set_title("Dose–Toxicity Curve\n(Zone 3 pericentral necrosis)")
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 6: Necrosis / metabolized ratio ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ratio = [n / (m + 1e-12) for n, m in zip(necro_v, metab_v)]
    ax6.bar(
        doses,
        ratio,
        color=plt.colormaps["Reds"](np.linspace(0.3, 0.9, len(doses))),
        edgecolor="black",
        linewidth=0.6,
        width=0.4,
    )
    ax6.set_xlabel("APAP Dose (g)")
    ax6.set_ylabel("Necrosis / Metabolism Ratio")
    ax6.set_title(
        "Necrotic vs Metabolic Loss Ratio\n(rising = toxicity dominates clearance)"
    )
    ax6.grid(True, axis="y", linestyle=":", alpha=0.6)

    noael_str = f"{noael_dose:.1f}g" if noael_dose else "not reached"
    plt.suptitle(
        f"Dose–Toxicity Analysis — APAP ABM\n"
        f"In silico NOAEL proxy (zone 3 >{NOAEL_FRAC*100:.0f}% dead): {noael_str}",
        fontsize=11,
        y=1.01,
    )

    out = os.path.join(IMAGE_DIR, "dose_toxicity.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == "__main__":
    results = run_dose_sweep()
    plot_results(results)
