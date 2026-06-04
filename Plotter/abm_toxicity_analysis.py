"""
ABM Toxicity & Clearance Analysis
----------------------------------
Plots four panels for a single ABM simulation run:
  1. Zone-wise toxicity progression over time
  2. Spatial dead-cell heatmap at end of simulation
  3. Clearance vs. toxicity tradeoff (cumulative metabolized mass vs. dead cell count)
  4. Dead cell fraction per zone (bar chart)

Limitation note: toxicity_field accumulates from probabilistic mass destruction
(fraction_to_destroy), not from mechanistic NAPQI — it is a surrogate marker.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from StochasticModel.LobuleQuadrantABM import LobuleQuadrant
from Archive.config import Config

IMAGE_DIR = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ── Simulation ────────────────────────────────────────────────────────────────


def run_abm_simulation() -> tuple[LobuleQuadrant, Config, list, list, list, list]:
    """
    Run a single ABM simulation and record zone toxicity, dead cell counts,
    and metabolized mass at each recorded step.

    Returns
    -------
    quadrant        : final LobuleQuadrant state
    config          : Config instance
    time_pts        : list of recorded timestamps (s)
    zone_tox        : dict {1,2,3} -> list of mean toxicity_field values
    dead_counts     : dict {1,2,3} -> list of cumulative dead pixel counts
    metab_history   : list of cumulative metabolized mass (µmol)
    """
    config = Config()
    dose = 7.5 / 151.163 * 1e6
    quadrant_mass = dose / 4
    # quadrant_mass = config.DOSE / 4

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    time_pts = []
    zone_tox = {1: [], 2: [], 3: []}
    dead_counts = {1: [], 2: [], 3: []}
    metab_history = []

    step = 0
    stopping_threshold = quadrant_mass * 1e-3
    record_every = 500

    print("Running ABM simulation for toxicity analysis…")

    while True:
        quadrant.compute_flux()

        sin_mass = np.sum(quadrant.mass_grid * quadrant.sin_mask)

        if sin_mass < stopping_threshold:
            print(f"  Stopped at step {step} | Sin mass: {sin_mass:.4e} µmol")
            break
        if step > 50_000:
            print(f"  Max steps reached at step {step}")
            break

        if step % record_every == 0:
            quadrant.record(save_frame=True)
            time_pts.append(quadrant.current_time)
            metab_history.append(quadrant.total_mass_metab)

            # Zone toxicity means (only over alive pixels)
            for z in (1, 2, 3):
                alive_mask = (quadrant.zonation == z) & quadrant.hep_mask
                dead_mask = (quadrant.zonation == z) & quadrant.is_cell_dead

                zone_tox[z].append(
                    quadrant.toxicity_field[alive_mask].mean()
                    if alive_mask.any()
                    else 0.0
                )
                dead_counts[z].append(int(dead_mask.sum()))

            print(
                f"  Step {step:>6} | Metab: {quadrant.total_mass_metab:.4e} µmol | "
                f"Dead Z1/Z2/Z3: "
                f"{dead_counts[1][-1]}/{dead_counts[2][-1]}/{dead_counts[3][-1]}"
            )
        else:
            quadrant.record(save_frame=False)

        step += 1

    return quadrant, config, time_pts, zone_tox, dead_counts, metab_history


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_abm_toxicity(
    quadrant: LobuleQuadrant,
    config: Config,
    time_pts: list,
    zone_tox: dict,
    dead_counts: dict,
    metab_history: list,
) -> None:

    zone_colors = {1: "#2196F3", 2: "#FF9800", 3: "#F44336"}
    zone_labels = {
        1: "Zone 1 (periportal)",
        2: "Zone 2 (midzonal)",
        3: "Zone 3 (pericentral)",
    }

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Zone-wise toxicity progression ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for z in (1, 2, 3):
        ax1.plot(
            time_pts,
            zone_tox[z],
            color=zone_colors[z],
            label=zone_labels[z],
            linewidth=2,
        )
    ax1.axhline(
        y=quadrant.toxicity_threshold,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Death threshold",
    )
    ax1.set_title(
        "Zone-wise Toxicity Progression\n(mean toxicity_field, alive cells only)"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mean Toxicity Field (a.u.)")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 2: Spatial dead-cell heatmap ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    display = np.full(quadrant.physio_grid.shape, np.nan)
    # sinusoids: grey
    display[quadrant.sin_mask] = 0.0
    # alive hepatocytes: normalised toxicity
    t_max = quadrant.toxicity_field[quadrant.hep_mask].max()
    if t_max > 0:
        display[quadrant.hep_mask] = quadrant.toxicity_field[quadrant.hep_mask] / t_max
    # dead hepatocytes: highlighted value
    display[quadrant.is_cell_dead] = 1.5

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("lightgrey")
    im = ax2.imshow(display, cmap=cmap, vmin=0, vmax=1.5, origin="upper")
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels(["Sinusoid", "Low tox.", "High tox.", "Dead"])
    ax2.set_title("Spatial Dead-Cell Map\n(end of simulation)")
    ax2.set_xlabel("Grid column")
    ax2.set_ylabel("Grid row")

    # overlay zone contours
    for z, col in zone_colors.items():
        zone_px = (quadrant.zonation == z).astype(float)
        ax2.contour(zone_px, levels=[0.5], colors=[col], linewidths=0.8, alpha=0.5)

    # ── Panel 3: Clearance vs. toxicity tradeoff ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    total_dead = [
        dead_counts[1][i] + dead_counts[2][i] + dead_counts[3][i]
        for i in range(len(time_pts))
    ]

    ax3_r = ax3.twinx()
    (l1,) = ax3.plot(
        time_pts,
        metab_history,
        color="steelblue",
        linewidth=2,
        label="Cumul. metabolized (µmol)",
    )
    (l2,) = ax3_r.plot(
        time_pts,
        total_dead,
        color="crimson",
        linewidth=2,
        linestyle="--",
        label="Total dead pixels",
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cumulative Metabolized Mass (µmol)", color="steelblue")
    ax3_r.set_ylabel("Dead Hepatocyte Pixels", color="crimson")
    ax3.tick_params(axis="y", labelcolor="steelblue")
    ax3_r.tick_params(axis="y", labelcolor="crimson")
    ax3.set_title("Clearance vs. Hepatotoxicity\n(surrogate — mass destruction driven)")
    lines = [l1, l2]
    ax3.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax3.grid(True, linestyle=":", alpha=0.6)

    # ── Panel 4: Dead-cell fraction per zone (bar) ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    zone_total_px, zone_dead_px, zone_fracs = {}, {}, {}
    for z in (1, 2, 3):
        total = int(
            (
                (quadrant.zonation == z) & (quadrant.hep_mask | quadrant.is_cell_dead)
            ).sum()
        )
        dead = int(((quadrant.zonation == z) & quadrant.is_cell_dead).sum())
        zone_total_px[z] = total
        zone_dead_px[z] = dead
        zone_fracs[z] = dead / total if total > 0 else 0.0

    bars = ax4.bar(
        [zone_labels[z] for z in (1, 2, 3)],
        [zone_fracs[z] for z in (1, 2, 3)],
        color=[zone_colors[z] for z in (1, 2, 3)],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, z in zip(bars, (1, 2, 3)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{zone_dead_px[z]}/{zone_total_px[z]}\n({zone_fracs[z]*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Fraction of Zone Hepatocytes Dead")
    ax4.set_title("Dead-Cell Fraction per Zone\n(end of simulation)")
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.tick_params(axis="x", labelsize=8)

    plt.suptitle(
        "ABM Hepatotoxicity & Clearance Analysis\n"
        "(Toxicity = surrogate via mass-destruction, not mechanistic NAPQI)",
        fontsize=11,
        y=1.01,
    )

    out_path = os.path.join(IMAGE_DIR, "abm_toxicity_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quadrant, config, time_pts, zone_tox, dead_counts, metab_history = (
        run_abm_simulation()
    )
    plot_abm_toxicity(quadrant, config, time_pts, zone_tox, dead_counts, metab_history)
