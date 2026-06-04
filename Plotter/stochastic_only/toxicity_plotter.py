"""
Plot the toxicity of the drug in the lobule using the Stochastic (ABM) model.
Visualizes hepatocyte toxicity, dead cells, and zone-averaged toxicity profiles.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from StochasticModel.config import Config
from StochasticModel.LobuleQuadrant import LobuleQuadrant

config = Config()

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def plot_toxicity_heatmap(quadrant: LobuleQuadrant):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(quadrant.toxicity_field, cmap="magma", origin="upper", vmin=0)
    cbar = plt.colorbar(im)
    cbar.set_label("Toxicity Field (NAPQI mass, µmol)", rotation=270, labelpad=15)
    plt.title("APAP Hepatocyte Toxicity Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("X-axis (Pixel)", fontsize=12)
    plt.ylabel("Y-axis (Pixel)", fontsize=12)
    plt.scatter(
        [quadrant.outlet_pos[1]],
        [quadrant.outlet_pos[0]],
        color="red",
        facecolors="none",
        s=200,
        label="Central Vein",
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "toxicity_heatmap_abm.png"), dpi=300)
    plt.show()


def plot_dead_cells(quadrant: LobuleQuadrant):
    is_alive = ~quadrant.is_cell_dead

    plt.figure(figsize=(8, 6))
    plt.imshow(is_alive, cmap="gray", origin="upper")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Dead", "Alive"])
    plt.title("Hepatocyte Viability", fontsize=14, fontweight="bold")
    plt.xlabel("X-axis (Pixel)", fontsize=12)
    plt.ylabel("Y-axis (Pixel)", fontsize=12)
    plt.scatter(
        [quadrant.outlet_pos[1]],
        [quadrant.outlet_pos[0]],
        color="red",
        facecolors="none",
        s=200,
        label="Central Vein",
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "hepatocyte_viability_abm.png"), dpi=300)
    plt.show()


def plot_zone_toxicity(quadrant: LobuleQuadrant):
    zones = quadrant.get_toxicity_zone_means()

    plt.figure(figsize=(8, 5))
    colors = ["#66BB6A", "#FFA726", "#EF5350"]
    bars = plt.bar(
        [f"Zone {z}" for z in zones.keys()],
        zones.values(),
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, val in zip(bars, zones.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.axhline(
        y=config.TOXICITY_THRESHOLD,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Death threshold",
    )
    plt.title("Average Toxicity by Zone", fontsize=14, fontweight="bold")
    plt.xlabel("Zone", fontsize=12)
    plt.ylabel("Mean Toxicity Field (µmol)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "zone_toxicity_abm.png"), dpi=300)
    plt.show()


def run_simulation():
    quadrant_mass = config.DOSE / 4

    quadrant = LobuleQuadrant(dose=quadrant_mass, allow_hepa_exchange=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3

    while True:
        quadrant.compute_flux()
        sin_mass = float(np.sum(quadrant.mass_grid * quadrant.sin_mask))

        if sin_mass < stopping_threshold:
            print(f"  Stopped at step {step} | Sin mass: {sin_mass:.4e} µmol")
            break
        if step > 50_000:
            print(f"  Max steps reached at step {step}")
            break

        quadrant.record(save_frame=(step % 20 == 0))
        step += 1

    print(f"Simulation completed in {step} steps.")
    return quadrant


if __name__ == "__main__":
    quadrant = run_simulation()
    plot_toxicity_heatmap(quadrant)
    plot_dead_cells(quadrant)
    plot_zone_toxicity(quadrant)
