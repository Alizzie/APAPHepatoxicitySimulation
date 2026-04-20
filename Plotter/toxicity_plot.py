"""
Plot the toxicity of the drug in the lobule using the LobuleQuadrant class with metabolism enabled.
This script simulates the metabolism of a drug injected into the lobule and visualizes the hepatocyte toxicity over time, as well as the zonation effects on drug concentration and toxicity.
"""

import sys
import os
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from config import Config
from LobuleQuadrant import LobuleQuadrant

config = Config()

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def plot_toxicity_heatmap(quadrant: LobuleQuadrant):

    if isinstance(quadrant, LobuleQuadrant):
        toxicity_grid = quadrant.metabolism.get_toxicity_field()
    else:
        toxicity_grid = quadrant.toxicity_field

    plt.figure(figsize=(8, 6))
    im = plt.imshow(toxicity_grid, cmap="magma", origin="upper", vmin=0)
    cbar = plt.colorbar(im)
    cbar.set_label("Protein Abdducts (Ci) [µM]", rotation=270, labelpad=15)

    plt.title("APAP Hepatocyte Toxicity Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("X-axis (Pixel)", fontsize=12)
    plt.ylabel("Y-axis (Pixel)", fontsize=12)
    plt.scatter(
        [quadrant.grid_size - 1],
        [quadrant.grid_size - 1],
        color="red",
        facecolors="none",
        s=200,
        label="Central Vein",
    )
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "toxicity_heatmap.png"), dpi=300)
    plt.show()


def plot_dead_cells(quadrant: LobuleQuadrant):

    if isinstance(quadrant, LobuleQuadrant):
        is_alive = quadrant.metabolism.is_alive
    else:
        is_alive = ~quadrant.is_cell_dead

    plt.figure(figsize=(8, 6))
    plt.imshow(is_alive, cmap="gray", origin="upper")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Dead", "Alive"])
    plt.title("Hepatocyte Viability", fontsize=14, fontweight="bold")
    plt.xlabel("X-axis (Pixel)", fontsize=12)
    plt.ylabel("Y-axis (Pixel)", fontsize=12)
    plt.scatter(
        [quadrant.grid_size - 1],
        [quadrant.grid_size - 1],
        color="red",
        facecolors="none",
        s=200,
        label="Central Vein",
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "hepatocyte_viability.png"), dpi=300)
    plt.show()


def plot_zone_concentrations(quadrant: LobuleQuadrant):

    if not isinstance(quadrant, LobuleQuadrant):
        zones = quadrant.get_toxicity_zone_means()
        plt.bar(zones.keys(), zones.values(), color="salmon")
        plt.title("Average Toxicity by Zone", fontsize=14, fontweight="bold")
        plt.xlabel("Zone", fontsize=12)
        plt.ylabel("Average Protein Adducts (Ci) [µM]", fontsize=12)
        plt.xticks([1, 2, 3])
    else:
        zone_concentrations = quadrant.metabolism.get_zone_means()
        zones = list(zone_concentrations.keys())

        metrics = ["P", "NAPQI", "GSH", "Ci", "S"]
        titles = [
            "APAP",
            "NAPQI",
            "Glutathione (GSH)",
            "Protein Adducts (Ci)",
            "Sulfate (S)",
        ]
        colors = ["skyblue", "salmon", "lightgreen", "mediumpurple", "lightcoral"]

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [zone_concentrations[zone][metric] for zone in zones]
            axes[i].bar(zones, values, color=colors[i])
            axes[i].set_title(
                f"Average {titles[i]} Concentration by Zone",
                fontsize=12,
                fontweight="bold",
            )
            axes[i].set_xlabel("Zone", fontsize=10)
            axes[i].set_ylabel(f"Average {titles[i]} Concentration (µM)", fontsize=10)
            axes[i].set_xticks([1, 2, 3])

        plt.suptitle(
            "Hepatocyte Metabolism Profile by Zonation",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "zone_toxicity.png"), dpi=300)
    plt.show()


def run_simulation():
    # Concentration = mass / volume
    target_uM = config.DOSE / 5.7
    total_grid_volume = config.V_PIXEL * (config.N_PIXELS**2)
    quadrant_mass = target_uM * total_grid_volume

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    step = 0
    for _ in range(20000):
        save_frame_interval = step % 20 == 0
        quadrant.compute_flux()
        quadrant.audit_mass2(step)
        quadrant.record(save_frame=save_frame_interval)
        step += 1

    print(f"Simulation completed in {step} steps.")
    return quadrant


if __name__ == "__main__":
    quadrant = run_simulation()
    plot_toxicity_heatmap(quadrant)
    plot_dead_cells(quadrant)
    plot_zone_concentrations(quadrant)
