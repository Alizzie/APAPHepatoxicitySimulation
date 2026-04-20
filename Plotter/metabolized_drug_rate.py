"""
Plot the rate of drug metabolized in the central vein over time.
This script simulates the transit of a drug through the lobule and visualizes the metabolized rate
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantDuplicate import LobuleQuadrant
from config import Config


def plot_metabolized_rate_analysis(quadrant: LobuleQuadrant):

    plt.figure(figsize=(10, 6))
    metabolized_rate = np.gradient(
        quadrant.metabolized_mass_history, quadrant.time_history
    )
    plt.plot(quadrant.time_history, metabolized_rate, color="purple", linewidth=2)
    plt.title(
        "Rate of Drug Metabolism in Central Vein Over Time",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Metabolism Rate (µmol/s)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.fill_between(
        quadrant.time_history, 0, metabolized_rate, color="purple", alpha=0.3
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(parent_dir, "images", "metabolized_rate_plot.png"), dpi=300
    )
    plt.show()


def run_simulation():
    config = Config()
    quadrant_mass = config.DOSE / 4
    print(f"Starting Metabolized Rate Analysis. Injecting: {quadrant_mass:.3e} µmol")

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3

    while quadrant.get_total_mass() > stopping_threshold:
        save_frame_interval = step % 20 == 0
        quadrant.compute_flux()
        quadrant.record(save_frame=save_frame_interval)

        print(f"Step {step} | Total Mass: {quadrant.get_total_mass():.6e} µmol")
        step += 1

    print(f"Simulation finished in {step} steps. Generating plots...")
    return quadrant


if __name__ == "__main__":
    quadrant = run_simulation()
    plot_metabolized_rate_analysis(quadrant)
