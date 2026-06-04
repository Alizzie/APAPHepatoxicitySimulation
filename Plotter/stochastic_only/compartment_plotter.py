"""Simulate the mass distribution across compartments (sinusoid, hepatocytes, exited) over time in the ABM. This is used to analyze how the drug moves and is metabolized within the lobule quadrant."""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from StochasticModel.LobuleQuadrant import LobuleQuadrant
from StochasticModel.config import Config

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def plot_compartment_analysis(
    quadrant: LobuleQuadrant, sin_mass_history: list, hep_mass_history: list
):
    plt.figure(figsize=(10, 6))

    plt.plot(
        quadrant.time_history,
        sin_mass_history,
        label="Drug in Bloodstream",
        color="red",
        linewidth=2.5,
    )
    plt.plot(
        quadrant.time_history,
        hep_mass_history,
        label="Drug Trapped in Tissue",
        color="blue",
        linewidth=2.5,
    )
    plt.plot(
        quadrant.time_history,
        quadrant.exited_mass_history,
        label="Exited Drug",
        color="green",
        linewidth=2.5,
    )

    plt.plot(
        quadrant.time_history,
        quadrant.metabolized_mass_history,
        label="Metabolized Drug",
        color="purple",
        linewidth=2.5,
    )

    plt.plot(
        quadrant.time_history,
        quadrant.mass_lost_to_necrosis_history,
        label="Mass Lost to Necrosis",
        color="orange",
        linewidth=2.5,
    )

    plt.title("Mass Distribution Over Time", fontweight="bold")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Drug Mass (µmol)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "mass_distribution_abm.png"), dpi=300)
    plt.show()


def run_simulation():
    config = Config()
    quadrant_mass = config.DOSE / 4
    print(f"Starting Compartment Analysis. Injecting: {quadrant_mass:.3e} µmol")

    quadrant = LobuleQuadrant(dose=quadrant_mass, allow_hepa_exchange=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3

    sin_mass_history = []
    hep_mass_history = []

    while quadrant.get_total_mass() > stopping_threshold:

        if step > 30000:
            break

        save_time_interval = step % 1000 == 0
        quadrant.compute_flux()
        quadrant.record()
        m_s = np.sum(quadrant.mass_grid * quadrant.sin_mask)
        m_h = np.sum(quadrant.mass_grid * quadrant.hep_mask)
        sin_mass_history.append(m_s)
        hep_mass_history.append(m_h)

        if save_time_interval:
            print(f"Step {step} | Total Mass: {quadrant.get_total_mass():.6e} µmol")

        step += 1

    print(f"Simulation finished in {step} steps. Generating plots...")
    return quadrant, sin_mass_history, hep_mass_history


if __name__ == "__main__":
    quadrant, sin_mass_history, hep_mass_history = run_simulation()
    plot_compartment_analysis(quadrant, sin_mass_history, hep_mass_history)
