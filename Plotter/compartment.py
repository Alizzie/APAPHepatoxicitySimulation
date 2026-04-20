import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantDuplicate import LobuleQuadrant
from config import Config

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
        label="Drug Delivered (Central Vein)",
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

    plt.title("Two-Compartment Mass Distribution Over Time", fontweight="bold")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Drug Mass (µmol)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "compartment_mass_plot.png"), dpi=300)
    plt.show()


def run_simulation():
    config = Config()
    quadrant_mass = config.DOSE / 4
    print(f"Starting Compartment Analysis. Injecting: {quadrant_mass:.3e} µmol")

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3

    sin_mass_history = []
    hep_mass_history = []

    while quadrant.get_total_mass() > stopping_threshold:
        save_time_interval = step % 1000 == 0
        quadrant.compute_flux()
        quadrant.record()
        m_s = np.sum(quadrant.C * quadrant.sin_mask * config.V_PIXEL)
        m_h = np.sum(quadrant.C * quadrant.hep_mask * config.V_PIXEL)
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
