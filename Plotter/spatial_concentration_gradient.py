"""
Plot the spatial concentration gradient of the drug along the diagonal of the lobule over time.
This script simulates the transit of a drug through the lobule and visualizes how the concentration
changes spatially from the inlet to the central vein at different time points.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantDuplicate import LobuleQuadrant
from config import Config


def plot_spatial_mass_gradient_analysis(
    spatial_history: list,
    recorded_times: list,
    config: Config,
    quadrant: LobuleQuadrant,
):
    print(f"Plotting spatial mass gradient...")

    plt.figure(figsize=(10, 6))

    pixel_width = config.LOBULE_SIZE / quadrant.grid_size
    distances = np.arange(len(spatial_history[0])) * pixel_width * np.sqrt(2)

    num_lines = 10
    indicies_to_plot = np.geomspace(1, len(spatial_history), num_lines, dtype=int) - 1
    cmap = plt.get_cmap("tab10")

    for i, idx in enumerate(indicies_to_plot):
        time_val = recorded_times[idx]
        concentration_data = spatial_history[idx]

        plt.plot(
            distances,
            concentration_data,
            label=f"Time = {time_val:.2f} s",
            color=cmap(i % 10),
            linewidth=2,
        )

    plt.title("Spatial Mass Gradient Along Diagonal of Lobule", fontsize=14)
    plt.xlabel("Distance from Inlet (m)", fontsize=12)
    plt.ylabel("Total Mass (µmol)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(parent_dir, "images", "spatial_mass_gradient.png"),
        dpi=300,
    )
    plt.show()

    print("Plotting 3D spatiotemporal mass surface...")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    pixel_width = config.LOBULE_SIZE / quadrant.grid_size
    distances = np.arange(len(spatial_history[0])) * pixel_width * np.sqrt(2)
    X, Y = np.meshgrid(distances, recorded_times)
    Z = np.array(spatial_history)

    surf = ax.plot_surface(X, Y, Z, cmap="plasma", edgecolor="none", alpha=0.8)
    ax.set_title("3D Spatiotemporal Drug Transit Wave", fontsize=14)
    ax.set_xlabel("Distance from Inlet (m)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_zlabel("Total Mass (µmol)", fontsize=12)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Mass (µmol)")
    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.savefig(
        os.path.join(parent_dir, "images", "spatiotemporal_mass_surface.png"),
        dpi=300,
    )
    plt.show()


def run_simulation():
    config = Config()
    quadrant_mass = config.DOSE / 4
    print(
        f"Starting Spatial Mass Gradient Analysis. Injecting: {quadrant_mass:.3e} µmol"
    )

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3

    spatial_history = []
    recorded_times = []

    while quadrant.get_total_mass() > stopping_threshold:
        save_time_interval = step % 1000 == 0
        quadrant.compute_flux()
        quadrant.record(save_frame=save_time_interval)

        if save_time_interval and step > 0:
            spatial_history.append(np.diag(quadrant.C) * config.V_PIXEL)
            recorded_times.append(quadrant.current_time)

        print(f"Step {step} | Total Mass: {quadrant.get_total_mass():.6e} µmol")

        step += 1

    print(f"Simulation finished in {step} steps. Generating plots...")
    print(f"Total Simulation Time: {quadrant.current_time:.2f} seconds")
    return spatial_history, recorded_times, config, quadrant


if __name__ == "__main__":
    spatial_history, recorded_times, config, quadrant = run_simulation()
    plot_spatial_mass_gradient_analysis(
        spatial_history, recorded_times, config, quadrant
    )
