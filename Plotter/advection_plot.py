"""
Plot the drug transit through the lobule using the LobuleQuadrant class.
This script simulates the diffusion and exchange of a drug injected into the lobule and visualizes the mass conservation and spatial distribution over time.
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from config import Config
from LobuleQuadrantDuplicate import LobuleQuadrant

config = Config()

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def plot_diffusion(quadrant: LobuleQuadrant):
    print(f"Plotting results...")
    plt.figure(figsize=(10, 6))

    plt.plot(
        quadrant.time_history,
        quadrant.grid_mass_history,
        label="Grid Mass (Drug in Lobule)",
        color="blue",
        linewidth=2,
    )

    plt.plot(
        quadrant.time_history,
        quadrant.exited_mass_history,
        label="Exited Mass (Drug in Central Vein)",
        color="green",
        linewidth=2,
    )

    plt.plot(
        quadrant.time_history,
        quadrant.total_system_mass_history,
        label="Total System Mass (Conservation Check)",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(
        quadrant.time_history,
        quadrant.metabolized_mass_history,
        label="Metabolized Mass (Drug Metabolized)",
        color="purple",
        linewidth=2,
    )

    plt.title("Mass Conservation During Lobule Transit", fontsize=14, fontweight="bold")
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Drug Mass (µmol)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, "mass_conservation_plot.png"), dpi=300)
    plt.show()


def get_diffusion_animation(quadrant: LobuleQuadrant):
    print("Generating video...")

    fig, ax = plt.subplots(figsize=(6, 6))
    initial_grid = quadrant.concentration_history[0] * config.V_PIXEL
    im = ax.imshow(initial_grid, cmap="viridis", origin="upper")

    ax.set_title("Lobule Drug Transit (Mass per Pixel)")
    ax.set_xlabel("X-axis (Pixels)")
    ax.set_ylabel("Y-axis (Pixels)")
    plt.colorbar(im, ax=ax, label="Mass (µmol)")

    def update(frame_index):
        current_grid = quadrant.concentration_history[frame_index] * config.V_PIXEL
        im.set_data(current_grid)

        max_val = np.max(current_grid)
        if max_val > 0:
            im.set_clim(0, max_val * 0.8)
        return [im]

    num_frames = len(quadrant.concentration_history)
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=50, blit=True
    )

    anim.save(
        os.path.join(IMAGE_FOLDER, "lobule_diffusion_animation.mp4"),
        writer="ffmpeg",
        fps=60,
    )
    plt.close()
    print("Video saved to images/lobule_diffusion_animation.mp4")


def run_simulation():
    # Concentration = mass / volume
    quadrant_mass = config.DOSE / 4
    print(f"Injecting mass: {quadrant_mass:.3e} µmol")

    quadrant = LobuleQuadrant(dose=quadrant_mass, exchange_on=True)

    step = 0
    stopping_threshold = quadrant_mass * 1e-3
    while quadrant.get_total_mass() > stopping_threshold:
        save_frame_interval = step % 20 == 0
        quadrant.compute_flux()
        quadrant.audit_mass2(step)
        quadrant.record(save_frame=save_frame_interval)
        step += 1

    print(f"Simulation completed in {step} steps.")
    return quadrant


if __name__ == "__main__":
    quadrant = run_simulation()
    plot_diffusion(quadrant)
    get_diffusion_animation(quadrant)
