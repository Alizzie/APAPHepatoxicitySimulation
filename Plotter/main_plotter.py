"""
Combine different plotting scripts to run a comprehensive analysis in on go.
"""

import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from config import Config
from LobuleQuadrantDuplicate import LobuleQuadrant as ABMQuadrant
from LobuleQuadrant import LobuleQuadrant as PDEQuadrant
from exited_drug_rate import plot_exit_rate_analysis
from metabolized_drug_rate import plot_metabolized_rate_analysis
from spatial_concentration_gradient import plot_spatial_mass_gradient_analysis
from advection_plot import plot_diffusion, get_diffusion_animation
from compartment import plot_compartment_analysis
from toxicity_plot import (
    plot_toxicity_heatmap,
    plot_dead_cells,
    plot_zone_concentrations,
)

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def run_simulation(discrete: bool = False):
    config = Config()
    target_uM = config.DOSE / 5.7
    total_grid_volume = config.V_PIXEL * (config.N_PIXELS**2)
    quadrant_mass = target_uM * total_grid_volume

    if not discrete:
        quadrant = PDEQuadrant(dose=quadrant_mass, exchange_on=True)
    else:
        quadrant = ABMQuadrant(dose=quadrant_mass, exchange_on=True)

    print(f"Starting Simulation. Injecting: {quadrant_mass:.3e} µmol")

    step = 0
    stopping_threshold = quadrant_mass * 1e-2

    spatial_history = []
    recorded_times = []
    sin_mass_history = []
    hep_mass_history = []

    while quadrant.get_total_mass() > stopping_threshold and step < 50000:
        save_time_interval = step % 1000 == 0
        quadrant.compute_flux()
        quadrant.record(save_frame=save_time_interval)
        m_s = np.sum(quadrant.C * quadrant.sin_mask * config.V_PIXEL)
        m_h = np.sum(quadrant.C * quadrant.hep_mask * config.V_PIXEL)
        sin_mass_history.append(m_s)
        hep_mass_history.append(m_h)

        if save_time_interval and step > 0:
            spatial_history.append(np.diag(quadrant.C) * config.V_PIXEL)
            recorded_times.append(quadrant.current_time)
        quadrant.audit_mass2(step)

        step += 1

    print(f"Simulation finished in {step} steps. Generating plots...")
    print(f"Total Simulation Time: {quadrant.current_time:.2f} seconds")
    return (
        spatial_history,
        recorded_times,
        config,
        quadrant,
        sin_mass_history,
        hep_mass_history,
    )


if __name__ == "__main__":
    (
        spatial_history,
        recorded_times,
        config,
        quadrant,
        sin_mass_history,
        hep_mass_history,
    ) = run_simulation(discrete=True)

    plot_diffusion(quadrant)
    plot_exit_rate_analysis(quadrant)
    plot_metabolized_rate_analysis(quadrant)
    plot_compartment_analysis(quadrant, sin_mass_history, hep_mass_history)
    plot_spatial_mass_gradient_analysis(
        spatial_history, recorded_times, config, quadrant
    )
    plot_toxicity_heatmap(quadrant)
    plot_dead_cells(quadrant)
    plot_zone_concentrations(quadrant)

    get_diffusion_animation(quadrant)
