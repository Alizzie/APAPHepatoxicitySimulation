"""
Main plotter for the Stochastic (ABM) model.
Runs a comprehensive analysis in one go.
"""

import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from StochasticModel.config import Config
from StochasticModel.LobuleQuadrant import LobuleQuadrant
from Plotter.stochastic_only.metabolized_drug_rate_plotter import (
    plot_metabolized_rate_analysis,
)
from Plotter.stochastic_only.spatial_concentration_gradient_plotter import (
    plot_spatial_mass_gradient_analysis,
)
from Plotter.shared.simulate_diffusion_mass_conservation import (
    plot_diffusion,
    get_diffusion_animation,
)
from Plotter.stochastic_only.compartment_plotter import plot_compartment_analysis
from Plotter.stochastic_only.toxicity_plotter import (
    plot_toxicity_heatmap,
    plot_dead_cells,
    plot_zone_toxicity,
)

IMAGE_FOLDER = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


def run_simulation():
    config = Config()
    tiny_safe_dose = config.DOSE / 4

    quadrant = LobuleQuadrant(dose=tiny_safe_dose, allow_hepa_exchange=True)

    print(f"Starting Pulse Simulation. Injecting: {tiny_safe_dose:.3e} µmol")

    step = 0
    stopping_threshold = tiny_safe_dose * 1e-2

    spatial_history = []
    recorded_times = []
    sin_mass_history = []
    hep_mass_history = []
    pulse_count = 1

    while (
        tiny_safe_dose
        - quadrant.total_mass_metab
        - quadrant.total_mass_lost_to_necrosis
    ) > stopping_threshold and step < 120000:

        quadrant.compute_flux()

        # Pulse reinjection: when grid is nearly empty, reinject exited mass
        current_grid_mass = quadrant.get_total_mass()
        if (
            current_grid_mass < (tiny_safe_dose * 0.01)
            and quadrant.total_mass_exited > 0
        ):
            mass_to_reinject = quadrant.total_mass_exited
            quadrant.mass_grid[quadrant.inlet_pos] += mass_to_reinject
            quadrant.total_mass_exited = 0.0
            pulse_count += 1
            print(
                f"Step {step} | Pulse {pulse_count} | Re-injecting {mass_to_reinject:.3e} µmol"
            )

        save_time_interval = step % 250 == 0
        quadrant.record(save_frame=save_time_interval)

        sin_mass_history.append(float(np.sum(quadrant.mass_grid * quadrant.sin_mask)))
        hep_mass_history.append(float(np.sum(quadrant.mass_grid * quadrant.hep_mask)))

        if save_time_interval:
            quadrant.audit_mass(step)

        if save_time_interval and step > 0:
            spatial_history.append(np.diag(quadrant.mass_grid))
            recorded_times.append(quadrant.current_time)

        step += 1

    print(f"Simulation finished in {step} steps. Total Pulses: {pulse_count}")
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
    ) = run_simulation()

    plot_diffusion(quadrant)
    plot_metabolized_rate_analysis(quadrant)
    plot_compartment_analysis(quadrant, sin_mass_history, hep_mass_history)
    plot_spatial_mass_gradient_analysis(
        spatial_history, recorded_times, config, quadrant
    )
    plot_toxicity_heatmap(quadrant)
    plot_dead_cells(quadrant)
    plot_zone_toxicity(quadrant)
    get_diffusion_animation(quadrant)
