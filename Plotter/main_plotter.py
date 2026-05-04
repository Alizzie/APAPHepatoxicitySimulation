"""
Combine different plotting scripts to run a comprehensive analysis in on go.
"""

import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from config import Config
from LobuleQuadrantABM import LobuleQuadrant as ABMQuadrant
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


import numpy as np


def run_simulation(discrete: bool = False):
    config = Config()

    target_concentration = 12 / 151.163 * 1e6
    # target_concentration = config.DOSE / 5.7
    # tiny_safe_dose = target_concentration * (config.V_PIXEL * (config.N_PIXELS**2))

    target_concentration = 12 / 151.163 * 1e6
    tiny_safe_dose = target_concentration / 4

    # Initialize the liver WITH the dose (so it starts flowing immediately)
    if not discrete:
        quadrant = PDEQuadrant(dose=tiny_safe_dose, exchange_on=True)
    else:
        quadrant = ABMQuadrant(dose=tiny_safe_dose, exchange_on=True)

    print(f"Starting Pulse Simulation. Injecting: {tiny_safe_dose:.3e} µmol")

    step = 0
    # Stop the whole simulation when 99% of the drug is permanently destroyed/trapped
    stopping_threshold = tiny_safe_dose * 1e-2

    spatial_history = []
    recorded_times = []
    sin_mass_history = []
    hep_mass_history = []

    pulse_count = 1

    # Run until the drug is metabolized, or we hit a max step limit
    while (
        tiny_safe_dose
        - quadrant.total_mass_metab
        - quadrant.total_mass_lost_to_necrosis
    ) > stopping_threshold and step < 120000:

        # --- LIVER PHYSICS ---
        quadrant.compute_flux()

        # --- THE PULSE REINJECTION LOGIC ---
        # Check if the liver grid is basically empty (e.g., less than 1% of the original dose is still flowing)
        current_grid_mass = quadrant.get_total_mass()

        if (
            current_grid_mass < (tiny_safe_dose * 0.01)
            and quadrant.total_mass_exited > 0
        ):

            # 1. Grab the mass waiting in the catch basin
            mass_to_reinject = quadrant.total_mass_exited

            # 2. Inject it back into the inlet
            quadrant.mass_grid[quadrant.inlet_pos] += mass_to_reinject

            # 3. ZERO OUT the exit tracker so we don't double-count the mass!
            quadrant.total_mass_exited = 0.0

            pulse_count += 1
            print(
                f"Step {step} | Pulse {pulse_count} | Re-injecting {mass_to_reinject:.3e} µmol"
            )

        # --- RECORDING & AUDITING ---
        save_time_interval = step % 250 == 0
        quadrant.record(save_frame=save_time_interval)
        m_s = np.sum(quadrant.mass_grid * quadrant.sin_mask)
        m_h = np.sum(quadrant.mass_grid * quadrant.hep_mask)
        sin_mass_history.append(m_s)
        hep_mass_history.append(m_h)

        # Because we cleanly empty the exit tracker when we re-inject,
        # your original local audit will work perfectly!
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
