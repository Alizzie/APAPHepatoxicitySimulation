"""
Plot the effect of varying cellular efflux on drug delivery to the sinusoid.
This script runs multiple simulations with different efflux rates and compares the cumulative mass exited.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantDuplicate import LobuleQuadrant
from config import Config

IMAGE_DIR = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def plot_efflux_analysis(results: dict, quadrant_mass: float):
    plt.figure(figsize=(10, 6))
    colors = ["red", "blue", "green", "purple"]

    for (label, data), color in zip(results.items(), colors):
        plt.plot(data["time"], data["exited"], label=label, color=color)

    plt.axhline(
        y=quadrant_mass, color="black", linestyle="--", alpha=0.5, label="Total Dose"
    )
    plt.title("Effect of Cellular Efflux on Drug Delivery")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Mass Exited (µmol)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "efflux_sweep.png"), dpi=300)
    plt.show()


def decide_scenarios():
    quad = LobuleQuadrant(dose=Config().DOSE / 4, exchange_on=True)
    if quad.__module__ == "LobuleQuadrantDuplicate":
        return {
            "No Efflux": 0.0,
            "Low Efflux": 0.0005,
            "Normal Efflux": 0.002,
            "High Efflux": 0.005,
        }
    else:
        return {
            "No Efflux": 0.0,
            "Low Efflux": 0.5,
            "Normal Efflux": 1.0,
            "High Efflux": 5.0,
        }


def run_simulation():
    quadrant_mass = Config().DOSE / 4
    scenarios = decide_scenarios()
    results = {}

    for label, multiplier in scenarios.items():
        print(f"\nRunning scenario: {label} (Efflux Multiplier: {multiplier})")

        config = Config()
        if LobuleQuadrant.__module__ == "LobuleQuadrantDuplicate":
            quadrant = LobuleQuadrant(
                dose=quadrant_mass,
                exchange_on=True,
                base_efflux_pct=multiplier,
            )
            print(f"Set base_efflux_pct to {multiplier:.3e} for {label}")
        else:
            config.CL_EFFLUX *= multiplier
            print(f"Adjusted CL_EFFLUX: {config.CL_EFFLUX:.3e} L/s per pixel")
            quadrant = LobuleQuadrant(
                dose=quadrant_mass, exchange_on=True, config_override=config
            )

        step = 0
        stopping_threshold = quadrant_mass * 1e-3

        while True:
            quadrant.compute_flux()

            sinusoid_mass = np.sum(quadrant.C * quadrant.sin_mask * config.V_PIXEL)

            if sinusoid_mass < stopping_threshold:
                print(
                    f"Stopping simulation for {label} at step {step} | "
                    f"Sinusoid Mass: {sinusoid_mass:.6e} µmol"
                )
                break

            if step > 50000:
                print(
                    f"Reached maximum steps for {label} at step {step} | "
                    f"Sinusoid Mass: {sinusoid_mass:.6e} µmol"
                )
                break

            save_time_interval = step % 1000 == 0
            quadrant.record(save_frame=save_time_interval)
            step += 1

            print(
                f"Step {step} | Total Mass in Grid: {quadrant.get_total_mass():.6e} µmol | "
                f"Sinusoid Mass: {sinusoid_mass:.6e} µmol"
            )

        results[label] = {
            "time": quadrant.time_history,
            "exited": quadrant.exited_mass_history,
        }

    return results, quadrant_mass


if __name__ == "__main__":
    results, quadrant_mass = run_simulation()
    plot_efflux_analysis(results, quadrant_mass)
