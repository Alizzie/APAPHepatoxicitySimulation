"""
Plot the effect of varying hepatocyte absorption (uptake) rate on drug delivery.
Higher uptake → more drug sequestered in hepatocytes → slower sinusoidal washout.
Mirrors efflux_washout_plot.py but sweeps CL_INFLUX / base_uptake_pct instead.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantABM import LobuleQuadrant
from config import Config

IMAGE_DIR = os.path.join(parent_dir, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_duplicate_module() -> bool:
    """Return True when the ABM variant is the 'Duplicate' flavour."""
    return LobuleQuadrant.__module__ == "LobuleQuadrantABM"


def decide_scenarios() -> dict[str, float]:
    """
    Return label → multiplier mapping.
    Duplicate variant: base_uptake_pct is set directly (absolute values).
    Standard variant : CL_INFLUX is scaled by the multiplier.
    """
    if _is_duplicate_module():
        return {
            "No Uptake": 0.0,
            "Low Uptake": 0.003,
            "Normal Uptake": 0.006,
            "High Uptake": 0.012,
        }
    else:
        return {
            "No Uptake": 0.0,
            "Low Uptake": 0.5,
            "Normal Uptake": 1.0,
            "High Uptake": 2.0,
        }


# ── Simulation ────────────────────────────────────────────────────────────────


def run_simulation() -> tuple[dict, float]:
    quadrant_mass = Config().DOSE / 4
    scenarios = decide_scenarios()
    results: dict[str, dict] = {}

    for label, multiplier in scenarios.items():
        print(f"\nRunning scenario: {label}  (Uptake multiplier / value: {multiplier})")

        config = Config()

        if _is_duplicate_module():
            quadrant = LobuleQuadrant(
                dose=quadrant_mass,
                exchange_on=True,
                base_uptake_pct=multiplier,  # set directly
            )
            print(f"  base_uptake_pct = {multiplier:.4f}")
        else:
            config.CL_INFLUX *= multiplier
            print(f"  Adjusted CL_INFLUX: {config.CL_INFLUX:.3e} L/s per pixel")
            quadrant = LobuleQuadrant(
                dose=quadrant_mass,
                exchange_on=True,
                config_override=config,
            )

        step = 0
        stopping_threshold = quadrant_mass * 1e-3

        while True:
            quadrant.compute_flux()

            sinusoid_mass = np.sum(quadrant.C * quadrant.sin_mask * config.V_PIXEL)

            if sinusoid_mass < stopping_threshold:
                print(
                    f"  Stopping at step {step} | "
                    f"Sinusoid Mass: {sinusoid_mass:.6e} µmol"
                )
                break

            if step > 50_000:
                print(
                    f"  Max steps reached at step {step} | "
                    f"Sinusoid Mass: {sinusoid_mass:.6e} µmol"
                )
                break

            if step % 1000 == 0:
                quadrant.record(save_frame=True)
                print(
                    f"  Step {step:>6} | Grid Mass: {quadrant.get_total_mass():.4e} µmol | "
                    f"Sin Mass: {sinusoid_mass:.4e} µmol | "
                    f"Exited: {quadrant.total_mass_exited:.4e} µmol"
                )
            else:
                quadrant.record(save_frame=False)

            step += 1

        results[f"{label} = {multiplier}"] = {
            "time": quadrant.time_history,
            "exited": quadrant.exited_mass_history,
        }

    return results, quadrant_mass


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_uptake_analysis(results: dict, quadrant_mass: float) -> None:
    colors = ["red", "blue", "green", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: cumulative exited mass ───────────────────────────────────
    ax = axes[0]
    for (label, data), color in zip(results.items(), colors):
        ax.plot(data["time"], data["exited"], label=label, color=color)

    ax.axhline(
        y=quadrant_mass,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Total Dose",
    )
    ax.set_title("Effect of Uptake Rate on Drug Washout\n(Cumulative Exited Mass)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Mass Exited (µmol)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower right")

    # ── Right panel: fraction exited (normalised) ────────────────────────────
    ax2 = axes[1]
    for (label, data), color in zip(results.items(), colors):
        fraction = [e / quadrant_mass for e in data["exited"]]
        ax2.plot(data["time"], fraction, label=label, color=color)

    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="100 % Dose")
    ax2.set_title("Effect of Uptake Rate on Drug Washout\n(Fraction of Dose Exited)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Fraction of Dose Exited (–)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(IMAGE_DIR, "uptake_sweep.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nFigure saved → {out_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, quadrant_mass = run_simulation()
    plot_uptake_analysis(results, quadrant_mass)
