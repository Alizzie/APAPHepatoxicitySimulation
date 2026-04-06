import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from config import Config

from FullLobule import FullLobule
from LobuleQuadrant import LobuleQuadrant

config = Config()


class LobuleVisualizer:
    """
    Visualization for either a LobuleQuadrant or a FullLobule.

    Parameters
    ----------
    model : LobuleQuadrant or FullLobule
    """

    def __init__(self, model):
        self.model = model
        self.is_full = isinstance(model, FullLobule)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_physio_grid(self):
        if self.is_full:
            g = self.model.quadrants["top-left"].physio_grid
            # tile without mirroring — matches _assemble layout
            return np.block([[g, g], [g, g]])
        return self.model.physio_grid

    def _get_pressure(self):
        if self.is_full:
            q = self.model.quadrants
            return np.block(
                [
                    [q["top-left"].P, q["top-right"].P],
                    [q["bottom-left"].P, q["bottom-right"].P],
                ]
            )
        return self.model.P

    def _get_velocity(self):
        if self.is_full:
            return self.model.assemble("vx"), self.model.assemble("vy")
        return self.model.vx, self.model.vy

    def _get_concentration(self):
        if self.is_full:
            return self.model.assemble("C")
        return self.model.C

    def _get_quadrants(self):
        """Return dict of {label: LobuleQuadrant} for history/tracking plots."""
        if self.is_full:
            return self.model.quadrants
        return {self.model.direction: self.model}

    def _title(self, base):
        label = "Full Lobule" if self.is_full else self.model.direction
        return f"{base} — {label}"

    # ── Public plot methods ───────────────────────────────────────────────────

    def lattice(self):
        """Plot sinusoid/hepatocyte grid."""
        grid = self._get_physio_grid()
        cmap = ListedColormap(["#1f77b4", "#d62728"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            grid, origin="upper", interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )
        ax.set_title(self._title("Lattice"))
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def flow(self):
        """Plot pressure field and velocity magnitude side by side."""
        P = self._get_pressure()
        vx, vy = self._get_velocity()
        mag = np.sqrt(vx**2 + vy**2) * 6000  # m/s → cm/min

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(self._title("Flow"))

        im0 = axes[0].imshow(P / 1000, cmap="viridis", origin="upper")
        axes[0].set_title("Pressure (kPa)")
        plt.colorbar(im0, ax=axes[0], label="kPa")

        im1 = axes[1].imshow(
            mag + 1e-20, cmap="jet_r", origin="upper", norm=mcolors.LogNorm()
        )
        axes[1].set_title("Velocity (cm/min)")
        plt.colorbar(im1, ax=axes[1], label="cm/min")

        plt.tight_layout()
        plt.show()

    def quiver(self, skip=None, width=0.003):
        P = self._get_pressure()
        vx, vy = self._get_velocity()
        if skip is None:
            skip = max(1, vx.shape[0] // 20)

        vx_s = vx[::skip, ::skip]
        vy_s = vy[::skip, ::skip]

        # normalize to unit length so all arrows are equally visible
        mag = np.sqrt(vx_s**2 + vy_s**2)
        mag[mag == 0] = 1
        vx_n = vx_s / mag
        vy_n = vy_s / mag

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(P / 1000, cmap="viridis", origin="upper", alpha=0.6)
        ax.quiver(
            np.arange(0, vx.shape[1], skip),
            np.arange(0, vx.shape[0], skip),
            vx_n,
            -vy_n,
            color="red",
            scale=30,
            width=width,
        )
        ax.set_title(self._title("Velocity Field"))
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def quiver_quadrants(self, skip=None, width=0.003):
        """2x2 per-quadrant velocity vectors. FullLobule only."""
        if not self.is_full:
            raise ValueError("quiver_quadrants requires a FullLobule model.")

        dirs = ["top-left", "top-right", "bottom-left", "bottom-right"]
        pos = {
            "top-left": (0, 0),
            "top-right": (0, 1),
            "bottom-left": (1, 0),
            "bottom-right": (1, 1),
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle("Velocity Field — per quadrant", fontsize=13)

        for d in dirs:
            q = self.model.quadrants[d]
            ax = axes[pos[d]]
            sk = skip or max(1, q.vx.shape[0] // 20)

            vx_s = q.vx[::sk, ::sk]
            vy_s = q.vy[::sk, ::sk]
            mag = np.sqrt(vx_s**2 + vy_s**2)
            mag[mag == 0] = 1

            ax.imshow(q.P / 1000, cmap="viridis", origin="upper", alpha=0.6)
            ax.quiver(
                np.arange(0, q.vx.shape[1], sk),
                np.arange(0, q.vx.shape[0], sk),
                vx_s / mag,
                -vy_s / mag,
                color="red",
                scale=30,
                width=width,
            )
            ax.set_title(d)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def concentration(self, step=0):
        """Plot sinusoid, hepatocyte, and total drug concentration."""
        C = self._get_concentration()
        grid = self._get_physio_grid()

        C_sin = np.where(grid == 1, C, 0)
        C_hepa = np.where(grid == 0, C, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(self._title(f"Drug Concentration  t={step * config.DT:.3f}s"))

        for ax, data, title, cmap in zip(
            axes,
            [C_sin, C_hepa, C],
            ["Sinusoids", "Hepatocytes", "Total"],
            ["Reds", "Blues", "viridis"],
        ):
            im = ax.imshow(data, cmap=cmap, origin="upper")
            ax.set_title(title)
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Concentration (µM)")

        plt.tight_layout()
        plt.show()

    def quadrants_side_by_side(self, step=0):
        """
        Plot all four quadrant concentrations in a 2x2 grid with a shared
        colour scale. Works only for FullLobule.
        """
        if not self.is_full:
            raise ValueError("quadrants_side_by_side requires a FullLobule model.")

        dirs = ["top-left", "top-right", "bottom-left", "bottom-right"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Quadrants at t = {step * config.DT:.3f}s", fontsize=13)

        positions = {
            "top-left": axes[0, 0],
            "top-right": axes[0, 1],
            "bottom-left": axes[1, 0],
            "bottom-right": axes[1, 1],
        }

        for d, ax in positions.items():
            im = ax.imshow(
                self.model.quadrants[d].C,
                cmap="viridis",
                origin="upper",
            )
            ax.set_title(d)
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Concentration (µM)")

        plt.tight_layout()
        plt.show()

    def history(self):
        """Plot total mass, inlet, and outlet concentration over time."""
        quadrants = self._get_quadrants()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(self._title("History"))

        for d, q in quadrants.items():
            t = q.time_history
            axes[0].plot(t, q.total_mass_history, label=d)
            axes[1].plot(t, q.inlet_concentration_history, label=d)
            axes[2].plot(t, q.outlet_concentration_history, label=d)

        axes[0].set_title("Total mass per Grid (µmol)")
        axes[1].set_title("Inlet concentration (µM)")
        axes[2].set_title("Outlet concentration (µM)")
        for ax in axes:
            ax.set_xlabel("t (s)")
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.show()
