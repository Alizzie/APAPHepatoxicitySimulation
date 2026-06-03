import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from LobuleQuadrantABM import LobuleQuadrant


def plot_toxicity_zones(lobule, ax=None, title="Hepatic Zonation") -> plt.Axes:
    """
    Plot the lobule grid coloured by toxicity zone (1, 2, 3).
    Sinusoid pixels are shown in a neutral grey.

    Parameters
    ----------
    lobule : LobuleQuadrant
        A fully initialised LobuleQuadrant instance.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates a new figure if None.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # ── build display grid ──────────────────────────────────────────────────
    # 0 = sinusoid, 1/2/3 = hepatocyte zone
    display = np.where(lobule.hep_mask, lobule.zonation, 0)

    # ── colours: sinusoid grey, zone1 green, zone2 amber, zone3 red ─────────
    cmap = ListedColormap(["#BDBDBD", "#66BB6A", "#FFA726", "#EF5350"])

    fig_created = ax is None
    if fig_created:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(
        display, cmap=cmap, vmin=0, vmax=3, origin="upper", interpolation="nearest"
    )

    # ── inlet / outlet markers ───────────────────────────────────────────────
    ax.plot(
        *lobule.inlet_pos[::-1],
        marker="*",
        markersize=12,
        color="royalblue",
        label="Inlet (portal)"
    )
    ax.plot(
        *lobule.outlet_pos[::-1],
        marker="*",
        markersize=12,
        color="navy",
        label="Outlet (central vein)"
    )

    # ── legend ───────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color="#BDBDBD", label="Sinusoid"),
        mpatches.Patch(color="#66BB6A", label="Zone 1 — periportal"),
        mpatches.Patch(color="#FFA726", label="Zone 2 — midzonal"),
        mpatches.Patch(color="#EF5350", label="Zone 3 — centrilobular"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")
    ax.tick_params(labelsize=8)

    if fig_created:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    lobule = LobuleQuadrant()
    plot_toxicity_zones(lobule)
