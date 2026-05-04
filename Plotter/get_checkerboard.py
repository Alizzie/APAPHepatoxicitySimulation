import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from LobuleQuadrantABM import LobuleQuadrant
from config import Config


def plot_physio_grid(quadrant):
    """
    Plots the structural grid of the LobuleQuadrant.
    Hepatocytes (0) are orange, Sinusoids (1) are red.
    """
    # Create a custom colormap:
    # Index 0 (Hepatocytes) -> 'orange'
    # Index 1 (Sinusoids) -> 'red'
    cmap = colors.ListedColormap(["orange", "red"])

    # Set up the plot
    plt.figure(figsize=(8, 8))

    # Display the grid
    img = plt.imshow(quadrant.physio_grid, cmap=cmap, interpolation="nearest")

    # Create a custom legend for clarity
    import matplotlib.patches as mpatches

    hepa_patch = mpatches.Patch(color="orange", label="Hepatocytes (Tissue)")
    sin_patch = mpatches.Patch(color="red", label="Sinusoids (Blood)")
    plt.legend(
        handles=[hepa_patch, sin_patch], loc="upper right", bbox_to_anchor=(1.35, 1)
    )

    plt.title(
        f"Lobule Quadrant Grid\n({quadrant.grid_size}x{quadrant.grid_size} pixels)"
    )
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")

    # Optional: Turn off the axis ticks if you just want to see the pure grid
    # plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("physio_grid.png", dpi=300)


if __name__ == "__main__":

    quadrant = LobuleQuadrant()
    plot_physio_grid(quadrant)
