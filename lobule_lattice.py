import numpy as np
import matplotlib.pyplot as plt
from config import Config
from scipy.ndimage import distance_transform_edt
from random import seed

seed(42)


class LiverLobuleLattice:
    """
    Class representing a lattice of squared hepatocytes and sinusoids organized into squared liver lobules.
    Each lobule corner contains a portal triad, and the center contains a central vein, forming a grid of lobules across the canvas.
    The lattice is defined by the number of lobules per side (GRID_N), the number of hepatocytes per lobule side (CYCTES_N), and the size of each hepatocyte in pixels (CYCTES_PX).

    Sinusoids are generated as paths from the portal triads to the central vein, with a certain probability of branching to create a more realistic vascular network.

    The class provides methods to calculate the positions of the central veins and portal triads, as well as to visualize the lattice using Matplotlib.

    """

    def __init__(self, config: Config):
        self.config = config
        self.lobule_size = (
            self.config.CYCTES_N * self.config.CYCTES_PX
        )  # size of each lobule in pixels
        self.lattice_size = (
            self.config.GRID_N * self.lobule_size
        )  # total size of the lattice in pixels
        self.canvas_size = (
            self.lattice_size + 2 * self.config.MARGIN
        )  # total canvas size including margins

        self.lattice = self._create_lattice()

    def _create_lattice(self):
        lat_model = np.zeros((self.lattice_size, self.lattice_size), dtype=int)

        for i in range(self.config.GRID_N):
            for j in range(self.config.GRID_N):

                lobule = self._create_lobule(
                    n_sources=self.config.SOURCES_NR,
                    branch_prob=self.config.BRANCH_PROB,
                )

                x_start = i * self.lobule_size
                x_end = x_start + self.lobule_size
                y_start = j * self.lobule_size
                y_end = y_start + self.lobule_size

                lat_model[x_start:x_end, y_start:y_end] = lobule
        return lat_model

    def _create_lobule(self, n_sources=8, branch_prob=0.15):
        grid = np.zeros(
            (self.lobule_size, self.lobule_size), dtype=int
        )  # 0 = hepatocyte, 1 = sinusoid
        center = (self.lobule_size // 2, self.lobule_size // 2)

        # Create evenly spaced sinusoids sources along edges
        sources = set()
        edge_positions = np.linspace(0, self.lobule_size - 1, n_sources, dtype=int)

        for i in edge_positions:
            sources.add((0, i))  # top
            sources.add((self.lobule_size - 1, i))  # bottom
            sources.add((i, 0))  # left
            sources.add((i, self.lobule_size - 1))  # right

        for s in sources:
            s_coord = s

            while s_coord != center:
                x, y = s_coord
                grid[x, y] = 1

                # branching
                if np.random.rand() < branch_prob:
                    branch = self._step_toward(s_coord, center)
                    bx, by = branch
                    if 0 <= bx < self.lobule_size and 0 <= by < self.lobule_size:
                        grid[bx, by] = 1

                # move toward central vein
                s_coord = self._step_toward(s_coord, center)

                x, y = s_coord
                if not (0 <= x < self.lobule_size and 0 <= y < self.lobule_size):
                    break

        grid[center] = 2  # central vein
        return grid

    def _step_toward(self, coordinate, target, randomness=0.5):
        x, y = coordinate
        tx, ty = target

        dx = np.sign(tx - x) if np.random.rand() > randomness else 0
        dy = np.sign(ty - y) if np.random.rand() > randomness else 0

        # add randomness (branching / irregularity)
        if np.random.rand() < randomness:
            if np.random.rand() < 0.5:
                dx = 0
            else:
                dy = 0

        if dx == 0 and dy == 0:
            dx = np.sign(tx - x)
            dy = np.sign(ty - y)

        return (x + dx, y + dy)

    def _compute_distance(self, grid):
        # distance from hepatocytes (0) to nearest sinusoid (1)
        return distance_transform_edt(grid == 0)

    def visualize_lattice(self, compute_distance=False):
        """Visualize the liver lobule lattice using Matplotlib. Optionally compute and display the distance from hepatocytes to the nearest sinusoid."""

        plt.figure(figsize=(10, 6))

        if compute_distance:
            plt.subplot(1, 2, 1)

        plt.imshow(self.lattice, cmap="viridis")
        for k in range(0, self.lattice_size + 1, self.lobule_size):
            plt.axhline(k - 0.5, color="white", linewidth=0.5)
            plt.axvline(k - 0.5, color="white", linewidth=0.5)
        plt.title("Liver Lobule Lattice")
        plt.colorbar(label="0 = Hepatocyte, 1 = Sinusoid, 2 = Central Vein")
        plt.axis("off")

        if compute_distance:
            distance = self._compute_distance(self.lattice)
            plt.subplot(1, 2, 2)
            plt.title("Distance to Nearest Sinusoid")
            plt.imshow(distance)
            for k in range(0, self.lattice_size + 1, self.lobule_size):
                plt.axhline(k - 0.5, color="white", linewidth=0.5)
                plt.axvline(k - 0.5, color="white", linewidth=0.5)
            plt.colorbar()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    config = Config()
    lattice = LiverLobuleLattice(config)
    lattice.visualize_lattice(compute_distance=True)
