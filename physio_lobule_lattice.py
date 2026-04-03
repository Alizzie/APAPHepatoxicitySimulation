import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import Config
from random import seed

seed(42)
config = Config()


class PhysiologicalLobuleModel:
    """
    Class representing a regular lattice of squared hepatocytes and sinusoids organized into squared liver lobules.
    Model based on Rezania et al. 2013, which simulates drug diffusion and metabolism in a regular grid of lobules.

    Parameters
    ----------
    direction : str or None
        If a direction string is given (e.g. "top-left"), only that quadrant
        is plotted.  If None, all four quadrants are assembled into a single
        full lobule and plotted together.

    Notes
    -----
    Physical pixel sizes are read from config:
        config.SINUSOID_SIZE  – side length of a sinusoid cell  [µm]
        config.HEPATOCYTE_SIZE – side length of a hepatocyte cell [µm]

    The lattice encodes cell type as:
        1  →  sinusoid   (every even row/column)
        0  →  hepatocyte (every odd row/column)
    """

    def __init__(
        self,
        grid_size: int = config.GRID_N,
        lat_dir: str = None,
    ):

        self.lattice_size = grid_size
        self.lattice_dirs = (
            ["top-left", "top-right", "bottom-left", "bottom-right"]
            if lat_dir is None
            else [lat_dir]
        )
        self.lobule = self._build_lobule()

    def _build_lobule(self):
        """
        Create a lobule based on four equivalent lattice models, with different portal triad positions.
        """

        lobule_lattices = {}

        for dir in self.lattice_dirs:
            inlet_pos = self._get_inlet_position(dir)
            outlet_pos = self._get_outlet_position(dir)
            lattice = self._build_lobule_quadrant()

            lobule_lattices[dir] = {
                "lattice": lattice,
                "inlet": inlet_pos,
                "outlet": outlet_pos,
            }

        return lobule_lattices

    def _build_lobule_quadrant(self):
        """
        Build one quadrant of the lobule, with a sinusoid path from the inlet (portal triad) to the outlet (central vein).
        It's a grid of hepatocyctes with sinusoids bordering each cell.

        The lattice is a 51x51 grid, with each cell representing a hepatocyte (0) or a sinusoid (1). The inlet is at one corner, and the outlet is at the opposite corner.
        """

        lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=int)
        for i in range(0, self.lattice_size, 2):
            lattice[i, :] = 1
            lattice[:, i] = 1

        return lattice

    # -----------------------------------------------------------------------------------
    # ── Lattice Construction Helpers ───────────────────────────────────────────────────────
    # -----------------------------------------------------------------------------------
    def _get_inlet_position(self, direction):
        if direction == "top-left":
            return (0, 0)
        elif direction == "top-right":
            return (0, self.lattice_size - 1)
        elif direction == "bottom-left":
            return (self.lattice_size - 1, 0)
        elif direction == "bottom-right":
            return (self.lattice_size - 1, self.lattice_size - 1)
        else:
            raise ValueError("Invalid direction")

    def _get_outlet_position(self, direction):
        if direction == "top-left":
            return (self.lattice_size - 1, self.lattice_size - 1)
        elif direction == "top-right":
            return (self.lattice_size - 1, 0)
        elif direction == "bottom-left":
            return (0, self.lattice_size - 1)
        elif direction == "bottom-right":
            return (0, 0)
        else:
            raise ValueError("Invalid direction")

    # -----------------------------------------------------------------------------------
    # --- Visualization Methods ─────────────────────────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------------------
    def visualize_lobule(self, direction=None):
        """Visualize the lobule lattice, sinusoids in red and hepatocytes in blue."""

        n = self.lattice_size
        cmap = ListedColormap(["#1f77b4", "#d62728"])

        def _cell_sizes(n):
            """Pixel size for each row/col index along one axis."""
            sizes = []
            for i in range(n):
                if i % 2 == 0:  # sinusoid
                    sizes.append(1 if i in (0, n - 1) else config.SIN_SIZE)
                else:  # hepatocyte
                    sizes.append(config.HEPA_SIZE)
            return sizes

        def _build_struc_matrix(lattice):
            """
            Expand the (n, n) lattice into a physically-sized 2D matrix
            by repeating each row and column according to its cell size.
            Sinusoid = 1, Hepatocyte = 0.
            """
            sizes = _cell_sizes(n)
            # repeat each row, then each column
            expanded = np.repeat(lattice, sizes, axis=0)
            expanded = np.repeat(expanded, sizes, axis=1)
            return expanded

        def _plot_markers(
            ax, inlet, outlet, sizes, row_offset=0, col_offset=0, show_legend=True
        ):
            """
            Add portal triad and central vein markers to the plot.

            Parameters
            ----------
            ax          : matplotlib Axes
            inlet       : (row, col) lattice index of the portal triad
            outlet      : (row, col) or None — lattice index of the central vein
            sizes       : list of pixel sizes per lattice index (from _cell_sizes)
            row_offset  : pixel offset added to y (for assembling full lobule)
            col_offset  : pixel offset added to x (for assembling full lobule)
            show_legend : whether to add legend labels (set False for duplicate quadrants)
            """
            edges = np.concatenate(([0], np.cumsum(sizes)))

            def _to_pixel(lattice_pos):
                """Centre pixel of a cell, accounting for mixed sinusoid/hepatocyte sizes."""
                px = (
                    edges[lattice_pos[1]] + edges[lattice_pos[1] + 1]
                ) / 2 + col_offset
                py = (
                    edges[lattice_pos[0]] + edges[lattice_pos[0] + 1]
                ) / 2 + row_offset
                return px, py

            markers = [(inlet, "Portal Triad", config.PT_COL, "o")]
            if outlet is not None:
                markers.append((outlet, "Central Vein", config.CV_COL, "o"))

            for pos, label, color, marker in markers:
                px, py = _to_pixel(pos)
                ax.scatter(
                    px,
                    py,
                    s=150,
                    c=color,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.8,
                    label=label if show_legend else "",
                    zorder=5,
                )

            if show_legend:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # ── single quadrant ──────────────────────────────────────────────────
        if direction is not None:
            mat = _build_struc_matrix(self.lobule[direction]["lattice"])
            fig, ax = plt.subplots(figsize=(5, 5))

            _plot_markers(
                ax,
                self.lobule[direction]["inlet"],
                self.lobule[direction]["outlet"],
                _cell_sizes(self.lattice_size),
            )
            ax.imshow(mat, origin="upper", interpolation="nearest", cmap=cmap)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # ── full lobule ───────────────────────────────────────────────────────
        # Build each quadrant matrix, then mirror and join.
        # Border sinusoids (1px) from adjacent quadrants add up to SIN_SIZE at joins.
        sizes = _cell_sizes(n)
        Q = sum(sizes)  # quadrant side in pixels

        mat_tl = _build_struc_matrix(self.lobule["top-left"]["lattice"])
        mat_tr = np.fliplr(_build_struc_matrix(self.lobule["top-right"]["lattice"]))
        mat_bl = np.flipud(_build_struc_matrix(self.lobule["bottom-left"]["lattice"]))
        mat_br = np.fliplr(
            np.flipud(_build_struc_matrix(self.lobule["bottom-right"]["lattice"]))
        )

        full = np.block(
            [
                [mat_tl, mat_tr],
                [mat_bl, mat_br],
            ]
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(
            full, origin="upper", interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )
        ax.axis("off")

        # portal triads: reuse _plot_markers with a pixel offset per quadrant
        offsets = {
            "top-left": (0, 0),
            "top-right": (0, Q),
            "bottom-left": (Q, 0),
            "bottom-right": (Q, Q),
        }
        for i, (dir_key, (row_off, col_off)) in enumerate(offsets.items()):
            quad = self.lobule[dir_key]
            _plot_markers(
                ax,
                inlet=quad["inlet"],
                outlet=None,  # outlet handled separately below
                sizes=sizes,
                row_offset=row_off,
                col_offset=col_off,
                show_legend=(i == 0),  # only add legend entry once
            )

        # central vein: single shared outlet at the pixel centre of the full lobule
        ax.scatter(
            Q,
            Q,
            s=200,
            c=config.CV_COL,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            label="Central Vein",
            zorder=5,
        )

        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        plt.tight_layout()
        plt.show()


# ---──────────────────────────────────────────────────────────────────────────────────────────────
# -- MAIN EXECUTION ───────────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = PhysiologicalLobuleModel()
    # for dir in model.lobule:
    #     model.visualize_lobule(dir)
    model.visualize_lobule("top-left")  # single quadrant
    model.visualize_lobule()  # full lobule with all quadrants
