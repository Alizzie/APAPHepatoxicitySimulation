import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from config import Config
from random import seed

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

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

        self.lobule, self.physio_grid = self._build_lobule()
        self.pressure_fields, self.velocity_fields = self._compute_darcy_advection(
            self.lattice_dirs
        )

        self.drug_conc = self._init_drug_concentration()

        # Tracking parameters
        self.total_mass_history = {d: [] for d in self.lattice_dirs}
        self.avg_cell_concentration = {d: [] for d in self.lattice_dirs}
        self.inlet_concentration_history = {d: [] for d in self.lattice_dirs}
        self.outlet_concentration_history = {d: [] for d in self.lattice_dirs}
        self.time_history = {d: [] for d in self.lattice_dirs}

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

        return lobule_lattices, self._build_struc_matrix(
            lobule_lattices[self.lattice_dirs[0]]["lattice"]
        )

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
    def _get_inlet_position(self, direction, lattice_size=None):
        if lattice_size is None:
            n = self.lattice_size
        else:
            n = lattice_size

        if direction == "top-left":
            return (0, 0)
        elif direction == "top-right":
            return (0, n - 1)
        elif direction == "bottom-left":
            return (n - 1, 0)
        elif direction == "bottom-right":
            return (n - 1, n - 1)
        else:
            raise ValueError("Invalid direction")

    def _get_outlet_position(self, direction, lattice_size=None):
        if lattice_size is None:
            n = self.lattice_size
        else:
            n = lattice_size

        if direction == "top-left":
            return (n - 1, n - 1)
        elif direction == "top-right":
            return (n - 1, 0)
        elif direction == "bottom-left":
            return (0, n - 1)
        elif direction == "bottom-right":
            return (0, 0)
        else:
            raise ValueError("Invalid direction")

    def _cell_sizes(self, n):
        """Pixel size for each row/col index along one axis."""
        sizes = []
        for i in range(n):
            if i % 2 == 0:  # sinusoid
                sizes.append(1 if i in (0, n - 1) else config.SIN_SIZE)
            else:  # hepatocyte
                sizes.append(config.HEPA_SIZE)
        return sizes

    def _build_struc_matrix(self, lattice):
        """
        Expand the (n, n) lattice into a physically-sized 2D matrix
        by repeating each row and column according to its cell size.
        Sinusoid = 1, Hepatocyte = 0.
        """
        sizes = self._cell_sizes(self.lattice_size)
        expanded = np.repeat(lattice, sizes, axis=0)
        expanded = np.repeat(expanded, sizes, axis=1)
        return expanded

    # -----------------------------------------------------------------------------------
    # --- Pressure AND FLOW CALCULATIONS ─────────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------------------

    def _compute_darcy_advection(self, directions=None):
        """
        Calculates the steady-state pressure field and velocity vectors using Darcy's Law.
        Solves only for top-left, then mirrors to get the other three quadrants.
        """
        pressure_fields = {}
        velocity_fields = {}

        # ── Solve only for top-left ───────────────────────────────────────────────
        direction = "top-left"
        inlet_pos = self._get_inlet_position(
            direction, lattice_size=self.physio_grid.shape[0]
        )
        outlet_pos = self._get_outlet_position(
            direction, lattice_size=self.physio_grid.shape[0]
        )

        n = self.physio_grid.shape[0]
        N = n * n
        A = lil_matrix((N, N))
        b = np.zeros(N)

        K_2d = np.where(self.physio_grid == 1, config.K_SIN, config.K_HEPA)
        K = K_2d.flatten()

        def get_idx(r, c):
            return r * n + c

        for r in range(n):
            for c in range(n):
                i = get_idx(r, c)

                if (r, c) == inlet_pos:
                    A[i, i] = 1
                    b[i] = config.P_INLET
                    continue
                if (r, c) == outlet_pos:
                    A[i, i] = 1
                    b[i] = config.P_OUTLET
                    continue

                neighbors = []
                if r > 0:
                    neighbors.append(get_idx(r - 1, c))
                if r < n - 1:
                    neighbors.append(get_idx(r + 1, c))
                if c > 0:
                    neighbors.append(get_idx(r, c - 1))
                if c < n - 1:
                    neighbors.append(get_idx(r, c + 1))

                total_conductance = 0
                for ni in neighbors:
                    k_avg = 2.0 * K[i] * K[ni] / (K[i] + K[ni])
                    if ni == get_idx(inlet_pos[0], inlet_pos[1]) or ni == get_idx(
                        outlet_pos[0], outlet_pos[1]
                    ):
                        k_avg = config.K_SIN
                    A[i, ni] = k_avg
                    total_conductance += k_avg

                A[i, i] = -total_conductance
                b[i] = 0

        P_tl = spsolve(A.tocsr(), b).reshape((n, n))

        spacing = config.LOBULE_SIZE / n
        grad_y, grad_x = np.gradient(P_tl, spacing)

        vx_tl = -(K_2d / config.BLOOD_VISCOSITY) * grad_x
        vy_tl = -(K_2d / config.BLOOD_VISCOSITY) * grad_y

        pressure_fields["top-left"] = P_tl
        velocity_fields["top-left"] = (vx_tl, vy_tl)

        # ── Mirror to other three quadrants ───────────────────────────────────────
        # top-right:    flip horizontally → vx reverses sign, vy stays
        pressure_fields["top-right"] = np.fliplr(P_tl)
        velocity_fields["top-right"] = (-np.fliplr(vx_tl), np.fliplr(vy_tl))

        # bottom-left:  flip vertically → vy reverses sign, vx stays
        pressure_fields["bottom-left"] = np.flipud(P_tl)
        velocity_fields["bottom-left"] = (np.flipud(vx_tl), -np.flipud(vy_tl))

        # bottom-right: flip both → both vx and vy reverse sign
        pressure_fields["bottom-right"] = np.fliplr(np.flipud(P_tl))
        velocity_fields["bottom-right"] = (
            -np.fliplr(np.flipud(vx_tl)),
            -np.flipud(np.fliplr(vy_tl)),
        )

        # ── Only return requested directions ─────────────────────────────────────
        pressure_fields = {d: pressure_fields[d] for d in directions}
        velocity_fields = {d: velocity_fields[d] for d in directions}

        return pressure_fields, velocity_fields

    # --------------------------------------------------------------------------------
    # --- Diffusion and Metabolism Calculations ─────────────────────────────────────────────────────────────
    # --------------------------------------------------------------------------------
    def _init_drug_concentration(self):
        """
        Initialize drug concentration matrix for each quadrant, with inlet concentration at the portal triad and zero elsewhere.
        """
        drug_conc = {}
        for direction in self.lattice_dirs:
            C = np.zeros(self.physio_grid.shape)
            drug_conc[direction] = C

            C[
                self._get_inlet_position(
                    direction, lattice_size=self.physio_grid.shape[0]
                )
            ] = config.INLET_CONC

        return drug_conc

    def compute_convective_flux(self, direction=None):
        """
        Advance drug concentration by one timestep using a first-order upwind
        scheme for advection (J = C·v).

        If direction is None, all four quadrants are assembled into a single
        2N x 2N matrix and updated in one unified pass — guaranteeing perfect
        symmetry and correct mass conservation with no ghost cell bookkeeping.

        Boundary conditions:
            - Exterior edges        : outflow only (zero padding)
            - Central vein (outlet) : single pixel sink at the quadrant junction
        """
        is_full = direction is None

        if is_full:
            C = np.block(
                [
                    [self.drug_conc["top-left"], self.drug_conc["top-right"]],
                    [self.drug_conc["bottom-left"], self.drug_conc["bottom-right"]],
                ]
            )
            VX = np.block(
                [
                    [
                        self.velocity_fields["top-left"][0],
                        self.velocity_fields["top-right"][0],
                    ],
                    [
                        self.velocity_fields["bottom-left"][0],
                        self.velocity_fields["bottom-right"][0],
                    ],
                ]
            )
            VY = np.block(
                [
                    [
                        self.velocity_fields["top-left"][1],
                        self.velocity_fields["top-right"][1],
                    ],
                    [
                        self.velocity_fields["bottom-left"][1],
                        self.velocity_fields["bottom-right"][1],
                    ],
                ]
            )
            n_single = C.shape[0] // 2
        else:
            C = self.drug_conc[direction]
            VX, VY = self.velocity_fields[direction]
            n_single = C.shape[0]

        dx = dy = config.LOBULE_SIZE / n_single

        # ── CFL guard (once, using assembled field) ───────────────────────────────
        max_v = max(np.max(np.abs(VX)), np.max(np.abs(VY)))
        cfl = max_v * config.DT / dx
        if cfl > 1.0:
            raise RuntimeError(
                f"CFL = {cfl:.3f} > 1 — reduce DT. Suggested DT < {dx / max_v:.2e} s"
            )

        # ── Neighbour concentrations (zero padding = outflow BC) ──────────────────
        C_pad = np.pad(C, 1, mode="constant", constant_values=0)
        C_L = C_pad[1:-1, :-2]
        C_R = C_pad[1:-1, 2:]
        C_U = C_pad[:-2, 1:-1]
        C_D = C_pad[2:, 1:-1]

        # ── Face velocities (no averaging, cell-own velocity) ─────────────────────
        VX_pad = np.pad(VX, 1, mode="edge")
        VY_pad = np.pad(VY, 1, mode="edge")
        VX_L = VX
        VX_R = VX_pad[1:-1, 2:]
        VY_T = VY
        VY_B = VY_pad[2:, 1:-1]

        # ── Upwind fluxes J = C·v ─────────────────────────────────────────────────
        F_L = np.where(VX_L > 0, VX_L * C_L, VX_L * C)
        F_R = np.where(VX_R > 0, VX_R * C, VX_R * C_R)
        G_T = np.where(VY_T > 0, VY_T * C_U, VY_T * C)
        G_B = np.where(VY_B > 0, VY_B * C, VY_B * C_D)

        adv = (F_L - F_R) / dx + (G_T - G_B) / dy
        C_new = C + config.DT * adv
        C_new = np.maximum(C_new, 0.0)  # no negative concentrations

        # ── Boundary conditions ───────────────────────────────────────────────────
        if is_full:
            q = C.shape[0] // 2
            C_new[q, q] = 0.0  # single pixel central vein outlet

            self.drug_conc["top-left"] = C_new[:q, :q]
            self.drug_conc["top-right"] = C_new[:q, q:]
            self.drug_conc["bottom-left"] = C_new[q:, :q]
            self.drug_conc["bottom-right"] = C_new[q:, q:]
        else:
            outlet = self._get_outlet_position(direction, lattice_size=C.shape[0])
            C_new[outlet] = 0.0
            self.drug_conc[direction] = C_new

        # ── Tracking ──────────────────────────────────────────────────────────────
        if is_full:
            for d in self.lattice_dirs:
                C_d = self.drug_conc[d]
                n_d = C_d.shape[0]
                inlet = self._get_inlet_position(d, lattice_size=n_d)
                outlet = self._get_outlet_position(d, lattice_size=n_d)
                self.total_mass_history[d].append(np.sum(C_d))
                self.avg_cell_concentration[d].append(np.mean(C_d))
                self.inlet_concentration_history[d].append(C_d[inlet])
                self.outlet_concentration_history[d].append(C_d[outlet])
                self.time_history[d].append(len(self.total_mass_history[d]) * config.DT)
        else:
            n_d = C_new.shape[0]
            inlet = self._get_inlet_position(direction, lattice_size=n_d)
            outlet = self._get_outlet_position(direction, lattice_size=n_d)
            self.total_mass_history[direction].append(np.sum(C_new))
            self.avg_cell_concentration[direction].append(np.mean(C_new))
            self.inlet_concentration_history[direction].append(C_new[inlet])
            self.outlet_concentration_history[direction].append(C_new[outlet])
            self.time_history[direction].append(
                len(self.total_mass_history[direction]) * config.DT
            )

        return C_new

    # -----------------------------------------------------------------------------------
    # --- Visualization Methods ─────────────────────────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------------------
    def visualize_lobule(self, direction=None):
        """Visualize the lobule lattice, sinusoids in red and hepatocytes in blue."""

        n = self.lattice_size
        cmap = ListedColormap(["#1f77b4", "#d62728"])

        def _plot_markers(
            ax, inlet, outlet, sizes, row_offset=0, col_offset=0, show_legend=True
        ):
            """
            Add portal triad and central vein markers to the plot.
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
            mat = self.physio_grid
            fig, ax = plt.subplots(figsize=(5, 5))

            _plot_markers(
                ax,
                self.lobule[direction]["inlet"],
                self.lobule[direction]["outlet"],
                self._cell_sizes(self.lattice_size),
            )
            ax.imshow(mat, origin="upper", interpolation="nearest", cmap=cmap)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # ── full lobule ───────────────────────────────────────────────────────
        # Build each quadrant matrix, then mirror and join.
        # Border sinusoids (1px) from adjacent quadrants add up to SIN_SIZE at joins.
        sizes = self._cell_sizes(self.lattice_size)
        Q = sum(sizes)  # quadrant side in pixels

        mat_tl = self.physio_grid
        mat_tr = np.fliplr(self.physio_grid)
        mat_bl = np.flipud(self.physio_grid)
        mat_br = np.fliplr(np.flipud(self.physio_grid))

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

    # -----------------------------------------------------------------------------------
    # --- VERIFYING METHODS ─────────────────────────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------------------

    def visualize_advection(self, direction=None):
        """
        Visualize pressure and velocity for one quadrant or the full lobule.
        If direction is None, all four quadrants are assembled into a single full lobule view.
        """
        if direction is not None:
            P = self.pressure_fields[direction]
            vx, vy = self.velocity_fields[direction]
            mag = np.sqrt(vx**2 + vy**2) * 6000
        else:
            p_tl = self.pressure_fields["top-left"]
            p_tr = self.pressure_fields["top-right"]
            p_bl = self.pressure_fields["bottom-left"]
            p_br = self.pressure_fields["bottom-right"]

            P = np.block(
                [
                    [p_tl, p_tr],
                    [p_bl, p_br],
                ]
            )

            vel_tl = np.sqrt(
                self.velocity_fields["top-left"][0] ** 2
                + self.velocity_fields["top-left"][1] ** 2
            )
            vel_tr = np.sqrt(
                self.velocity_fields["top-right"][0] ** 2
                + self.velocity_fields["top-right"][1] ** 2
            )
            vel_bl = np.sqrt(
                self.velocity_fields["bottom-left"][0] ** 2
                + self.velocity_fields["bottom-left"][1] ** 2
            )
            vel_br = np.sqrt(
                self.velocity_fields["bottom-right"][0] ** 2
                + self.velocity_fields["bottom-right"][1] ** 2
            )

            mag = (
                np.block(
                    [
                        [vel_tl, vel_tr],
                        [vel_bl, vel_br],
                    ]
                )
                * 6000
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(P / 1000, cmap="viridis", origin="upper")
        axes[0].set_title("Pressure Field (kPa)")
        plt.colorbar(im0, ax=axes[0], label="kPa")

        im1 = axes[1].imshow(
            mag + 1e-20,
            cmap="jet_r",
            origin="upper",
            norm=mcolors.LogNorm(),
        )
        axes[1].set_title("Velocity Magnitude (cm/min)")
        plt.colorbar(im1, ax=axes[1], label="cm/min")
        plt.tight_layout()
        plt.show()

        # show quiver plot of velocity vectors
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(P / 1000, cmap="viridis", origin="upper")
        axes[0].set_title("Pressure Field (kPa)")

        quiver_skip = 10  # plot velocity vectors every N pixels for clarity
        offsets = {
            "top-left": (0, 0),
            "top-right": (0, P.shape[1] // 2),
            "bottom-left": (P.shape[0] // 2, 0),
            "bottom-right": (P.shape[0] // 2, P.shape[1] // 2),
        }
        if direction is not None:
            vx, vy = self.velocity_fields[direction]
            axes[1].quiver(
                np.arange(0, vx.shape[1], quiver_skip),
                np.arange(0, vx.shape[0], quiver_skip),
                vx[::quiver_skip, ::quiver_skip],
                vy[::quiver_skip, ::quiver_skip],
                color="red",
                scale=50,
                width=0.002,
            )
        else:
            for dir_key in self.lattice_dirs:
                vx, vy = self.velocity_fields[dir_key]
                row_off, col_off = offsets[dir_key]
                axes[1].quiver(
                    np.arange(0, vx.shape[1], quiver_skip) + col_off,
                    np.arange(0, vx.shape[0], quiver_skip) + row_off,
                    vx[::quiver_skip, ::quiver_skip],
                    vy[::quiver_skip, ::quiver_skip],
                    color="red",
                    scale=50,
                    width=0.002,
                )

        plt.tight_layout()
        plt.show()

    # ---──────────────────────────────────────────────────────────────────────────────────────────────
    # -- MAIN EXECUTION ───────────────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────────────────────────────


def simulate_advection(model, steps=1000):
    # Check CFL
    n = model.drug_conc["top-left"].shape[0]
    vx, vy = model.velocity_fields["top-left"]
    dx = 750e-6 / n
    max_v = max(np.max(np.abs(vx)), np.max(np.abs(vy)))
    DT_optimal = 0.9 * dx / max_v
    print(f"CFL = {max_v * config.DT / dx:.4f}")
    print(f"Optimal DT: {DT_optimal:.4e}")

    for step in range(steps):
        print(f"Time step {step+1}/{steps}")

        C = model.compute_convective_flux()

        if step % 5000 == 0 or step == 0:
            print(f"  Total drug mass: {np.sum(C):.4e} mM")
            for d in model.lattice_dirs:
                print(
                    f"  {d}: inlet={model.inlet_concentration_history[d][-1]:.4e} mM, "
                    f"outlet={model.outlet_concentration_history[d][-1]:.4e} mM"
                )

            lattice_grid = model.physio_grid
            if C.shape[0] != lattice_grid.shape[0]:
                grid = np.repeat(lattice_grid, 2, axis=0)
                grid = np.repeat(grid, 2, axis=1)
            else:
                grid = lattice_grid

            C_sin = np.where(grid == 1, C, 0)  # zero out hepatocyte concentrations

            C_hepa = np.where(grid == 0, C, 0)  # zero out sinusoid concentrations

            fig, ax = plt.subplots(1, 3, figsize=(12, 5))
            im0 = ax[0].imshow(
                C_sin,
                cmap="Reds",
                origin="upper",
            )
            ax[0].set_title("Drug Concentration in Sinusoids")
            plt.colorbar(im0, ax=ax[0], label="Concentration (mM)")

            im1 = ax[1].imshow(
                C_hepa,
                cmap="Blues",
                origin="upper",
            )

            im3 = ax[2].imshow(
                C,
                cmap="viridis",
                origin="upper",
            )
            ax[2].set_title("Total Drug Concentration")
            plt.colorbar(im3, ax=ax[2], label="Concentration (mM)")

            ax[1].set_title("Drug Concentration in Hepatocytes")
            plt.colorbar(im1, ax=ax[1], label="Concentration (mM)")
            plt.tight_layout()
            fig.suptitle(f"Drug Concentration at t = {step * config.DT:.3f}s")
            plt.show()


if __name__ == "__main__":
    model = PhysiologicalLobuleModel()
    # for dir in model.lobule:
    #     model.visualize_lobule(dir)
    # model.visualize_lobule("top-left")  # single quadrant
    # model.visualize_lobule()  # full lobule with all quadrants
    # model.visualize_advection()

    # visualize all four velocity fields 2x2
    simulate_advection(model, steps=20000)
