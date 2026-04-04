import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from config import Config

config = Config()

# ══════════════════════════════════════════════════════════════════════════════
# ── LobuleQuadrant ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


class LobuleQuadrant:
    """
    One symmetry element of a liver lobule — a square lattice of hepatocytes
    interlaced with sinusoids. Portal triad (inlet) at one corner, central
    vein contribution (outlet) at the opposite corner.

    Parameters
    ----------
    direction : str
        One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    grid_size : int
        Number of lattice cells per side (default config.GRID_N).
    """

    DIRS = ["top-left", "top-right", "bottom-left", "bottom-right"]

    def __init__(self, direction: str, grid_size: int = config.GRID_N):
        if direction not in self.DIRS:
            raise ValueError(f"direction must be one of {self.DIRS}")

        self.direction = direction
        self.grid_size = grid_size
        self.physio_grid = self._build_struc_matrix()

        self.grid_dim = self.physio_grid.shape[0]
        self.inlet_pos = self._get_corner("inlet", self.grid_dim)
        self.outlet_pos = self._get_corner("outlet", self.grid_dim)

        self.P, self.vx, self.vy = self._compute_darcy()
        self.C = self._init_concentration()

        self.total_mass_history = []
        self.inlet_concentration_history = []
        self.outlet_concentration_history = []
        self.time_history = []

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _cell_sizes(self):
        sizes = []
        for i in range(self.grid_size):
            if i % 2 == 0:
                sizes.append(1 if i in (0, self.grid_size - 1) else config.SIN_SIZE)
            else:
                sizes.append(config.HEPA_SIZE)
        return sizes

    def _build_struc_matrix(self):
        lattice = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(0, self.grid_size, 2):
            lattice[i, :] = 1
            lattice[:, i] = 1
        sizes = self._cell_sizes()
        expanded = np.repeat(lattice, sizes, axis=0)
        expanded = np.repeat(expanded, sizes, axis=1)
        return expanded

    def _get_corner(self, role: str, n: int):
        corners = {
            "top-left": {"inlet": (0, 0), "outlet": (n - 1, n - 1)},
            "top-right": {"inlet": (0, n - 1), "outlet": (n - 1, 0)},
            "bottom-left": {"inlet": (n - 1, 0), "outlet": (0, n - 1)},
            "bottom-right": {"inlet": (n - 1, n - 1), "outlet": (0, 0)},
        }
        return corners[self.direction][role]

    # ── Darcy flow ────────────────────────────────────────────────────────────

    def _compute_darcy(self):
        A = lil_matrix((self.grid_dim * self.grid_dim, self.grid_dim * self.grid_dim))
        b = np.zeros(self.grid_dim * self.grid_dim)
        K_2d = np.where(self.physio_grid == 1, config.K_SIN, config.K_HEPA)
        K = K_2d.flatten()

        def idx(r, c):
            return r * self.grid_dim + c

        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                i = idx(r, c)
                if (r, c) == self.inlet_pos:
                    A[i, i] = 1
                    b[i] = config.P_INLET
                    continue
                if (r, c) == self.outlet_pos:
                    A[i, i] = 1
                    b[i] = config.P_OUTLET
                    continue

                neighbors = []
                if r > 0:
                    neighbors.append(idx(r - 1, c))
                if r < self.grid_dim - 1:
                    neighbors.append(idx(r + 1, c))
                if c > 0:
                    neighbors.append(idx(r, c - 1))
                if c < self.grid_dim - 1:
                    neighbors.append(idx(r, c + 1))

                total_conductance = 0
                for ni in neighbors:
                    k = 2.0 * K[i] * K[ni] / (K[i] + K[ni])
                    if ni in (idx(*self.inlet_pos), idx(*self.outlet_pos)):
                        k = config.K_SIN
                    A[i, ni] = k
                    total_conductance += k
                A[i, i] = -total_conductance

        P = spsolve(A.tocsr(), b).reshape((self.grid_dim, self.grid_dim))
        spacing = config.LOBULE_SIZE / self.grid_dim
        grad_y, grad_x = np.gradient(P, spacing, spacing)
        vx = -(K_2d / config.BLOOD_VISCOSITY) * grad_x
        vy = -(K_2d / config.BLOOD_VISCOSITY) * grad_y
        return P, vx, vy

    # ── Convective flux ────────────────────────────────────────────────────────
    def check_cfl(self):
        """Check CFL condition for explicit convection update."""
        n = self.C.shape[0]
        dx = config.LOBULE_SIZE / n
        max_v = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        cfl = max_v * config.DT / dx
        if cfl > 1.0:
            raise RuntimeError(
                f"CFL = {cfl:.3f} > 1. Suggested DT < {dx / max_v:.2e} s"
            )

    def compute_flux(self):
        """Compute one time step of concentration update due to convection on the unified grid."""
        C = self.C.copy()
        self.check_cfl()
        dx = dy = config.LOBULE_SIZE / C.shape[0]

        # ── Neighbour concentrations (zero padding = outflow BC) ──────────────────
        C_pad = np.pad(C, 1, mode="constant", constant_values=0)
        C_L = C_pad[1:-1, :-2]
        C_R = C_pad[1:-1, 2:]
        C_U = C_pad[:-2, 1:-1]
        C_D = C_pad[2:, 1:-1]

        # ── Face velocities (no averaging, cell-own velocity) ─────────────────────
        VX_pad = np.pad(self.vx, 1, mode="edge")
        VY_pad = np.pad(self.vy, 1, mode="edge")
        VX_L = self.vx
        VX_R = VX_pad[1:-1, 2:]
        VY_T = self.vy
        VY_B = VY_pad[2:, 1:-1]

        # ── Upwind fluxes J = C·v ─────────────────────────────────────────────────
        F_L = np.where(VX_L > 0, VX_L * C_L, VX_L * C)
        F_R = np.where(VX_R > 0, VX_R * C, VX_R * C_R)
        G_T = np.where(VY_T > 0, VY_T * C_U, VY_T * C)
        G_B = np.where(VY_B > 0, VY_B * C, VY_B * C_D)

        adv = (F_L - F_R) / dx + (G_T - G_B) / dy
        r, c = self.inlet_pos
        print(
            f"{self.direction} inlet {self.inlet_pos}: "
            f"C={C[r,c]:.2f} "
            f"F_L={F_L[r,c]:.4e} F_R={F_R[r,c]:.4e} "
            f"G_T={G_T[r,c]:.4e} G_B={G_B[r,c]:.4e} "
            f"adv={adv[r,c]:.4e}"
        )
        C_new = C + config.DT * adv
        C_new = np.maximum(C_new, 0.0)  # no negative concentrations

        C_new[self.outlet_pos] = 0.0  # enforce zero concentration at outlet
        self.C = C_new

        return C_new

    # ── Concentration ─────────────────────────────────────────────────────────

    def _init_concentration(self):
        C = np.zeros(self.physio_grid.shape)
        C[self.inlet_pos] = config.INLET_CONC
        return C

    def reset_concentration(self):
        self.C = self._init_concentration()

    # ── Tracking ──────────────────────────────────────────────────────────────

    def record(self):
        self.total_mass_history.append(np.sum(self.C))
        self.inlet_concentration_history.append(self.C[self.inlet_pos])
        self.outlet_concentration_history.append(self.C[self.outlet_pos])
        self.time_history.append(len(self.time_history) * config.DT)
