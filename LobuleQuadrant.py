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
        max_vx = np.max(np.abs(self.vx))
        max_vy = np.max(np.abs(self.vy))
        cfl = max(max_vx, max_vy) * config.DT / dx
        if cfl > 1.0:
            raise RuntimeError(
                f"CFL = {cfl:.3f} > 1. Suggested DT < {dx / max(max_vx, max_vy):.2e} s"
            )

        D_max = config.D_SIN
        diff_cfl = 2 * D_max * config.DT / dx**2
        if diff_cfl > 1.0:
            raise RuntimeError(
                f"Diffusive CFL = {diff_cfl:.3f} > 1. Suggested DT < {dx**2 / (2*D_max):.2e} s"
            )

    def compute_flux(self):
        """
        Operator splitting:
        1. Advection  — explicit upwind, subcycled to satisfy CFL
        2. Diffusion  — implicit (ADI), unconditionally stable
        """
        C = self.C.copy()
        dx = dy = config.LOBULE_SIZE / C.shape[0]
        n = C.shape[0]

        # ── Diffusion coefficient field ───────────────────────────────────────
        D = np.where(self.physio_grid == 1, config.D_SIN, config.D_HEPA)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 — Advection (explicit upwind, subcycled)
        # ══════════════════════════════════════════════════════════════════════
        max_v = np.max(np.abs(self.vx)) + np.max(np.abs(self.vy))
        if max_v > 0:
            dt_adv = 0.9 * dx / max_v  # CFL-stable substep
        else:
            dt_adv = config.DT
        n_sub = max(1, int(np.ceil(config.DT / dt_adv)))
        dt_adv = config.DT / n_sub  # exact subdivision

        VX_pad = np.pad(self.vx, 1, mode="edge")
        VY_pad = np.pad(self.vy, 1, mode="edge")
        VX_R = VX_pad[1:-1, 2:]
        VY_B = VY_pad[2:, 1:-1]

        for _ in range(n_sub):
            C_pad = np.pad(C, 1, mode="constant", constant_values=0)
            C_L = C_pad[1:-1, :-2]
            C_R = C_pad[1:-1, 2:]
            C_T = C_pad[:-2, 1:-1]
            C_B = C_pad[2:, 1:-1]

            F_L = np.where(self.vx > 0, self.vx * C_L, self.vx * C)
            F_R = np.where(VX_R > 0, VX_R * C, VX_R * C_R)
            G_T = np.where(self.vy > 0, self.vy * C_T, self.vy * C)
            G_B = np.where(VY_B > 0, VY_B * C, VY_B * C_B)

            adv = (F_L - F_R) / dx + (G_T - G_B) / dy
            C = C + dt_adv * adv
            C = np.maximum(C, 0.0)
            C[self.outlet_pos] = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 — Diffusion implicit (ADI: alternating direction)
        # Solve row-by-row in x, then column-by-column in y
        # ══════════════════════════════════════════════════════════════════════
        dt_diff = config.DT
        r_x = D * dt_diff / dx**2  # shape (n, n)
        r_y = D * dt_diff / dy**2

        # ── x-sweep: solve tridiagonal along each row ─────────────────────────
        C_half = np.zeros_like(C)
        for i in range(n):
            r = r_x[i, :]  # (n,) diffusion numbers
            d = np.zeros(n)
            d[:] = C[i, :]

            # Thomas algorithm for: -r*C[j-1] + (1+2r)*C[j] - r*C[j+1] = d[j]
            a = -r  # sub-diagonal
            b = 1 + 2 * r  # main diagonal
            c_diag = -r  # super-diagonal

            # no-flux BCs: ghost = boundary cell → modify first and last
            b_mod = b.copy()
            c_mod = c_diag.copy()
            a_mod = a.copy()
            b_mod[0] = 1 + r[0]  # only one neighbor
            b_mod[-1] = 1 + r[-1]

            C_half[i, :] = self._thomas(a_mod, b_mod, c_mod, d)

        # ── y-sweep: solve tridiagonal along each column ──────────────────────
        C_new = np.zeros_like(C_half)
        for j in range(n):
            r = r_y[:, j]
            d = C_half[:, j]

            a = -r
            b = 1 + 2 * r
            c_diag = -r

            b_mod = b.copy()
            b_mod[0] = 1 + r[0]
            b_mod[-1] = 1 + r[-1]

            C_new[:, j] = self._thomas(a, b_mod, c_diag, d)

        C_new = np.maximum(C_new, 0.0)
        C_new[self.outlet_pos] = 0.0
        self.C = C_new
        return C_new

    def _thomas(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system."""
        n = len(d)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        x = np.zeros(n)

        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
        for i in range(1, n):
            m = b[i] - a[i] * c_[i - 1]
            c_[i] = c[i] / m
            d_[i] = (d[i] - a[i] * d_[i - 1]) / m

        x[-1] = d_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i + 1]
        return x

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
