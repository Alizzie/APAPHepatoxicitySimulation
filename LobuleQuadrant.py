import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label
from config import Config

config = Config()


class LobuleQuadrant:
    """
    Spatiotemporal model of a liver lobule quadrant.
    Features:
    - Darcy flow velocity field
    - Explicit upwind advection with strict boundary mass tracking
    - Strict mass-conservative sinusoid-hepatocyte exchange
    - Instant intracellular mixing (Hepatocytes as well-mixed units)
    - Explicit sinusoid-only diffusion
    """

    DIRS = ["top-left", "top-right", "bottom-left", "bottom-right"]

    def __init__(
        self, direction: str, grid_size: int = config.GRID_N, dose: float = config.DOSE
    ):
        if direction not in self.DIRS:
            raise ValueError(f"direction must be one of {self.DIRS}")

        self.direction = direction
        self.checkboard_size = grid_size
        self.physio_grid = self._build_struc_matrix()
        self.sin_mask = self.physio_grid == 1
        self.hep_mask = self.physio_grid == 0

        # Each hepa block get unique id
        self.hep_labels, self.num_heps = label(self.physio_grid == 0)

        self.grid_size = self.physio_grid.shape[0]
        self.inlet_pos = self._get_corner("inlet", self.grid_size)
        self.outlet_pos = self._get_corner("outlet", self.grid_size)

        self.P, self.vx, self.vy = self._compute_simple_flow()
        self.C = self._init_concentration()

        # Systemic Reservoir (starts with entire dose in blood, then exchanges with grid over time)
        self.c_reservoir = dose / config.V_BLOOD

        self.total_mass_history = []
        self.time_history = []

        print(
            f"Initialized {direction} quadrant with grid size {self.grid_size}x{self.grid_size}"
        )
        print(
            f"Total pixels: {self.grid_size**2}, Hepatocytes: {self.num_heps}, Sinusoids: {self.grid_size**2 - self.num_heps}"
        )
        print(f"Inlet position: {self.inlet_pos}, Outlet position: {self.outlet_pos}")
        print(f"Initial reservoir concentration: {self.c_reservoir:.3e} µM")

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _cell_sizes(self):
        sizes = []
        for i in range(self.checkboard_size):
            if i % 2 == 0:
                sizes.append(
                    1 if i in (0, self.checkboard_size - 1) else config.SIN_SIZE
                )
            else:
                sizes.append(config.HEPA_SIZE)
        return sizes

    def _build_struc_matrix(self):
        lattice = np.zeros((self.checkboard_size, self.checkboard_size), dtype=int)
        for i in range(0, self.checkboard_size, 2):
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
    def _compute_simple_flow(self):
        """
        An easier, mathematically bulletproof alternative to Darcy flow.
        Forces uniform fluid movement exclusively right and down through the sinusoids,
        naturally branching at intersections and ignoring dead ends.
        """
        vx = np.zeros((self.grid_size, self.grid_size))
        vy = np.zeros((self.grid_size, self.grid_size))
        P = np.zeros(
            (self.grid_size, self.grid_size)
        )  # Dummy pressure for the visualizer

        sin_mask = self.physio_grid == 1

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if not sin_mask[r, c]:
                    continue

                # Check if the neighbor to the right/bottom is also a sinusoid
                can_go_right = (c < self.grid_size - 1) and sin_mask[r, c + 1]
                can_go_down = (r < self.grid_size - 1) and sin_mask[r + 1, c]

                # Assign velocities based on available clear paths
                if can_go_right and can_go_down:
                    # Intersection: split flow evenly (45 degree vector)
                    vx[r, c] = config.U_X * 0.7071
                    vy[r, c] = config.U_X * 0.7071
                elif can_go_right:
                    # Straight horizontal channel
                    vx[r, c] = config.U_X
                elif can_go_down:
                    # Straight vertical channel
                    vy[r, c] = config.U_X
                elif (r, c) == self.outlet_pos:
                    vx[r, c] = config.U_X
                    vy[r, c] = config.U_X

        return P, vx, vy

    # ══════════════════════════════════════════════════════════════════════════
    # ── compute_flux
    # ══════════════════════════════════════════════════════════════════════════
    def compute_flux(self, dt=None):
        """
        Computes one time step of drug transport and metabolism in the lobule quadrant.
        Steps:
        1. Advection of drug in (solely) sinusoids with strict mass tracking at inlet/outlet
        2. Conservative exchange between sinusoids and hepatocytes based on local concentrations
        3. Instant intracellular mixing within hepatocytes
        4. Diffusion of drug in sinusoids (explicit finite difference)
        5. First-order metabolic decay in hepatocytes
        """
        dx = dy = config.LOBULE_SIZE / self.C.shape[0]
        sin_mask = self.physio_grid == 1
        hep_mask = self.physio_grid == 0

        C_sin = self.C * sin_mask
        C_hep = self.C * hep_mask

        # STEP 1 & 2 — Advection with Boundary Mass Tracking
        max_v = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)))
        dt_adv = (0.9 * dx / max_v) if max_v > 0 else dt or config.DT
        n_sub = max(1, int(np.ceil(config.DT / dt_adv)))
        dt_adv = config.DT / n_sub
        print(
            f"Subcycling advection into {n_sub} steps of {dt_adv:.3e} s each, max velocity = {max_v:.3e} m/s, CFL = {max_v * dt_adv / dx:.3f}, old dt = {config.DT:.3e} s"
        )

        # Add ghost cells to velocity fields for flux calculations at boundaries
        vx_pad = np.pad(self.vx, 1, mode="edge")
        vy_pad = np.pad(self.vy, 1, mode="edge")
        vx_r = vx_pad[1:-1, 2:]
        vy_b = vy_pad[2:, 1:-1]

        C_sin_tmp = C_sin.copy()
        mass_entered_grid = 0.0
        mass_left_grid = 0.0

        for _ in range(n_sub):
            mass_before = np.sum(C_sin_tmp) * config.V_PIXEL
            C_sin_tmp[self.inlet_pos] = self.c_reservoir  # Injection

            C_pad = np.pad(C_sin_tmp, 1, mode="constant", constant_values=0.0)
            C_pad[1:-1, 0] = np.where(self.vx[:, 0] > 0, C_sin_tmp[:, 0], 0.0)
            C_pad[1:-1, -1] = np.where(self.vx[:, -1] > 0, C_sin_tmp[:, -1], 0.0)
            C_pad[0, 1:-1] = np.where(self.vy[0, :] > 0, C_sin_tmp[0, :], 0.0)
            C_pad[-1, 1:-1] = np.where(self.vy[-1, :] > 0, C_sin_tmp[-1, :], 0.0)

            C_L = C_pad[1:-1, :-2]
            C_R = C_pad[1:-1, 2:]
            C_T = C_pad[:-2, 1:-1]
            C_B = C_pad[2:, 1:-1]

            F_L = np.where(self.vx > 0, self.vx * C_L, self.vx * C_sin_tmp)
            F_R = np.where(vx_r > 0, vx_r * C_sin_tmp, vx_r * C_R)
            G_T = np.where(self.vy > 0, self.vy * C_T, self.vy * C_sin_tmp)
            G_B = np.where(vy_b > 0, vy_b * C_sin_tmp, vy_b * C_B)

            adv = (F_L - F_R) / dx + (G_T - G_B) / dy
            C_sin_tmp = np.maximum(C_sin_tmp + dt_adv * adv, 0.0) * sin_mask

            # Absorption / Drain at outlet
            mass_out_this_step = C_sin_tmp[self.outlet_pos] * config.V_PIXEL
            mass_left_grid += mass_out_this_step
            C_sin_tmp[self.outlet_pos] = 0.0

            mass_after = np.sum(C_sin_tmp) * config.V_PIXEL
            mass_entered_grid += mass_after + mass_out_this_step - mass_before

        C_sin = C_sin_tmp
        print(
            f"Advection step complete. Mass entered grid: {mass_entered_grid:.3e} µM·m³, Mass left grid: {mass_left_grid:.3e} µM·m³, Net change: {mass_entered_grid - mass_left_grid:.3e} µM·m³"
        )

        # STEP 3 — Exchange (Conservative) + Intracellular Mixing
        # Uptake into hepatocytes: mass leaving sinusoid = F_unbound * CL_influx * C_sin * dt / V_sin
        # Efflux back to sinusoid: mass leaving hepatocyte = CL_efflux * C_hep * dt / V_hep
        mass_leaving_sin = (config.F_UNBOUND * config.CL_INFLUX * C_sin) * config.DT
        mass_leaving_hep = (config.CL_EFFLUX * C_hep) * config.DT

        hep_pad = np.pad(hep_mask.astype(float), 1, mode="constant", constant_values=0)
        sin_pad = np.pad(sin_mask.astype(float), 1, mode="constant", constant_values=0)

        hep_nbrs = (
            hep_pad[:-2, 1:-1]
            + hep_pad[2:, 1:-1]
            + hep_pad[1:-1, :-2]
            + hep_pad[1:-1, 2:]
        )
        sin_nbrs = (
            sin_pad[:-2, 1:-1]
            + sin_pad[2:, 1:-1]
            + sin_pad[1:-1, :-2]
            + sin_pad[1:-1, 2:]
        )

        # Share mass leaving each pixel equally among neighboirng pixels of the opposite type
        s_give = np.divide(
            mass_leaving_sin, hep_nbrs, out=np.zeros_like(C_sin), where=hep_nbrs > 0
        )
        h_give = np.divide(
            mass_leaving_hep, sin_nbrs, out=np.zeros_like(C_hep), where=sin_nbrs > 0
        )

        s_give_pad = np.pad(s_give, 1, mode="constant", constant_values=0)
        h_give_pad = np.pad(h_give, 1, mode="constant", constant_values=0)

        # Mass received by each pixel is the sum of contributions from neighbors
        m_rec_hep = (
            s_give_pad[:-2, 1:-1]
            + s_give_pad[2:, 1:-1]
            + s_give_pad[1:-1, :-2]
            + s_give_pad[1:-1, 2:]
        ) * hep_mask

        m_rec_sin = (
            h_give_pad[:-2, 1:-1]
            + h_give_pad[2:, 1:-1]
            + h_give_pad[1:-1, :-2]
            + h_give_pad[1:-1, 2:]
        ) * sin_mask

        # Update concentrations based on mass leaving and mass received, ensuring no negative concentrations
        mass_sin = (
            C_sin * config.V_PIXEL
            - np.where(hep_nbrs > 0, mass_leaving_sin, 0)
            + m_rec_sin
        )

        mass_hep = (
            C_hep * config.V_PIXEL
            - np.where(sin_nbrs > 0, mass_leaving_hep, 0)
            + m_rec_hep
        )

        C_sin = np.maximum(mass_sin / config.V_PIXEL, 0.0) * sin_mask
        C_hep = np.maximum(mass_hep / config.V_PIXEL, 0.0) * hep_mask

        # --- Intracellular Mixing: Spread drug evenly inside each cell ---
        # Sum mass in each label, count pixels, and calculate averages
        hep_sums = np.bincount(self.hep_labels.ravel(), weights=C_hep.ravel())
        hep_cnts = np.bincount(self.hep_labels.ravel())
        hep_avgs = np.divide(
            hep_sums, hep_cnts, out=np.zeros_like(hep_sums), where=hep_cnts > 0
        )
        C_hep = hep_avgs[self.hep_labels] * hep_mask

        # STEP 4 — Sinusoid Diffusion
        # Continuous equation: dC/dt = D * (d²C/dx² + d²C/dy²)
        # Discretized with explicit finite difference: C_new = C_old + r * (C_L + C_R + C_T + C_B - 4*C_center)
        r = config.D_SIN * config.DT / dx**2
        C_pad = np.pad(C_sin, 1, mode="edge")
        M_pad = np.pad(sin_mask.astype(float), 1, mode="constant")
        C_L = C_pad[1:-1, :-2]
        M_L = M_pad[1:-1, :-2]
        C_R = C_pad[1:-1, 2:]
        M_R = M_pad[1:-1, 2:]
        C_T = C_pad[:-2, 1:-1]
        M_T = M_pad[:-2, 1:-1]
        C_B = C_pad[2:, 1:-1]
        M_B = M_pad[2:, 1:-1]

        flux = (
            (C_L - C_sin) * M_L
            + (C_R - C_sin) * M_R
            + (C_T - C_sin) * M_T
            + (C_B - C_sin) * M_B
        )

        C_sin = np.maximum(C_sin + r * flux * sin_mask, 0.0)

        # STEP 5 — Systemic Balance
        # Replenish reservoir with mass that left grid
        new_blood_mass = (
            (self.c_reservoir * config.V_BLOOD)
            - (mass_entered_grid * config.N_SINUSOIDS)
            + (mass_left_grid * config.N_SINUSOIDS)
        )
        self.c_reservoir = max(new_blood_mass / config.V_BLOOD, 0.0)

        self.C = C_sin + C_hep
        return self.C

    # ── Mass Audit & Diagnostics ──────────────────────────────────────────────

    def get_total_mass(self):
        m_s = np.sum(self.C * (self.physio_grid == 1) * config.V_PIXEL)
        m_h = np.sum(self.C * (self.physio_grid == 0) * config.V_PIXEL)
        return m_s + m_h

    def audit_mass(self, step_num=0):
        grid_m = self.get_total_mass()
        total_m = (grid_m * config.N_SINUSOIDS) + (self.c_reservoir * config.V_BLOOD)

        # DOSE is already in µmol, so we compare directly. No need for Diff_2 anymore!
        diff = total_m - config.DOSE

        print(
            f"=== STEP {step_num} | Total Mass: {total_m:.6e} µmol | "
            f"Expected: {config.DOSE:.6e} µmol | Diff: {diff:.6e} ==="
        )

    def audit_mass2(self, step_num=0):
        """Prints a strict accounting of every molecule in the simulation."""
        grid_mass = self.get_total_mass()
        liver_mass = grid_mass * config.N_SINUSOIDS
        blood_mass = self.c_reservoir * config.V_BLOOD
        total_mass = liver_mass + blood_mass

        print(f"\n=== STEP {step_num} MASS AUDIT ===")
        print(f"Grid Mass (1 Lobule): {grid_mass:.6e}")
        print(f"Liver Mass (Total):   {liver_mass:.6e}")
        print(f"Blood Reservoir Mass: {blood_mass:.6e}")
        print(f"Total System Mass:    {total_mass:.6e}")
        print(f"Expected (DOSE):      {config.DOSE:.6e}")

        leak = total_mass - config.DOSE
        if abs(leak) > 1e-10:
            print(f"⚠️ MASS LEAK DETECTED: {leak:.6e}")
        else:
            print(f"✅ Mass Conserved. (Diff: {leak:.6e})")
        print("============================\n")

    def _init_concentration(self):
        return np.zeros(self.physio_grid.shape)

    def record(self):
        self.total_mass_history.append(self.get_total_mass())
        self.time_history.append(len(self.total_mass_history) * config.DT)
