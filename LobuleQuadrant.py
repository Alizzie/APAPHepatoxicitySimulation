import numpy as np
from scipy.ndimage import label
from config import Config
from MetabolismModel import MetabolismModel


class LobuleQuadrant:
    """
    Spatiotemporal model of a liver lobule quadrant.
    """

    def __init__(
        self,
        grid_size: int = None,
        dose: float = None,
        exchange_on: bool = True,
        config_override: Config = None,
        metabolism_on: bool = True,
    ):
        self.config = config_override or Config()
        self.checkboard_size = grid_size or self.config.GRID_N
        self.physio_grid = self._build_struc_matrix()
        self.sin_mask = self.physio_grid == 1
        self.hep_mask = self.physio_grid == 0
        self.hep_labels, self.num_heps = label(
            self.physio_grid == 0
        )  # Label connected hepatocyte blocks for intracellular mixing

        self.lobule_dose = dose or self.config.DOSE  # in mass units (µmol)
        self.exchange_on = exchange_on

        self.grid_size = self.physio_grid.shape[0]
        self.inlet_pos = (0, 0)
        self.outlet_pos = (self.grid_size - 1, self.grid_size - 1)

        self.vx, self.vy = self._compute_simple_flow()
        self.C = self._init_concentration()

        self.metabolism = None
        if metabolism_on:
            self.metabolism = MetabolismModel(
                physio_grid=self.physio_grid,
                hep_labels=self.hep_labels,
                inlet_pos=self.inlet_pos,
                outlet_pos=self.outlet_pos,
            )

        self.total_mass_exited = 0.0
        self.total_mass_metab = 0.0
        self.mass_injected_so_far = 0.0

        self.current_time = 0.0
        self.time_history = []
        self.exited_mass_history = []
        self.metabolized_mass_history = []
        self.total_system_mass_history = []
        self.grid_mass_history = []
        self.concentration_history = []

        print(
            f"Total pixels: {self.grid_size**2}, Hepatocytes: {self.num_heps}, Sinusoids: {self.grid_size**2 - self.num_heps}"
        )
        print(f"Inlet position: {self.inlet_pos}, Outlet position: {self.outlet_pos}")

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _cell_sizes(self):
        sizes = []
        for i in range(self.checkboard_size):
            if i % 2 == 0:
                sizes.append(
                    1 if i in (0, self.checkboard_size - 1) else self.config.SIN_SIZE
                )
            else:
                sizes.append(self.config.HEPA_SIZE)
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

    # ── Flow Field Velocity based ───────────────────────────────────────────────────
    def _compute_simple_flow(self):
        """
        Computes a simple velocity field that follows the sinusoid channels,
        with flow entering at the top-left and exiting at the bottom-right.
        """
        vx = np.zeros((self.grid_size, self.grid_size))
        vy = np.zeros((self.grid_size, self.grid_size))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if not self.sin_mask[r, c]:
                    continue

                # Check if the neighbor to the right/bottom is also a sinusoid
                can_go_right = (c < self.grid_size - 1) and self.sin_mask[r, c + 1]
                can_go_down = (r < self.grid_size - 1) and self.sin_mask[r + 1, c]

                # Assign velocities based on available clear paths
                if can_go_right and can_go_down:
                    # Intersection: split flow evenly (45 degree vector)
                    vx[r, c] = self.config.U_X * 0.7071
                    vy[r, c] = self.config.U_X * 0.7071
                elif can_go_right:
                    # Straight horizontal channel
                    vx[r, c] = self.config.U_X
                elif can_go_down:
                    # Straight vertical channel
                    vy[r, c] = self.config.U_X
                elif (r, c) == self.outlet_pos:
                    vx[r, c] = 0
                    vy[r, c] = 0

        return vx, vy

    # ══════════════════════════════════════════════════════════════════════════
    # ── compute_flux
    # ══════════════════════════════════════════════════════════════════════════
    def compute_flux(self, dt=None):
        """
        Computes one time step of drug transport and metabolism in the lobule quadrant.
        Steps:
        1. Advection of drug in (solely) sinusoids with strict mass tracking at inlet/outlet
        2. Conservative exchange between sinusoids and hepatocytes based on local concentrations
        4. Diffusion of drug in sinusoids (explicit finite difference)
        """
        dx = dy = self.config.LOBULE_SIZE / self.C.shape[0]

        C_sin = self.C * self.sin_mask
        C_hep = self.C * self.hep_mask

        # STEP 1  — Advection with Boundary Mass Tracking
        step_dt = dt if dt is not None else self.config.DT

        # CFL Stability Check for Advection
        max_outflow = np.max(np.abs(self.vx) + np.abs(self.vy))
        if max_outflow > 0:
            dt_cfl = dx / max_outflow
            if step_dt > dt_cfl * 0.99:
                raise ValueError(
                    f"CFL stability violation! Provided dt ({step_dt:.3e} s) is larger "
                    f"than the maximum stable dt ({dt_cfl:.3e} s). Please provide a smaller dt."
                )

        # Add ghost cells to velocity fields for flux calculations at boundaries
        vx_pad = np.pad(self.vx, 1, mode="edge")
        vy_pad = np.pad(self.vy, 1, mode="edge")
        vx_r = vx_pad[1:-1, 2:]
        vy_b = vy_pad[2:, 1:-1]

        C_sin_tmp = C_sin.copy()

        # injection_time = 2.0

        # if self.mass_injected_so_far < self.lobule_dose:
        #     dose_per_second = self.lobule_dose / injection_time
        #     mass_this_step = dose_per_second * step_dt

        #     if self.mass_injected_so_far + mass_this_step > self.lobule_dose:
        #         mass_this_step = self.lobule_dose - self.mass_injected_so_far
        #     added_conc = mass_this_step / self.config.V_PIXEL
        #     C_sin_tmp[self.inlet_pos] += added_conc
        #     self.mass_injected_so_far += mass_this_step

        C_pad = np.pad(C_sin_tmp, 1, mode="constant", constant_values=0)
        C_L = C_pad[1:-1, :-2]
        C_R = C_pad[1:-1, 2:]
        C_T = C_pad[:-2, 1:-1]
        C_B = C_pad[2:, 1:-1]

        F_L = np.where(self.vx > 0, self.vx * C_L, self.vx * C_sin_tmp)
        F_R = np.where(vx_r > 0, vx_r * C_sin_tmp, vx_r * C_R)
        G_T = np.where(self.vy > 0, self.vy * C_T, self.vy * C_sin_tmp)
        G_B = np.where(vy_b > 0, vy_b * C_sin_tmp, vy_b * C_B)

        adv = (F_L - F_R) / dx + (G_T - G_B) / dy
        C_sin_tmp = np.maximum(C_sin_tmp + step_dt * adv, 0.0) * self.sin_mask

        # Absorption / Drain at outlet
        mass_out = C_sin_tmp[self.outlet_pos] * self.config.V_PIXEL
        self.total_mass_exited += mass_out
        C_sin_tmp[self.outlet_pos] = 0.0

        C_sin = C_sin_tmp

        # STEP 2 — Exchange (Conservative) + Intracellular Mixing
        # Uptake into hepatocytes: mass leaving sinusoid = F_unbound * CL_influx * C_sin * dt / V_sin
        # Efflux back to sinusoid: mass leaving hepatocyte = CL_efflux * C_hep * dt / V_hep
        if self.exchange_on:
            C_sin, C_hep = self._hepatocyte_exchange(C_sin, C_hep)

        if self.metabolism is not None:
            self.metabolism.P = np.copy(C_hep)
            mass_before = np.sum(C_hep) * self.config.V_PIXEL
            self.metabolism.step()
            self.metabolism.record()
            C_hep = np.copy(self.metabolism.P)
            self.hep_mask = self.metabolism.hep_mask

            C_hep[~self.hep_mask] = 0.0

            mass_after = np.sum(C_hep) * self.config.V_PIXEL
            self.total_mass_metab += mass_before - mass_after

        # STEP 3 — Sinusoid Diffusion
        # Continuous equation: dC/dt = D * (d²C/dx² + d²C/dy²)
        # Discretized with explicit finite difference: C_new = C_old + r * (C_L + C_R + C_T + C_B - 4*C_center)
        r = self.config.D_SIN * self.config.DT / dx**2

        if r > 0.25:
            raise ValueError(
                f"Diffusion stability violation! Fourier number (r={r:.3f}) exceeds 0.25. "
                f"Please provide a smaller dt."
            )

        C_pad = np.pad(C_sin, 1, mode="edge")
        M_pad = np.pad(self.sin_mask.astype(float), 1, mode="constant")
        C_L, M_L = C_pad[1:-1, :-2], M_pad[1:-1, :-2]
        C_R, M_R = C_pad[1:-1, 2:], M_pad[1:-1, 2:]
        C_T, M_T = C_pad[:-2, 1:-1], M_pad[:-2, 1:-1]
        C_B, M_B = C_pad[2:, 1:-1], M_pad[2:, 1:-1]

        flux = (
            (C_L - C_sin) * M_L
            + (C_R - C_sin) * M_R
            + (C_T - C_sin) * M_T
            + (C_B - C_sin) * M_B
        )

        C_sin = np.maximum(C_sin + r * flux * self.sin_mask, 0.0)

        self.C = C_sin + C_hep
        return self.C

    def _hepatocyte_exchange(self, C_sin, C_hep):
        mass_leaving_sin = (
            self.config.F_UNBOUND * self.config.CL_INFLUX * C_sin
        ) * self.config.DT
        mass_leaving_hep = (self.config.CL_EFFLUX * C_hep) * self.config.DT

        hep_pad = np.pad(
            self.hep_mask.astype(float), 1, mode="constant", constant_values=0
        )
        sin_pad = np.pad(
            self.sin_mask.astype(float), 1, mode="constant", constant_values=0
        )

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
        ) * self.hep_mask

        m_rec_sin = (
            h_give_pad[:-2, 1:-1]
            + h_give_pad[2:, 1:-1]
            + h_give_pad[1:-1, :-2]
            + h_give_pad[1:-1, 2:]
        ) * self.sin_mask

        # Update concentrations based on mass leaving and mass received, ensuring no negative concentrations
        mass_sin = (
            C_sin * self.config.V_PIXEL
            - np.where(hep_nbrs > 0, mass_leaving_sin, 0)
            + m_rec_sin
        )

        mass_hep = (
            C_hep * self.config.V_PIXEL
            - np.where(sin_nbrs > 0, mass_leaving_hep, 0)
            + m_rec_hep
        )

        C_sin = np.maximum(mass_sin / self.config.V_PIXEL, 0.0) * self.sin_mask
        C_hep = np.maximum(mass_hep / self.config.V_PIXEL, 0.0) * self.hep_mask

        # --- Intracellular Mixing: Spread drug evenly inside each cell ---
        # Sum mass in each label, count pixels, and calculate averages
        hep_sums = np.bincount(self.hep_labels.ravel(), weights=C_hep.ravel())
        hep_cnts = np.bincount(self.hep_labels.ravel())
        hep_avgs = np.divide(
            hep_sums, hep_cnts, out=np.zeros_like(hep_sums), where=hep_cnts > 0
        )

        C_hep = hep_avgs[self.hep_labels] * self.hep_mask
        return C_sin, C_hep

    # ── Mass Audit & Diagnostics ──────────────────────────────────────────────

    def get_total_mass(self):
        m_s = np.sum(self.C * self.sin_mask * self.config.V_PIXEL)
        m_h = np.sum(self.C * self.hep_mask * self.config.V_PIXEL)
        return m_s + m_h

    def audit_mass2(self, step_num=0):
        """Prints a strict accounting of every molecule in the simulation."""
        grid_mass = self.get_total_mass()
        current_total = grid_mass + self.total_mass_exited + self.total_mass_metab

        leak = current_total - self.lobule_dose

        print(f"\n=== STEP {step_num} MASS AUDIT ===")
        print(f"Grid Mass (1 Lobule): {grid_mass:.6e}")
        print(f"Exited Mass (Total):   {self.total_mass_exited:.6e}")
        print(f"Metabolized Mass (Total): {self.total_mass_metab:.6e}")
        print(f"Total System Mass:    {current_total:.6e}")
        print(f"Target Total Dose:      {self.lobule_dose:.6e}")

        if abs(leak) > 1e-10:
            print(f"⚠️ MASS LEAK DETECTED: {leak:.6e}")
        else:
            print(f"✅ Mass Conserved. (Diff: {leak:.6e})")
        print("============================\n")

    def _init_concentration(self):
        C = np.zeros(self.physio_grid.shape)
        C[self.inlet_pos] = self.lobule_dose / self.config.V_PIXEL
        return C

    def record(self, dt=None, save_frame=False):
        step_dt = dt if dt is not None else self.config.DT
        self.current_time += step_dt

        self.time_history.append(self.current_time)
        self.exited_mass_history.append(self.total_mass_exited)
        self.total_system_mass_history.append(
            self.get_total_mass() + self.total_mass_exited + self.total_mass_metab
        )
        self.grid_mass_history.append(self.get_total_mass())
        self.metabolized_mass_history.append(self.total_mass_metab)

        if save_frame:
            self.concentration_history.append(self.C.copy())
