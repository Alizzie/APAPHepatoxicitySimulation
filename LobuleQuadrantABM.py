import numpy as np
from scipy.ndimage import label
from config import Config

config = Config()


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
        base_uptake_pct: float = 0.006,  # drug uptake rate from blood to tissue
        base_efflux_pct: float = 0.004,  # drug efflux rate from tissue to blood
    ):
        self.config = config_override if config_override else config

        self.checkboard_size = grid_size or config.GRID_N
        self.physio_grid = self._build_struc_matrix()
        self.sin_mask = self.physio_grid == 1
        self.hep_mask = self.physio_grid == 0
        self.hep_labels, self.num_heps = label(
            self.physio_grid == 0
        )  # Label connected hepatocyte blocks for intracellular mixing

        self.lobule_dose = dose or config.DOSE  # in mass units (µmol)
        self.exchange_on = exchange_on
        self.base_uptake_pct = base_uptake_pct
        self.base_efflux_pct = base_efflux_pct

        self.grid_size = self.physio_grid.shape[0]
        self.inlet_pos = (0, 0)
        self.outlet_pos = (self.grid_size - 1, self.grid_size - 1)

        self.zonation = self._build_zone_map()
        self.PROB_CLEAR_ZONE1 = 1e-4
        self.PROB_CLEAR_ZONE2 = 1.5e-4
        self.PROB_CLEAR_ZONE3 = 3e-4
        self.fraction_to_destroy = 0.001

        self.toxicity_field = np.zeros_like(self.physio_grid, dtype=float)
        self.is_cell_dead = np.zeros_like(self.physio_grid, dtype=bool)
        self.toxicity_threshold = 1.0
        self.toxicity_impact = 0.02

        self.C = self._init_concentration()

        self.total_mass_exited = 0.0
        self.total_mass_metab = 0.0

        self.current_time = 0.0
        self.time_history = []
        self.exited_mass_history = []
        self.total_system_mass_history = []
        self.metabolized_mass_history = []
        self.grid_mass_history = []
        self.concentration_history = []
        self.reflux_mass = 0.0

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

    def _build_zone_map(self):
        """
        Assigns zone 1, 2, or 3 to each hepatocyte pixel based on its
        Manhattan distance from the inlet corner.

        With ZONATION hepatocyctes per side and 25 total hepatocytes
        along the diagonal, zones split as 8/8/9.
        """

        n = self.physio_grid.shape[0]

        rows = np.arange(n)
        cols = np.arange(n)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")

        in_r, in_c = self.inlet_pos
        out_r, out_c = self.outlet_pos

        dist = np.abs(rr - in_r) + np.abs(cc - in_c)
        max_dist = np.abs(out_r - in_r) + np.abs(out_c - in_c)
        dist_norm = dist / max_dist

        # Split into 3 equal zones based on normalized distance
        unique_labels = np.unique(self.hep_labels[self.hep_mask])
        n_hep = len(unique_labels)

        n_z1 = n_hep // 3
        n_z2 = n_hep // 3

        # Mean the distance for one hepatocyte block (label)
        label_dist = {}
        for label in unique_labels:
            mask = self.hep_labels == label
            label_dist[label] = dist_norm[mask].mean()

        sorted_labels = sorted(label_dist, key=lambda l: label_dist[l])

        label_zone = {}
        for i, label in enumerate(sorted_labels):
            if i < n_z1:
                label_zone[label] = 1
            elif i < n_z1 + n_z2:
                label_zone[label] = 2
            else:
                label_zone[label] = 3

        # Build zone map for all pixels
        zone_map = np.zeros_like(self.physio_grid, dtype=int)
        for label, zone in label_zone.items():
            zone_map[self.hep_labels == label] = zone

        return zone_map

    # ══════════════════════════════════════════════════════════════════════════
    # ── compute_flux
    # ══════════════════════════════════════════════════════════════════════════
    def compute_flux(self):
        C_sin = self.C * self.sin_mask
        n = self.grid_size

        # Generate all random values at once
        flux_pct = np.random.normal(0.8, 0.1, (n, n))
        split_flux = np.random.normal(0.50, 0.1, (n, n))
        split_ref = np.random.normal(0.50, 0.1, (n, n))

        mass_flux = C_sin * flux_pct
        mass_ref = C_sin - mass_flux
        self.reflux_mass += float(np.sum(mass_ref * self.sin_mask))

        m_Right = mass_flux * split_flux
        m_Down = mass_flux * (1.0 - split_flux)
        m_Left = mass_ref * split_ref
        m_Up = mass_ref * (1.0 - split_ref)

        # Mask for valid neighbors
        can_R = np.zeros((n, n), bool)
        can_D = np.zeros((n, n), bool)
        can_L = np.zeros((n, n), bool)
        can_U = np.zeros((n, n), bool)

        can_R[:, :-1] = self.sin_mask[:, :-1] & self.sin_mask[:, 1:]
        can_D[:-1, :] = self.sin_mask[:-1, :] & self.sin_mask[1:, :]
        can_L[:, 1:] = self.sin_mask[:, 1:] & self.sin_mask[:, :-1]
        can_U[1:, :] = self.sin_mask[1:, :] & self.sin_mask[:-1, :]

        C_new = np.zeros((n, n))

        # Each direction: shifted contribution lands in the target cell
        C_new[:, 1:] += np.where(can_R[:, :-1], m_Right[:, :-1], 0)  # Right
        C_new[1:, :] += np.where(can_D[:-1, :], m_Down[:-1, :], 0)  # Down
        C_new[:, :-1] += np.where(can_L[:, 1:], m_Left[:, 1:], 0)  # Left
        C_new[:-1, :] += np.where(can_U[1:, :], m_Up[1:, :], 0)  # Up

        C_new += np.where(self.sin_mask & ~can_R, m_Right, 0)
        C_new += np.where(self.sin_mask & ~can_D, m_Down, 0)
        C_new += np.where(self.sin_mask & ~can_L, m_Left, 0)
        C_new += np.where(self.sin_mask & ~can_U, m_Up, 0)

        mass_out = C_new[self.outlet_pos] * config.V_PIXEL
        self.total_mass_exited += mass_out
        C_new[self.outlet_pos] = 0.0
        self.C = C_new + (self.C * self.hep_mask)

        if self.exchange_on:
            C_sin = self.C * self.sin_mask
            C_hep = self.C * self.hep_mask
            C_sin, C_hep = self._hepatocyte_exchange(C_sin, C_hep)
            self.C = C_sin + C_hep

        clearance_probs = np.zeros_like(C_hep)
        clearance_probs[self.zonation == 1] = self.PROB_CLEAR_ZONE1
        clearance_probs[self.zonation == 2] = self.PROB_CLEAR_ZONE2
        clearance_probs[self.zonation == 3] = self.PROB_CLEAR_ZONE3

        clear_random_cells = (
            np.random.rand(*C_hep.shape) < clearance_probs
        ) * self.hep_mask
        mass_destroyed = C_hep * self.fraction_to_destroy * clear_random_cells
        self.toxicity_field += mass_destroyed * self.toxicity_impact
        just_died = (
            self.toxicity_field >= self.toxicity_threshold
        ) & ~self.is_cell_dead

        self.is_cell_dead[just_died] = True
        self.hep_mask = self.hep_mask & ~self.is_cell_dead

        C_hep -= mass_destroyed
        spilled_to_sin = C_hep * just_died
        C_hep -= spilled_to_sin
        self.total_mass_metab += (
            np.sum(mass_destroyed + spilled_to_sin) * config.V_PIXEL
        )
        self.C = C_sin + C_hep

        return self.C

    def _hepatocyte_exchange(self, C_sin, C_hep):

        pct_uptake = np.clip(
            np.random.normal(
                self.base_uptake_pct, self.base_uptake_pct * 0.2, C_sin.shape
            ),
            0,
            1,
        )
        pct_efflux = np.clip(
            np.random.normal(
                self.base_efflux_pct, self.base_efflux_pct * 0.2, C_hep.shape
            ),
            0,
            1,
        )

        current_sin_mass = C_sin * config.V_PIXEL
        current_hep_mass = C_hep * config.V_PIXEL
        mass_leaving_sin = current_sin_mass * pct_uptake
        mass_leaving_hep = current_hep_mass * pct_efflux

        # Find boundaries (cell membranes)
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

        # Membrane exchange: even distribution across all neighboring cells of the opposite type
        s_give = np.divide(
            mass_leaving_sin, hep_nbrs, out=np.zeros_like(C_sin), where=hep_nbrs > 0
        )
        h_give = np.divide(
            mass_leaving_hep, sin_nbrs, out=np.zeros_like(C_hep), where=sin_nbrs > 0
        )

        s_give_pad = np.pad(s_give, 1, mode="constant", constant_values=0)
        h_give_pad = np.pad(h_give, 1, mode="constant", constant_values=0)

        # Tissue receives from blood
        m_rec_hep = (
            s_give_pad[:-2, 1:-1]
            + s_give_pad[2:, 1:-1]
            + s_give_pad[1:-1, :-2]
            + s_give_pad[1:-1, 2:]
        ) * self.hep_mask

        # Blood receives from tissue
        m_rec_sin = (
            h_give_pad[:-2, 1:-1]
            + h_give_pad[2:, 1:-1]
            + h_give_pad[1:-1, :-2]
            + h_give_pad[1:-1, 2:]
        ) * self.sin_mask

        # update concentrations: current mass - given + received
        new_mass_sin = (
            current_sin_mass - np.where(hep_nbrs > 0, mass_leaving_sin, 0) + m_rec_sin
        )
        new_mass_hep = (
            current_hep_mass - np.where(sin_nbrs > 0, mass_leaving_hep, 0) + m_rec_hep
        )

        C_sin = new_mass_sin / config.V_PIXEL
        C_hep = new_mass_hep / config.V_PIXEL

        # average concentration in each hepatocyte block
        hep_sums = np.bincount(self.hep_labels.ravel(), weights=C_hep.ravel())

        alive_weights = self.hep_mask.astype(float).ravel()
        hep_cnts = np.bincount(self.hep_labels.ravel(), weights=alive_weights)
        hep_avgs = np.divide(
            hep_sums, hep_cnts, out=np.zeros_like(hep_sums), where=hep_cnts > 0
        )

        C_hep = hep_avgs[self.hep_labels] * self.hep_mask

        return C_sin, C_hep

    # ── Mass Audit & Diagnostics ──────────────────────────────────────────────

    def get_total_mass(self):
        m_s = np.sum(self.C * self.sin_mask * config.V_PIXEL)
        m_h = np.sum(self.C * self.hep_mask * config.V_PIXEL)
        return m_s + m_h

    def audit_mass(self, step_num=0):
        grid_m = self.get_total_mass()
        total_m = (grid_m * config.N_SINUSOIDS) + self.total_mass_exited

        # DOSE is already in µmol, so we compare directly. No need for Diff_2 anymore!
        diff = total_m - config.DOSE

        print(
            f"=== STEP {step_num} | Total Mass: {total_m:.6e} µmol | "
            f"Expected: {config.DOSE:.6e} µmol | Diff: {diff:.6e} ==="
        )

    def audit_mass2(self, step_num=0):
        """Prints a strict accounting of every molecule in the simulation."""
        grid_mass = self.get_total_mass()
        exited_mass = self.total_mass_exited
        current_total = grid_mass + exited_mass + self.total_mass_metab

        leak = current_total - self.lobule_dose

        print(f"\n=== STEP {step_num} MASS AUDIT ===")
        print(f"Grid Mass (1 Lobule): {grid_mass:.6e}")
        print(f"Exited Mass (Total):   {exited_mass:.6e}")
        print(f"Total System Mass:    {current_total:.6e}")
        print(f"Expected (DOSE):      {self.lobule_dose:.6e}")
        print(f"Metabolized Mass: {self.total_mass_metab:.6e}")

        if abs(leak) > 1e-10:
            print(f"⚠️ MASS LEAK DETECTED: {leak:.6e}")
        else:
            print(f"✅ Mass Conserved. (Diff: {leak:.6e})")
        print("============================\n")

    def _init_concentration(self):
        C = np.zeros(self.physio_grid.shape)
        C[self.inlet_pos] = self.lobule_dose / config.V_PIXEL
        return C

    def record(self, dt=None, save_frame=False):
        step_dt = dt if dt is not None else config.DT
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

    def get_toxicity_zone_means(self):
        out = {}
        for z in (1, 2, 3):
            mask = (self.zonation == z) & self.hep_mask
            if mask.sum() == 0:
                out[z] = 0.0
            else:
                out[z] = self.toxicity_field[mask].mean()
        return out
