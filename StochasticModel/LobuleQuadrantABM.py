"""LobuleQuadrantABM.py"""

import numpy as np
from scipy.ndimage import label
from config import Config
from random import seed

seed(10)
config = Config()


class LobuleQuadrant:
    """
    Spatiotemporal model of a liver lobule quadrant.
    Each quadrant is a checkered grid of sinusoid (1) and hepatocyte (0) pixels.
    Drug mass is tracked in each pixel, with diffusion-like transport through the sinusoids and exchange between sinusoids and hepatocytes.
    Hepatocytes metabolize the drug, producing a toxic metabolite (NAPQI) that accumulates in a toxicity field. If the toxicity exceeds a threshold, the cell dies and its mass is lost to necrosis.

    The model includes zonation, with hepatocytes in different zones having different metabolic rates. The inlet is at the top-left corner and the outlet is at the bottom-right corner, creating a flow path through the lobule.
    Mass conservation is strictly tracked, with detailed audits at each step to ensure that all mass is accounted for, including drug in the grid, drug that has exited, drug metabolized, and drug lost to necrosis.

    The class provides methods to compute the flux of drug mass through the system, update metabolism and exchange, and record the history of mass distribution for analysis and visualization.

    Note: This is a stochastic model, so results will vary between runs. The random seed is set for reproducibility in testing.
    """

    def __init__(
        self,
        dose: float = config.DOSE,
        allow_hepa_exchange: bool = True,
        base_uptake_pct: float = config.BASE_UPTAKE_PCT,
        base_efflux_pct: float = config.BASE_EFFLUX_PCT,
    ):
        # Init parameters and state variables
        self.lobule_dose = dose
        self.allow_hepa_exchange = allow_hepa_exchange
        self.base_uptake_pct = base_uptake_pct
        self.base_efflux_pct = base_efflux_pct

        # Build the physiological grid and masks
        physio_grid = self._build_struc_matrix()
        physio_grid_size = physio_grid.shape[0]
        self.sin_mask = physio_grid == 1
        self.hep_mask = physio_grid == 0
        self.hep_labels, self.num_heps = label(physio_grid == 0)

        self.inlet_pos = (0, 0)
        self.outlet_pos = (physio_grid_size - 1, physio_grid_size - 1)
        self.mass_grid = self._init_concentration(physio_grid)

        # Zonation and metabolic parameters
        self.zonation, self.dist_norm = self._build_zone_map(
            physio_grid, physio_grid_size
        )

        # Toxicity tracking
        self.toxicity_field = np.zeros_like(physio_grid, dtype=float)
        self.is_cell_dead = np.zeros_like(physio_grid, dtype=bool)

        # Mass audit trackers
        self.total_mass_exited = 0.0
        self.total_mass_metab = 0.0
        self.total_mass_lost_to_necrosis = 0.0

        # Time and history tracking
        self.current_time = 0.0
        self.time_history = []
        self.exited_mass_history = []
        self.total_system_mass_history = []
        self.metabolized_mass_history = []
        self.mass_lost_to_necrosis_history = []
        self.grid_mass_history = []
        self.concentration_history = []
        self.reflux_mass = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # ── Initialization Helpers (Grid building) ─────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    def _cell_sizes(self):
        grid_size = config.GRID_N
        sizes = []
        for i in range(grid_size):
            if i % 2 == 0:
                sizes.append(1 if i in (0, grid_size - 1) else config.SIN_SIZE)
            else:
                sizes.append(config.HEPA_SIZE)
        return sizes

    def _build_struc_matrix(self):
        """Creates a checkered pattern of sinusoid (1) and hepatocyte (0) pixels, then expands each pixel into a block based on the specified sizes for sinusoids and hepatocytes."""
        grid_size = config.GRID_N
        lattice = np.zeros((grid_size, grid_size), dtype=int)
        for i in range(0, grid_size, 2):
            lattice[i, :] = 1
            lattice[:, i] = 1
        sizes = self._cell_sizes()
        expanded = np.repeat(lattice, sizes, axis=0)
        expanded = np.repeat(expanded, sizes, axis=1)
        return expanded

    def _build_zone_map(
        self, physio_grid: np.ndarray = None, physio_grid_size: int = None
    ):
        """
        Assigns zone 1, 2, or 3 to each hepatocyte pixel based on its
        Manhattan distance from the inlet corner.

        With ZONATION hepatocyctes per side and 25 total hepatocytes
        along the diagonal, zones split as 8/8/9.
        """

        rows = np.arange(physio_grid_size)
        cols = np.arange(physio_grid_size)
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

        # Use sorted labels to assign zones: closest 1/3 to inlet is Zone 1, middle 1/3 is Zone 2, farthest 1/3 is Zone 3
        label_zone = {}
        for i, label in enumerate(sorted_labels):
            if i < n_z1:
                label_zone[label] = 1
            elif i < n_z1 + n_z2:
                label_zone[label] = 2
            else:
                label_zone[label] = 3

        # Build zone map for all pixels
        zone_map = np.zeros_like(physio_grid, dtype=int)
        for label, zone in label_zone.items():
            zone_map[self.hep_labels == label] = zone

        return zone_map, dist_norm

    def _init_concentration(self, physio_grid: np.ndarray):
        m = np.zeros(physio_grid.shape)
        m[self.inlet_pos] = self.lobule_dose
        return m

    # ══════════════════════════════════════════════════════════════════════════
    # ── compute_flux
    # ══════════════════════════════════════════════════════════════════════════
    def compute_flux(self):
        """Performs one full transport + metabolism update cycle, returning the updated mass grid."""
        self.mass_grid = self._transport_update()
        m_sin = self.mass_grid * self.sin_mask
        m_hep = self.mass_grid * self.hep_mask

        if self.allow_hepa_exchange:
            m_sin, m_hep = self._hepatocyte_exchange(m_sin, m_hep)
            self.mass_grid = m_sin + m_hep

        m_sin, m_hep = self._metabolism_update(m_sin, m_hep)
        self.mass_grid = m_sin + m_hep

        return self.mass_grid

    def _transport_update(self):
        m_sin = self.mass_grid * self.sin_mask
        n = self.mass_grid.shape[0]

        # Generate all random values at once
        flux_pct = np.random.normal(config.FLUX_PCT, 0.1, (n, n))
        split_flux = np.random.normal(0.50, 0.1, (n, n))
        split_ref = np.random.normal(0.50, 0.1, (n, n))

        mass_flux = m_sin * flux_pct
        mass_ref = m_sin - mass_flux
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

        m_new = np.zeros((n, n))

        # Each direction: shifted contribution lands in the target cell
        m_new[:, 1:] += np.where(can_R[:, :-1], m_Right[:, :-1], 0)  # Right
        m_new[1:, :] += np.where(can_D[:-1, :], m_Down[:-1, :], 0)  # Down
        m_new[:, :-1] += np.where(can_L[:, 1:], m_Left[:, 1:], 0)  # Left
        m_new[:-1, :] += np.where(can_U[1:, :], m_Up[1:, :], 0)  # Up

        m_new += np.where(self.sin_mask & ~can_R, m_Right, 0)
        m_new += np.where(self.sin_mask & ~can_D, m_Down, 0)
        m_new += np.where(self.sin_mask & ~can_L, m_Left, 0)
        m_new += np.where(self.sin_mask & ~can_U, m_Up, 0)

        mass_out = m_new[self.outlet_pos]
        self.total_mass_exited += mass_out
        m_new[self.outlet_pos] = 0.0
        return m_new + (self.mass_grid * self.hep_mask)

    def _hepatocyte_exchange(self, m_sin, m_hep):
        C_sin = m_sin / config.V_PIXEL
        C_hep = m_hep / config.V_PIXEL

        # Net flux only flows DOWN the concentration gradient
        net_flux_concentration = np.maximum(
            C_sin - C_hep, 0
        )  # only sin→hep if C_sin > C_hep
        efflux_concentration = np.maximum(
            C_hep - C_sin, 0
        )  # only hep→sin if C_hep > C_sin

        # Convert back to mass flux
        mass_leaving_sin = (
            net_flux_concentration * config.V_PIXEL * self.base_uptake_pct
        )
        mass_leaving_hep = efflux_concentration * config.V_PIXEL * self.base_efflux_pct

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
        new_mass_sin = m_sin - np.where(hep_nbrs > 0, mass_leaving_sin, 0) + m_rec_sin
        new_mass_hep = m_hep - np.where(sin_nbrs > 0, mass_leaving_hep, 0) + m_rec_hep

        # average concentration in each hepatocyte block
        hep_sums = np.bincount(self.hep_labels.ravel(), weights=new_mass_hep.ravel())

        alive_weights = self.hep_mask.astype(float).ravel()
        hep_cnts = np.bincount(self.hep_labels.ravel(), weights=alive_weights)
        hep_avgs = np.divide(
            hep_sums, hep_cnts, out=np.zeros_like(hep_sums), where=hep_cnts > 0
        )

        m_hep = hep_avgs[self.hep_labels] * self.hep_mask

        return new_mass_sin, m_hep

    def _metabolism_update(self, m_sin, m_hep):
        # 1. Determine total mass to be metabolized this step
        mass_to_process = m_hep * config.CLEARANCE_RATE * self.hep_mask

        # 2. Split the processed mass: ~90% safe, ~10% toxic (NAPQI)
        mass_napqi_base = mass_to_process * config.NAPQI_FRACTION
        mass_metab_safe = mass_to_process * (1.0 - config.NAPQI_FRACTION)

        # 3. Apply Zonation to the toxic path (NAPQI)
        zonation_multiplier = np.ones_like(m_hep)
        zonation_multiplier[self.zonation == 1] = config.ZONATION_MULT_ZONE1
        zonation_multiplier[self.zonation == 2] = config.ZONATION_MULT_ZONE2
        zonation_multiplier[self.zonation == 3] = config.ZONATION_MULT_ZONE3

        mass_napqi = mass_napqi_base * zonation_multiplier

        # 4. Update the mass and audit trackers
        self.total_mass_metab += np.sum(mass_metab_safe + mass_napqi)
        m_hep -= mass_metab_safe + mass_napqi

        # 5. Accumulate Toxicity
        # Convert mass to concentration for the field
        self.toxicity_field += mass_napqi
        # Temporary debug print
        # 6. Handle Cell Death
        just_died = (
            self.toxicity_field >= config.TOXICITY_THRESHOLD
        ) & ~self.is_cell_dead

        mass_lost_this_step = np.sum(m_hep[just_died])
        self.total_mass_lost_to_necrosis += mass_lost_this_step

        m_hep[just_died] = 0.0  # Remove all mass from newly dead cells (Necrosis)
        self.is_cell_dead[just_died] = True  # Mark these cells as deads
        self.hep_mask = self.hep_mask & ~self.is_cell_dead
        return m_sin, m_hep

    # ══════════════════════════════════════════════════════════════════════════
    # ── Mass Auditing & Recording (to ensure mass conservation and track history)
    # ══════════════════════════════════════════════════════════════════════════

    def audit_mass(self, step_num=0):
        """Prints a strict accounting of every molecule in the simulation."""
        grid_mass = np.sum(self.mass_grid)
        exited_mass = self.total_mass_exited
        current_total = (
            grid_mass
            + exited_mass
            + self.total_mass_metab
            + self.total_mass_lost_to_necrosis
        )

        leak = current_total - self.lobule_dose

        print(f"\n=== STEP {step_num} MASS AUDIT ===")
        print(f"Grid Mass (1 Lobule): {grid_mass:.6e}")
        print(f"Exited Mass (Total):   {exited_mass:.6e}")
        print(f"Total System Mass:    {current_total:.6e}")
        print(f"Expected (DOSE):      {self.lobule_dose:.6e}")
        print(f"Metabolized Mass: {self.total_mass_metab:.6e}")
        print(f"Mass Lost to Necrosis: {self.total_mass_lost_to_necrosis:.6e}")

        if abs(leak) > 1e-10:
            print(f"⚠️ MASS LEAK DETECTED: {leak:.6e}")
        else:
            print(f"✅ Mass Conserved. (Diff: {leak:.6e})")
        print("============================\n")

    def record(self, dt=None, save_frame=False):
        """Records the current state of the system for history tracking and visualization."""
        step_dt = dt if dt is not None else config.DT
        self.current_time += step_dt

        self.time_history.append(self.current_time)
        self.exited_mass_history.append(self.total_mass_exited)
        self.total_system_mass_history.append(
            np.sum(self.mass_grid)
            + self.total_mass_exited
            + self.total_mass_metab
            + self.total_mass_lost_to_necrosis
        )
        self.grid_mass_history.append(np.sum(self.mass_grid))
        self.metabolized_mass_history.append(self.total_mass_metab)
        self.mass_lost_to_necrosis_history.append(self.total_mass_lost_to_necrosis)

        if save_frame:
            conc = self.mass_grid / config.V_PIXEL
            self.concentration_history.append(conc.copy())

    # ══════════════════════════════════════════════════════════════════════════
    # ── Utility Methods for Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def get_toxicity_zone_means(self):
        """Returns the mean toxicity level for each zone, calculated only over alive hepatocyte pixels in that zone."""
        out = {}
        for z in (1, 2, 3):
            mask = (self.zonation == z) & self.hep_mask
            if mask.sum() == 0:
                out[z] = 0.0
            else:
                out[z] = self.toxicity_field[mask].mean()
        return out
