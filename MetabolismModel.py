import numpy as np
from config import Config

config = Config()


class MetabolismModel:
    """
    Implements the intracellular APAP metabolism ODE system (Eq. 3-7)
    from Chalhoub et al., coupled to the LobuleQuadrant transport model.

    State variables per hepatocyte pixel:
        P  — APAP concentration                (replaces C_hep)
        Sulfate  — sulfate cofactor concentration
        GSH  — GSH (glutathione) concentration
        NAPQI  — NAPQI concentration
        Ci — protein adduct concentration      (hepatotoxicity marker)

    Zones (portal → central, along inlet→outlet diagonal):
        Zone 1 (periportal)   : first  n_z1 hepatocytes
        Zone 2 (midzonal)     : next   n_z2 hepatocytes
        Zone 3 (pericentral)  : last   n_z3 hepatocytes  ← most toxic
    """

    def __init__(self, physio_grid, hep_labels, inlet_pos, outlet_pos):
        self.physio_grid = physio_grid
        self.hep_labels = hep_labels
        self.hep_mask = physio_grid == 0
        self.sin_mask = physio_grid == 1

        self.inlet_pos = inlet_pos
        self.outlet_pos = outlet_pos

        self.zone_map = self._build_zone_map()

        self.k450_map = self._build_k450_map()

        shape = physio_grid.shape
        self.P = np.zeros(shape)
        self.Sulfate = np.full(shape, config.S_INIT) * self.hep_mask
        self.GSH = np.full(shape, config.G_INIT) * self.hep_mask
        self.NAPQI = np.zeros(shape)
        self.Ci = np.zeros(shape)

        self.zone_toxicity_history = {1: [], 2: [], 3: []}
        self.zone_P_history = {1: [], 2: [], 3: []}
        self.zone_N_history = {1: [], 2: [], 3: []}
        self.zone_G_history = {1: [], 2: [], 3: []}

    # ── Zone construction ─────────────────────────────────────────────────────

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

    def _build_k450_map(self):
        """
        Zone-specific CYP450 rate constant map.
        Zone 3 (pericentral) has highest CYP450 expression.
        """

        k450_map = np.zeros(self.physio_grid.shape)
        k450_map[self.zone_map == 1] = config.K_450_ZONE1
        k450_map[self.zone_map == 2] = config.K_450_ZONE2
        k450_map[self.zone_map == 3] = config.K_450_ZONE3
        return k450_map

    # ── Metabolism step ─────────────────────────────────────────────────────
    def step(self):
        """
        Integrate metabolism ODES for one time step using explicit Euler method.
        The equations are from paper 2 (Eq. 3-7).

        Returns:
            P: ndarray
                Updated hepatocyte APAP concentration grid (µM) = new C_hep
        """

        P = self.P
        Sulfate = self.Sulfate
        GSH = self.GSH
        NAPQI = self.NAPQI
        Ci = self.Ci
        k450_map = self.k450_map
        hepa_mask = self.hep_mask

        # ══════════════════════════════════════════════════════════════════
        # Eq. 3 — dP/dt: APAP inside hepatocyte
        #
        #   dP/dt = - k_S * S * P          (sulfation)
        #           - k_G * P              (glucuronidation)
        #           - k_450 * P            (CYP450 → NAPQI)
        #           + k_N * N              (NAPQI back-reaction, minor)
        #           - efflux               (return to sinusoid, already in flux)
        #           + influx               (uptake from sinusoid, already in flux)
        # ══════════════════════════════════════════════════════════════════
        dP_dt = (
            -config.K_S * Sulfate * P
            - config.K_G * P
            - k450_map * P
            + config.K_N * NAPQI
        ) * hepa_mask

        # ══════════════════════════════════════════════════════════════════
        # Eq. 4 — dS/dt: sulfate cofactor
        #
        #   dS/dt = - k_S * S * P          (consumed by sulfation)
        #           + b_s                  (baseline production)
        #           - d_s * S              (natural decay)
        # ══════════════════════════════════════════════════════════════════
        dS_dt = (
            -config.K_S * Sulfate * P + config.BS - config.DS * Sulfate
        ) * hepa_mask

        # ══════════════════════════════════════════════════════════════════
        # Eq. 5 — dN/dt: NAPQI (toxic intermediate)
        #
        #   dN/dt = + k_450 * P            (produced from APAP by CYP450)
        #           - k_N * N              (back-reaction to APAP)
        #           - k_GSH * N * G        (detoxified by GSH)
        #           - k_PSH * N            (binds proteins → adducts)
        # ══════════════════════════════════════════════════════════════════
        dN_dt = (
            +k450_map * P
            - config.K_N * NAPQI
            - config.K_GSH * NAPQI * GSH
            - config.K_PSH * NAPQI
        ) * hepa_mask

        # ══════════════════════════════════════════════════════════════════
        # Eq. 6 — dG/dt: GSH (glutathione, detoxifier)
        #
        #   dG/dt = - k_GSH * N * G        (consumed detoxifying NAPQI)
        #           + b_G                  (baseline synthesis)
        #           - d_G * G              (natural decay)
        # ══════════════════════════════════════════════════════════════════
        dG_dt = (-config.K_GSH * NAPQI * GSH + config.BG - config.DG * GSH) * hepa_mask

        # ══════════════════════════════════════════════════════════════════
        # Eq. 7 — dCi/dt: protein adducts (irreversible damage marker)
        #
        #   dCi/dt = k_PSH * N             (NAPQI binding to proteins)
        # ══════════════════════════════════════════════════════════════════
        dCi_dt = (config.K_PSH * NAPQI) * hepa_mask

        # ── Explicit Euler integration ────────────────────────────────────
        self.P = np.maximum(P + config.DT * dP_dt, 0.0) * hepa_mask
        self.Sulfate = np.maximum(Sulfate + config.DT * dS_dt, 0.0) * hepa_mask
        self.GSH = np.maximum(GSH + config.DT * dG_dt, 0.0) * hepa_mask
        self.NAPQI = np.maximum(NAPQI + config.DT * dN_dt, 0.0) * hepa_mask
        self.Ci = np.maximum(Ci + config.DT * dCi_dt, 0.0) * hepa_mask

        return self.P

    # -- Zone diagnostics ────────────────────────────────────────────────────
    def get_zone_means(self):
        """
        Returns mean concentration of P, NAPQI, GSH, and Ci for each zone.
        Used to track zone-specific toxicity progression over time.
        """

        out = {}
        for z in (1, 2, 3):
            mask = (self.zone_map == z) & self.hep_mask

            n_px = mask.sum()
            if n_px == 0:
                out[z] = {"P": 0, "NAPQI": 0, "GSH": 0, "Ci": 0}
            else:
                out[z] = {
                    "P": self.P[mask].mean(),
                    "NAPQI": self.NAPQI[mask].mean(),
                    "GSH": self.GSH[mask].mean(),
                    "Ci": self.Ci[mask].mean(),
                }
        return out

    def record(self):
        """Append current zone means to history for plotting later."""
        means = self.get_zone_means()
        for z in (1, 2, 3):
            self.zone_P_history[z].append(means[z]["P"])
            self.zone_N_history[z].append(means[z]["NAPQI"])
            self.zone_G_history[z].append(means[z]["GSH"])
            self.zone_toxicity_history[z].append(means[z]["Ci"])

    def get_toxicity_field(self):
        """Return the full 2D protein adduct field for visualization"""
        return self.Ci
