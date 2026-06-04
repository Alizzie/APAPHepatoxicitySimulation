class Config:
    """Configuration for the PDE Lobule Model."""

    GRID_N = 51  # checkerboard dimension (number of cells per side)
    HEPA_SIZE = 8  # hepatocyte block size in pixels
    SIN_SIZE = 2  # sinusoid channel width in pixels

    DT = 0.001  # timestep (s)
    DOSE = 26450  # umol -> 4g APAP

    LOBULE_SIZE = 750e-6  # physical lobule side length (m)
    D_SINUSOID = 10e-6  # L - sinusoid diameter (m)

    @classmethod
    def N_PIXELS(cls):
        """Calculates total pixels dynamically based on grid dimensions."""
        total_pixels = 0
        for i in range(cls.GRID_N):
            if i % 2 == 0:
                # Boundary sinusoids are 1px, inner sinusoids are SIN_SIZE
                total_pixels += 1 if i in (0, cls.GRID_N - 1) else cls.SIN_SIZE
            else:
                total_pixels += cls.HEPA_SIZE
        return total_pixels

    @classmethod
    def PIXEL_WIDTH(cls):
        """Calculates pixel width dynamically."""
        return cls.LOBULE_SIZE / cls.N_PIXELS()

    @classmethod
    def V_PIXEL(cls):
        """Calculates volume represented by one pixel dynamically."""
        return cls.PIXEL_WIDTH() * cls.PIXEL_WIDTH() * cls.D_SINUSOID * 1000

    U_X = 1e-4  # blood velocity in sinusoids (m/s) (Reverted to 1e-4 for stability)
    D_SIN = 2.22e-10  # sinusoid diffusion coefficient (m²/s)

    # ── Unit conversions ──────────────────────────────────────────────────────
    _per_day = 1 / 86400  # 1/d        → 1/s
    _uL_per_day = 1e-9 / 86400  # µL/d       → m³/s
    _per_day_to_per_s = 1 / 86400  # Simple time conversion

    # ── Sinusoid ↔ hepatocyte exchange ──────────────────────────────
    V_SINUSOID = 2.89e-11  # L
    V_HEPATOCYTE = 3.4e-12  # L
    # Step 1: macroscopic clearance from paper (µL/d/sinusoid → m³/s)
    _CL_INFLUX_MACRO = 1.65 * _uL_per_day  # 1.909e-14 m³/s
    _CL_EFFLUX_MACRO = 0.603 * _uL_per_day  # 6.979e-15 m³/s

    # Step 2: intrinsic rate constants (divide by macroscopic volume)
    _RATE_INFLUX = _CL_INFLUX_MACRO / (V_SINUSOID * 1e-03)  # ~0.66  s⁻¹
    _RATE_EFFLUX = _CL_EFFLUX_MACRO / (V_HEPATOCYTE * 1e-03)  # ~2.05  s⁻¹

    # Step 3: per-pixel clearance (multiply by pixel volume)
    @classmethod
    def CL_INFLUX(cls):
        return cls._RATE_INFLUX * cls.V_PIXEL()

    @classmethod
    def CL_EFFLUX(cls):
        return cls._RATE_EFFLUX * cls.V_PIXEL()

    F_UNBOUND = 0.75  # unbound fraction of APAP in plasma (dimensionless)

    # ── Metabolism — Chalhoub et al. Table 1 (Corrected Units) ───────────────

    # GSH turnover
    DG = 2 * _per_day_to_per_s  # ~2.315e-5 s⁻¹
    BG = 4.0412e-4 * _per_day_to_per_s * 1e6  # ~0.00467 µM/s
    K_GSH = 5.44e7 * _per_day_to_per_s * 1e-6  # ~6.29e-4 µM⁻¹s⁻¹

    # Glucuronidation
    K_G = 2.99 * _per_day_to_per_s  # ~3.461e-5 s⁻¹

    # Sulfation
    K_S = 7.684e3 * _per_day_to_per_s * 1e-6  # ~8.89e-8 µM⁻¹s⁻¹
    BS = 7.7941e-4 * _per_day_to_per_s * 1e6  # ~0.00902 µM/s
    DS = 2 * _per_day_to_per_s  # ~2.315e-5 s⁻¹

    # CYP450 → NAPQI  (base rate, zone-specific multipliers applied below)
    K_450 = 0.315 * _per_day_to_per_s  # ~3.646e-6 s⁻¹

    # NAPQI kinetics
    K_N = 0.0315 * _per_day_to_per_s  # ~3.646e-7 s⁻¹
    K_PSH = 100 * _per_day_to_per_s  # ~1.157e-3 s⁻¹

    # ── Zonation — CYP450 gradient (periportal → pericentral) ────────────────
    K_450_ZONE1 = K_450 * 1.0  # zone 1 * 2 periportal   (baseline)
    K_450_ZONE2 = K_450 * 2  # zone 2 midlobular
    K_450_ZONE3 = K_450 * 5  # zone 3 pericentral  (1.3× highest CYP450)

    # ── Toxicity thresholds ─────────────────────────────────────────────────
    TOXI_THRESHOLD = 1.0  # µM Ci threshold for hepatocyte death

    # ── Steady-state initial conditions ──────────────────────────────────────
    S_INIT = BS / DS  # µM — sulfate at equilibrium
    G_INIT = BG / DG  # µM — GSH at equilibrium

    def __call__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError as exc:
            raise ValueError(f"No such config attribute: {attr}") from exc
