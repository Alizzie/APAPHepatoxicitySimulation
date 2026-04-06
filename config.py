from curses.ascii import BS


class Config:

    # ── Unit conversions ──────────────────────────────────────────────────────
    _per_day = 1 / 86400  # 1/d        → 1/s
    _L_per_mol_per_day = 1e-3 / 86400  # L/mol/d    → m³/mol/s
    _mol_per_L_per_day = 1e3 / 86400  # mol/L/d    → mol/m³/s
    _uL_per_day = 1e-9 / 86400  # µL/d       → m³/s

    # ── Grid geometry ─────────────────────────────────────────────────────────
    GRID_N = 51  # checkerboard dimension (number of cells per side)
    HEPA_SIZE = 8  # hepatocyte block size in pixels
    SIN_SIZE = 2  # sinusoid channel width in pixels
    LOBULE_SIZE = 750e-6  # physical lobule side length (m)
    ZONATION = 8  # hepatocytes per zone (zone 3 gets remainder)

    # Derived pixel count and size
    # Expansion: 26 sinusoid rows (1 or 2px) + 25 hepatocyte rows (8px) = 250px
    N_PIXELS = 250
    DX = LOBULE_SIZE / N_PIXELS  # 3.0e-6 m per pixel

    # ── Darcy flow ────────────────────────────────────────────────────────────
    BLOOD_VISCOSITY = 0.0035  # Pa·s
    P_INLET = 103000  # Pa
    P_OUTLET = 101800  # Pa
    K_SIN = 1.123e-12  # sinusoid permeability (m²)
    K_HEPA = 7.35e-14  # hepatocyte permeability (m²)

    # ── Transport ─────────────────────────────────────────────────────────────
    U_X = 1e-4  # blood velocity in sinusoids (m/s)  Table 1
    D_SIN = 2.22e-10  # sinusoid diffusion coefficient (m²/s)  Table 1

    # ── Physical volumes in Liters ─────────────────────────
    D_SINUSOID = 10e-6  # L - sinusoid diameter (m)
    N_SINUSOIDS = 5.23e9  # L -number of sinusoids in whole liver
    V_SINUSOID = 2.89e-11  # L
    V_HEPATOCYTE = 3.4e-12  # L
    V_BLOOD = 5.7  #  L

    # Pixel volume — sinusoid depth assumed = D_SINUSOID
    V_PIXEL = DX * DX * D_SINUSOID * 1000  # L

    # ── Sinusoid ↔ hepatocyte exchange ──────────────────────────────
    # Step 1: macroscopic clearance from paper (µL/d/sinusoid → m³/s)
    _CL_INFLUX_MACRO = 1.65 * _uL_per_day  # 1.909e-14 m³/s
    _CL_EFFLUX_MACRO = 0.603 * _uL_per_day  # 6.979e-15 m³/s

    # Step 2: intrinsic rate constants (divide by macroscopic volume)
    _RATE_INFLUX = _CL_INFLUX_MACRO / (V_SINUSOID * 1e-03)  # ~0.66  s⁻¹
    _RATE_EFFLUX = _CL_EFFLUX_MACRO / (V_HEPATOCYTE * 1e-03)  # ~2.05  s⁻¹

    # Step 3: per-pixel clearance (multiply by pixel volume)
    CL_INFLUX = _RATE_INFLUX * V_PIXEL  # m³/s per pixel
    CL_EFFLUX = _RATE_EFFLUX * V_PIXEL  # m³/s per pixel

    F_UNBOUND = 0.75  # unbound fraction of APAP in plasma (dimensionless)

    # ── Simulation ────────────────────────────────────────────────────────────
    DT = 1e-4  # timestep (s)
    DOSE = 6610.0  # umol - Total initial drug mass administered to the system

    # ── Metabolism — Chalhoub et al. Table 1 ─────────────────────────────────
    # GSH turnover
    DG = 2 * _per_day  # 2.315e-5  s⁻¹        natural GSH decay
    BG = (
        4.0412e-4 * _mol_per_L_per_day * 1e6
    )  # uM/s  GSH production (converted from mol/m³/s)
    K_GSH = 5.44e7 * _L_per_mol_per_day * 1e-6  # µM⁻¹s⁻¹ GSH-NAPQI reaction

    # Glucuronidation
    K_G = 2.99 * _per_day  # 3.461e-5  s⁻¹

    # Sulfation
    K_S = 7.684e3 * _L_per_mol_per_day * 1e-6  # µM⁻¹s⁻¹  sulfation
    BS = 7.7941e-4 * _mol_per_L_per_day * 1e6  # uM/s   sulfate production
    DS = 2 * _per_day  # 2.315e-5  s⁻¹        sulfate decay

    # CYP450 → NAPQI  (base rate, zone-specific multipliers applied below)
    K_450 = 0.315 * _per_day  # 3.646e-6  s⁻¹

    # NAPQI kinetics
    K_N = 0.0315 * _per_day  # 3.646e-7  s⁻¹   back-reaction
    K_PSH = 100 * _per_day  # 1.157e-3  s⁻¹   protein binding

    # ── Zonation — CYP450 gradient (periportal → pericentral) ────────────────
    K_450_ZONE1 = K_450 * 1.0  # zone 1 periportal   (baseline)
    K_450_ZONE2 = K_450 * 2.0  # zone 2 midzonal
    K_450_ZONE3 = K_450 * 4.0  # zone 3 pericentral  (4× highest CYP450)

    # ── Steady-state initial conditions ──────────────────────────────────────
    S_INIT = BS / DS  # µM — sulfate at equilibrium
    G_INIT = BG / DG  # µM — GSH at equilibrium

    # ── Colours ───────────────────────────────────────────────────────────────
    BG_COL = "#0d1117"
    LOBULE_C = "#161b22"
    LOBULE_B = "#21262d"
    PT_COL = "#f78166"
    CV_COL = "#58a6ff"
    FLOW_COL = "#ffa657"
    DIFF_COL = "#3fb950"
    TEXT_COL = "#e6edf3"
    ACCENT = "#d2a8ff"

    def __call__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError as exc:
            raise ValueError(f"No such config attribute: {attr}") from exc
