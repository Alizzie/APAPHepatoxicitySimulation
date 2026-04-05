class Config:
    # ── GRID SQUARED LATTICE ─────────────────────────────────────────────────────────────
    GRID_N = 51  # grid dimension
    HEPA_SIZE = 8  # hepatocyte size (unit)
    SIN_SIZE = 2  # sinusoid size (unit)
    SIN_BORDER = 1  # sinusoid border (unit)
    ZONATION = 8  # number of hepatocyte for zone 1, 2, zone 3 is 8 + 1
    LOBULE_SIZE = 750e-6  # lobule size (m)

    # --- FLOW CALCULATIONS ─────────────────────────────────────────────────────────────
    BLOOD_VISCOSITY = 0.0035  # blood viscosity (Pa.s)
    P_INLET = 103000  # inlet pressure (Pa)
    P_OUTLET = 101800  # outlet pressure (Pa)
    K_SIN = 1.123e-12  # sinusoid permeability (m^2)
    K_HEPA = 7.35e-14  # hepatocyte permeability (m^2)

    # ── DIFFUSION (Rezania et al. Table 2, converted cm²/min → m²/s) ─────────────────
    D_SIN = 2.5e-4 / 6000  # cm²/min → m²/s = 4.167e-8 m²/s
    D_HEPA = 2.5e-5 / 6000  # cm²/min → m²/s = 4.167e-9 m²/s
    U_X = 1e-4  # # blood flow velocity in sinusoids (m/s), from APAP paper Table 1
    V_BLOOD = 5.7e-3  # blood volume in a liver (m³), from APAP paper Table 1
    DOSE = 6610.0  # drug dose (uM)
    D_SINUSOID = 10e-6  # sinusoid diameter (m)
    N_SINUSOIDS = 5.23e9  # number of sinusoids per lobule

    # ── SINUSOID ↔ HEPATOCYTE EXCHANGE (Scaled to Pixel) ─────────────────────────
    # 1 µL/d = 1e-9 m³ / 86400 s = 1.157e-14 m³/s
    _uL_per_day_to_m3_per_s = 1e-9 / 86400

    # 1. Calculate the exact physical volume of a single 2D grid pixel
    N_PIXELS = 250  # Based on GRID_N=51 expansion (25*8 + 24*2 + 2)
    DX = LOBULE_SIZE / N_PIXELS  # 3.0e-6 m
    V_PIXEL = DX * DX * D_SINUSOID  # 9.0e-17 m³

    # Apply the pixel volume so the grid mass doesn't artificially inflate
    V_SIN = V_PIXEL
    V_HEPAT = V_PIXEL

    # 2. Extract the intrinsic rate constants (1/s) from the paper's macroscopic data
    _MACRO_V_SIN = 2.89e-11 * 1e-3  # 2.89e-14 m³
    _MACRO_V_HEP = 3.4e-12 * 1e-3  # 3.40e-15 m³

    _MACRO_CL_INFLUX = 1.65 * _uL_per_day_to_m3_per_s
    _MACRO_CL_EFFLUX = 0.603 * _uL_per_day_to_m3_per_s

    _RATE_INFLUX = _MACRO_CL_INFLUX / _MACRO_V_SIN  # ~0.66 s⁻¹
    _RATE_EFFLUX = _MACRO_CL_EFFLUX / _MACRO_V_HEP  # ~2.05 s⁻¹

    # 3. Apply those rates to the pixel volumes to get Per-Pixel Clearance
    CL_INFLUX = _RATE_INFLUX * V_SINUSOID
    CL_EFFLUX = _RATE_EFFLUX * V_HEPATOCYTE

    F_UNBOUND = 0.75  # unbound fraction (dimensionless)

    # ── Simulation defaults ────────────────────────────────────────────────────────
    DT = 2.5e-5  # time step (s)
    DECAY = 0.03  # first-order metabolic consumption rate
    INLET_CONC = 1000  # drug concentration at inlet (uM)
    # Volumes from APAP paper Table 1
    # 1 L = 1e-3 m³
    V_SINUSOID = 2.89e-11 * 1e-3  # 2.89e-14 m³
    V_HEPATOCYTE = 3.4e-12 * 1e-3  # 3.4e-15  m³

    # ── Simulation defaults ────────────────────────────────────────────────────────
    DT = 2.5e-5  # time step (s)
    DECAY = 0.03  # first-order metabolic consumption rate
    INLET_CONC = 1000  # drug concentration at inlet (uM)

    # ── Colours ───────────────────────────────────────────────────────────────────
    BG = "#0d1117"
    LOBULE_C = "#161b22"
    LOBULE_B = "#21262d"
    PT_COL = "#f78166"  # portal triad  (arterial red-orange)
    CV_COL = "#58a6ff"  # central vein  (venous blue)
    FLOW_COL = "#ffa657"  # flow arrow
    DIFF_COL = "#3fb950"  # diffusion link
    TEXT_COL = "#e6edf3"
    ACCENT = "#d2a8ff"

    def __call__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError as exc:
            raise ValueError(f"No such config attribute: {attr}") from exc
