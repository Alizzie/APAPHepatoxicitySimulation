class Config:
    # ── GRID RANDOM WALK ──────────────────────────────────────────────────────────────────────
    GRID_N = 1  # number of lobules per side
    CYCTES_N = 16  # hepatocytes per lobule side
    CYCTES_PX = 10  # hepatocyte size (px)
    MARGIN = 22  # canvas margin
    PT_R = 5  # portal triad radius (px)
    CV_R = 7  # central vein radius (px)
    SOURCES_NR = 3  # number of sinusoid sources per lobule edge
    BRANCH_PROB = 0.15  # probability of sinusoid branching at each

    # ── GRID SQUARED LATTICE ─────────────────────────────────────────────────────────────
    GRID_N = 51  # grid dimension
    HEPA_SIZE = 8  # hepatocyte size (unit)
    SIN_SIZE = 2  # sinusoid size (unit)
    SIN_BORDER = 1  # sinusoid border (unit)
    ZONATION = 8  # number of hepatocyte for zone 1, 2, zone 3 is 8 + 1

    # --- FLOW CALCULATIONS ─────────────────────────────────────────────────────────────
    BLOOD_VISCOSITY = 0.0035  # blood viscosity (Pa.s)
    P_INLET = 103000  # inlet pressure (Pa)
    P_OUTLET = 101800  # outlet pressure (Pa)
    K_SIN = 1.123e-12  # sinusoid permeability (m^2)
    K_HEPA = 7.35e-14  # hepatocyte permeability (m^2)

    # ── Simulation defaults ────────────────────────────────────────────────────────
    DT = 0.05  # time step (s)
    DECAY = 0.03  # first-order metabolic consumption rate
    INLET_CONC = 1.0  # portal triad O2 / nutrient concentration (normalised)

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
