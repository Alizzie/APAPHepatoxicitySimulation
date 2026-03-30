class Config:
    # ── GRID ──────────────────────────────────────────────────────────────────────
    GRID_N = 1  # number of lobules per side
    CYCTES_N = 16  # hepatocytes per lobule side
    CYCTES_PX = 10  # hepatocyte size (px)
    MARGIN = 22  # canvas margin
    PT_R = 5  # portal triad radius (px)
    CV_R = 7  # central vein radius (px)
    SOURCES_NR = 3  # number of sinusoid sources per lobule edge
    BRANCH_PROB = 0.15  # probability of sinusoid branching at each

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
