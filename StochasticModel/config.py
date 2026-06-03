class Config:
    """Configuration for the LobuleQuadrantABM model and related analyses."""

    DOSE = 26450  # umol -> 4g APAP

    GRID_N = 51  # number of pixels along one edge of the quadrant grid
    HEPA_SIZE = 8  # hepatocyte block size in pixels
    SIN_SIZE = 2  # sinusoid channel width in pixels

    LOBULE_SIZE = 750e-6  # physical lobule side length (m)
    D_SINUSOID = 10e-6  # L - sinusoid diameter (m)

    @property
    def N_PIXELS(self):
        """Calculates total pixels dynamically based on grid dimensions."""
        total_pixels = 0
        for i in range(self.GRID_N):
            if i % 2 == 0:
                # Boundary sinusoids are 1px, inner sinusoids are SIN_SIZE
                total_pixels += 1 if i in (0, self.GRID_N - 1) else self.SIN_SIZE
            else:
                total_pixels += self.HEPA_SIZE
        return total_pixels

    @property
    def PIXEL_WIDTH(self):
        """Calculates pixel width dynamically."""
        return self.LOBULE_SIZE / self.N_PIXELS

    @property
    def V_PIXEL(self):
        """Calculates volume represented by one pixel dynamically."""
        return self.PIXEL_WIDTH * self.PIXEL_WIDTH * self.D_SINUSOID * 1000

    BASE_UPTAKE_PCT = 0.216  # drug uptake rate from blood to tissue
    BASE_EFFLUX_PCT = 0.216 / 2.7  # drug efflux rate from tissue to blood
    INLET_SLOWDOWN_FACTOR = 0.15  # factor to slow down drug movement at the inlet to simulate physiological flow resistance

    FLUX_PCT = 0.9  # mean percentage of drug that moves during transport updates (with some randomness)

    CLEARANCE_RATE = 6.9e-5  # ln(2) / 10s half-life for drug clearance from hepatocytes
    DT = 0.001  # timestep (s)

    TOXICITY_THRESHOLD = 0.5
    NAPQI_FRACTION = 0.10
    ZONATION_MULT_ZONE1 = 0.5
    ZONATION_MULT_ZONE2 = 1.0
    ZONATION_MULT_ZONE3 = 3

    def __call__(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError as exc:
            raise ValueError(f"No such config attribute: {attr}") from exc
