import numpy as np
from config import Config
from LobuleQuadrant import LobuleQuadrant

config = Config()

# ══════════════════════════════════════════════════════════════════════════════
# ── FullLobule ────────────────────────════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════


class FullLobule:
    """
    Complete liver lobule assembled from four LobuleQuadrant instances.
    Portal triads at the four outer corners; central vein at the centre
    of the assembled 2N x 2N grid.

    Parameters
    ----------
    grid_size : int
        Lattice size passed to each quadrant (default config.GRID_N).
    """

    DIRS = ["top-left", "top-right", "bottom-left", "bottom-right"]

    def __init__(self, grid_size: int = config.GRID_N):
        self.quadrants = {d: LobuleQuadrant(d, grid_size) for d in self.DIRS}
        for q in self.quadrants.values():
            q.check_cfl()  # sanity check on initial velocities

    # ── Helpers ───────────────────────────────────────────────────────────────

    def assemble(self, attr):
        """Stack a per-quadrant array attribute into the full 2N x 2N matrix."""
        q = self.quadrants
        return np.block(
            [
                [getattr(q["top-left"], attr), getattr(q["top-right"], attr)],
                [getattr(q["bottom-left"], attr), getattr(q["bottom-right"], attr)],
            ]
        )

    def _split(self, C_full):
        q = C_full.shape[0] // 2
        self.quadrants["top-left"].C = C_full[:q, :q]
        self.quadrants["top-right"].C = C_full[:q, q:]
        self.quadrants["bottom-left"].C = C_full[q:, :q]
        self.quadrants["bottom-right"].C = C_full[q:, q:]

    # ── Time step ─────────────────────────────────────────────────────────────

    def compute_flux(self):
        """
        Advance each quadrant independently (they are physically decoupled).
        """
        C_blocks = {}
        for d in self.DIRS:
            C_blocks[d] = self.quadrants[d].compute_flux()
            self.quadrants[d].record()

        return np.block(
            [
                [C_blocks["top-left"], C_blocks["top-right"]],
                [C_blocks["bottom-left"], C_blocks["bottom-right"]],
            ]
        )
