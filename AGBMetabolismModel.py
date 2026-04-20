import numpy as np


class AgentBasedMetabolism:
    def __init__(self, physio_grid, zone_map):
        self.hep_mask = physio_grid == 0
        self.zone_map = zone_map

        self.APAP_packets = np.zeros_like(physio_grid)
        self.NAPQI_packets = np.zeros_like(physio_grid)

        self.GSH_S_pool = np.full(physio_grid.shape, 50) * self.hep_mask
        self.damage_points = np.zeros_like(physio_grid)
        self.is_alive = np.ones_like(physio_grid, dtype=bool) * self.hep_mask

    def step(self):
        active_cells = self.hep_mask & self.is_alive

        # RULE 1: Probabilistic CYP450 Metabolism
        cyp_prob = np.zeros_like(self.APAP_packets, dtype=float)
        cyp_prob[self.zone_map == 1] = 0.05  # 5% chance per step
        cyp_prob[self.zone_map == 2] = 0.10  # 10% chance per step
        cyp_prob[self.zone_map == 3] = 0.20  # 20% chance per step

        new_napqi = (
            np.random.binomial(self.APAP_packets.astype(int), cyp_prob) * active_cells
        )

        self.APAP_packets -= new_napqi
        self.NAPQI_packets += new_napqi

        # RULE 2: Detoxification (GSH neutralizes NAPQI 1-to-1)
        can_detox = (self.NAPQI_packets > 0) & (self.GSH_S_pool > 0) & active_cells

        detox_amount = np.minimum(self.NAPQI_packets, self.GSH_S_pool) * can_detox
        self.NAPQI_packets -= detox_amount
        self.GSH_S_pool -= detox_amount

        # RULE 3: Toxicity (Damage is taken if NAPQI exists but GSH is empty)
        taking_damage = (self.NAPQI_packets > 0) & (self.GSH_S_pool == 0) & active_cells
        self.damage_points += self.NAPQI_packets * taking_damage
        self.NAPQI_packets[taking_damage] = 0

        # RULE 4: Cell Death (Necrosis)
        death_threshold = 100
        just_died = (self.damage_points >= death_threshold) & self.is_alive

        # Change the state of the agent to DEAD
        self.is_alive[just_died] = False
