from LobuleQuadrant import LobuleQuadrant
from config import Config

config = Config()


def calculate_dose_in_blood(dose: float) -> float:
    return dose / config.V_BLOOD / 4  # Divide by 4 for quarter-lobule model


if __name__ == "__main__":
    # config.Dose in umol, V_BLOOD in L → concentration in µmol/L (µM)
    dose_in_blood = calculate_dose_in_blood(dose=config.DOSE)
    print(f"Initial dose in blood: {dose_in_blood:.3e} µM")

    # Create a quadrant instance
    quadrant = LobuleQuadrant(grid_size=10, dose=dose_in_blood)
