# APAP Hepatotoxicity Simulation

A spatially explicit stochastic model of acetaminophen (APAP) transport, metabolism, and hepatotoxicity within a 2D liver lobule quadrant.

---

## Overview

This project simulates how APAP (paracetamol) distributes and causes liver damage at the cellular level. The liver lobule is represented as a checkerboard grid of sinusoidal blood channels and hepatocyte tissue blocks, inspired by the lattice-based framework of Rezania et al. (2013).

Two models are implemented:

- **Stochastic Model** (`StochasticModel/`) — the primary contribution. Tracks drug mass with stochastic sinusoidal transport, concentration-driven hepatocyte exchange, and NAPQI-driven necrosis.
- **PDE/ODE Model** (`PDEModel/`) — a deterministic reference framework adapting the coupled transport-metabolism approach of Franiatte et al. (2019) to 2D.

---

## Project Structure

```
├── StochasticModel/
│   ├── LobuleQuadrant.py           # Main stochastic model
│   ├── config.py                   # Model parameters
│   └── ...
├── PDEModel/
│   ├── LobuleQuadrant.py           # Deterministic PDE/ODE model
│   ├── MetabolismModel.py          # ODE metabolism system (APAP, GSH, NAPQI, sulfate, adducts)
│   ├── config_o.py                 # Model parameters
│   └── ...
├── Plotter/
│   ├── stochastic_only/            # Stochastic model analysis scripts
│   │   ├── main_plotter.py         # Run all stochastic plots in one go
│   │   ├── simulate_bioavailability_uptake.py
│   │   ├── simulate_dose_toxicity.py
│   │   ├── simulate_toxicity_clearance_analysis.py
│   │   ├── toxicity_plotter.py
│   │   ├── compartment_plotter.py
│   │   ├── spatial_concentration_gradient_plotter.py
│   │   ├── metabolized_drug_rate_plotter.py
│   │   └── plot_zonation.py
│   └── shared/                     # Shared plotting utilities
│       ├── simulate_diffusion_mass_conservation.py
│       └── plot_grid_structure.py
├── images/                         # Generated figures (auto-created)
└── README.md
```

---

## Key Features

- 2D checkerboard lobule grid (51×51 logical lattice, ~230×230 pixels)
- Hepatic zonation via Manhattan distance from inlet (Zone 1 periportal / Zone 2 midzonal / Zone 3 pericentral)
- Stochastic flux-splitting sinusoidal transport with strict mass conservation
- Concentration-driven sinusoid–hepatocyte exchange calibrated to F_H ≈ 0.80
- Zonation-dependent NAPQI production and hepatocyte necrosis
- Pulse reinjection mechanism to simulate continuous hepatic circulation
- Dose-toxicity sweep from 0.5g to 12g APAP
- Full mass audit at every timestep (grid + exited + metabolised + necrosis = dose)

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Usage

### Run all stochastic plots in one go

```bash
python  Plotter/stochastic_only/main_plotter.py
```

### Run individual analyses

```bash
# Bioavailability parameter sweep (uptake calibration to F_H ≈ 0.80)
python Plotter/stochastic_only/simulate_bioavailability_uptake.py

# Dose-toxicity analysis (0.5g to 12g APAP)
python Plotter/stochastic_only/simulate_dose_toxicity.py

# Toxicity heatmap and zone-averaged toxicity
python Plotter/stochastic_only/toxicity_plotter.py

# Compartment mass distribution over time
python Plotter/stochastic_only/compartment_plotter.py

# Zonation map
python Plotter/stochastic_only/plot_zonation.py
```

---

## Model Parameters

All parameters are defined in `StochasticModel/config.py` (stochastic model) and `PDEModel/config_o.py` (PDE/ODE model). Key values:

| Parameter | Value | Source |
|---|---|---|
| Lobule side length | 750 µm | Rezania et al. (2013) |
| Grid dimension | 51×51 | Rezania et al. (2013) |
| Dose (4g APAP) | 26,450 µmol | Franiatte et al. (2019) |
| Blood velocity | 10⁻⁴ m/s | Franiatte et al. (2019) |
| Base uptake rate | 0.2167 | Calibrated to F_H ≈ 0.80 |
| Timestep | 0.001 s | — |

---

## References

- Rezania et al. (2013). *A physiologically-based flow network model for hepatic drug elimination I: regular lattice lobule model.* Theoretical Biology and Medical Modelling, 10:52.
- Franiatte et al. (2019). *A computational model for hepatotoxicity by coupling drug transport and acetaminophen metabolism equations.* Int J Numer Meth Biomed Engng, 35:e3234.