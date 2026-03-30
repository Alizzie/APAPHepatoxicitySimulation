# Virtual Twin of Liver
The goal of this project is to build a simplified virtual liver that can be used for predicting hepatotoxcitity. 

We want to determine when APAP damages the liver at clinical doses, and whether that damage threshold shifts depending on who the patient is.



----
# Lattice Model
The liver is organized in hexagonal lobules with every corner being a portal triad that runs the blood vessel to the central vein positioned at the middle of the lobule. 

For the simplified version, we model the liver as this:
- Lobules are squared with 4 portal triad at each corner.
- Sinusoidals goes diagonally from the portal triad as source to the central vein, optionally with more sources along the edges, spread evenly.
- Diffusion circular to neighboring squares considering the blood permeability / Hydraulic conductivity of the sinusoidsal wall


# Equations
- 