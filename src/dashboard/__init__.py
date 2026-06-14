"""Local Streamlit dashboard for the HZO Bayesian-optimization project.

A 'fancier notebook' (no notebook): browse past runs/data, train/view the GP phase
map + 3D surface, render the HfO2 crystal structures, and run experiments (train +
suggest next points). All heavy lifting is delegated to the existing packages
(``visualization``, ``models``, ``optimization``, ``preprocessing``) — this package is
only the presentation layer.
"""
