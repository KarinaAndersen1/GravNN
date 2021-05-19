        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    
    df_file = "Data/Dataframes/sh_stats_Brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    C55_r0_gm = SphericalHarmonics(model_file, degree=55, trajectory=trajectory)
    C55_a = C55_r0_gm.load().accelerations

    C110_r0_gm = SphericalHarmonics(model_file, degree=110, trajectory=trajectory)
    C110_a = C110_r0_gm.load().accelerations

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations
        

    


    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page

    vlim= [0, 30]
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    map_vis.save(plt.gcf(), directory + "sh_brillouin_true_map.pdf")


    # vlim= [0, 30]
    # grid_true = Grid(trajectory=trajectory, accelerations=C55_a-C22_a)
    # map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    # map_vis.save(plt.gcf(), directory + "sh_brillouin_55_map.pdf")


    # grid_true = Grid(trajectory=trajectory, accelerations=C110_a-C22_a)
    # map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    # map_vis.save(plt.gcf(), directory + "sh_brillouin_110_map.pdf")


    plt.show()
if __name__ == "__main__":
    main()
