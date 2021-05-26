        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Moon()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    
    df_file = "Data/Dataframes/sh_stats_moon_Brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations
    
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations
        
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)

    directory = os.path.abspath('.') +"/Plots/Moon/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page
    
    my_cmap = 'viridis'
    vlim= [0, 60]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)#,log_scale=True)
    map_vis.add_colorbar(im, '[mGal]', vlim, extend='max')
    map_vis.save(plt.gcf(), directory + "sh_brillouin_map.pdf")

    mapUnit = 'mGal'
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60, 60]

    map_vis.newFig()
    my_cmap = 'viridis'
    vlim= [0, 40]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)#,log_scale=True)
    map_vis.add_colorbar(im, '[mGal]', vlim, extend='max')
    map_vis.save(plt.gcf(), directory + "sh_brillouin_map_half.pdf")


    plt.show()
if __name__ == "__main__":
    main()
