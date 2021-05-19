        
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
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Data import standardize_output


def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapVisualization('mGal')
    map_vis.fig_size = map_vis.full_page
    #map_vis.tick_interval = [60, 60]

    my_cmap = 'viridis'
    vlim= [0, 30]

    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180

    df_file ='C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\pinn_df.data'
    df = pd.read_pickle(df_file)

    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    
    for i in range(len(df)):
        row = df.iloc[i]
        model_id = row['id']
        config, model = load_config_and_model(model_id, df)

        x_transformer = config['x_transformer'][0]
        a_transformer = config['a_transformer'][0]

        x = x_transformer.transform(trajectory.positions)
        output = model.predict(x)
        U, a, lap, curl = standardize_output(output, config)

        a_pred = a_transformer.inverse_transform(a)
        grid_true = Grid(trajectory=trajectory, accelerations=a_pred)
        map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
        map_vis.save(plt.gcf(), directory + "pinn_brillouin_" + str(row['num_units']) + ".pdf")


    plt.show()
if __name__ == "__main__":
    main()
