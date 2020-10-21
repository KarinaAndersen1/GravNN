import os
import pickle
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import talos
from keras.utils.layer_utils import count_params
from numpy.random import seed
from sklearn.model_selection import train_test_split
from talos import Analyze, Deploy, Evaluate, Predict, Reporting, Restore, Scan
from tensorflow.keras.optimizers import SGD, Adadelta, Adam, Nadam, RMSprop
from tensorflow.keras.regularizers import l2

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.NN_Base import NN_Base
from GravNN.GravityModels.NNSupport.NN_hyperparam import NN_hyperparam
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors.MaxAbsTransform import MaxAbsTransform
from GravNN.Preprocessors.MinMaxTransform import MinMaxTransform
from GravNN.Preprocessors.RobustTransform import RobustTransform
from GravNN.Preprocessors.MinMaxStandardTransform  import MinMaxStandardTransform
from GravNN.Preprocessors.StandardTransform import StandardTransform
from GravNN.Support.transformations import (cart2sph,
                                     check_fix_radial_precision_errors,
                                     project_acceleration, sphere2cart)
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
import pandas as pd

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["RUNFILES_DIR"] = "/Users/johnmartin/Library/Python/3.7/share/plaidml"# plaidml might exist in different location. Look for "/usr/local/share/plaidml" and replace in above path
# os.environ["PLAIDML_NATIVE_PATH"] = "/Users/johnmartin/Library/Python/3.7/lib/libplaidml.dylib" # libplaidml.dylib might exist in different location. Look for "/usr/local/lib/libplaidml.dylib" and replace in above path

seed(1)
def main(case):
    """
    Loads all parameters tested and find the one with the lowest validation loss
    Populate a network with those parameters and retrain
    """
    print("Case: " + str(case))
    planet = Earth()

    # directory = "./Hyperparams/UniformDist/MinMaxTransform/259200/"
    # cases = ['071320150333.csv', '071320150753.csv', '071320150805.csv' , 
    #                 '071420143549.csv', '071420143558.csv', '071420143613.csv']
    # preprocessor = MinMaxTransform()
    # trajectory = UniformDist(planet, planet.radius, point_count)

    #case = 6 # moved to main at bottom
    directory = "./Hyperparams/RandomDist/MinMaxStandardTransform/259200/"
    #cases = ['072120093037.csv', '072120093056.csv', '072120093113.csv' , 
    #                '072120093251.csv', '072120093223.csv', '072120093140.csv']
    cases = ['073120092741.csv', '073020072039.csv', '073020072054.csv' , # Cases from 7/29 - 7/30
                    '072920140740.csv', '072920140722.csv', '072920140702.csv']
    preprocessor = MinMaxStandardTransform()
    point_count = 259200 
    #point_count = 720*720*2
    
    trajectory = RandomDist(planet, [planet.radius, planet.radius+5000.0], point_count)
    #trajectory = RandomDist(planet, [planet.radius+330.0*1000-2500 , planet.radius + 330.0*1000+2500], point_count) #LEO
    epochs = 100
    bs = 1
    plot_maps = True

    experiment_dir = generate_experiment_dir(trajectory, preprocessor)
    save_location = "./Files/Final_NN/" + experiment_dir + "/case_"+str(case) + "/"
    a = Analyze(directory + 'case_' + str(case) + '/' + cases[case-1]) 



    df = a.data
    run = df[df['val_loss'] == df['val_loss'].min()]
    params = run.iloc[0].to_dict()
    params['epochs'] = epochs
    params['batch_size'] = bs
    params['optimizer'] = eval(params['optimizer'].split('.')[-1].split('\'')[0]) #Nadam
    #params['kernel_initializer'] = eval(params['kernel_initializer'])
    if eval(params['kernel_regularizer']) is None:
        params['kernel_regularizer'] = None

    # if case == 2 or case == 3:
    #     params['lr'] = 0.05
    
    '''
    params['kernel_regularizer'] = 'l2'
    params['first_unit'] = 256
    params['first_neuron'] = 128
    params['hidden_layers'] = 0
    params['dropout'] = 0.4
    params['batch_size'] = 10
    params['lr'] = 0.1
    params['epochs'] = 50
    params['optimizer'] = Adadelta
    params['losses'] = 'mean_absolute_error'
    params['losses'] = 'mean_absolute_percentage_error'
    params['optimizer'] = SGD
    params['losses'] = 'mean_absolute_error'# 'mean_squared_error'
    '''


    sh_file = planet.sh_hf_file
    max_deg = 1000

    gravityModelMap = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    gravityModelMap.load() 
    gravityModelMapC20 = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    gravityModelMapC20.load() 
    gravityModelMap.accelerations -= gravityModelMapC20.accelerations

    pos_sphere = cart2sph(trajectory.positions)
    pos_sphere = check_fix_radial_precision_errors(pos_sphere)
    acc_proj = project_acceleration(pos_sphere, gravityModelMap.accelerations)
    
    preprocessor.percentTest = 0.3
    preprocessor.split(pos_sphere, acc_proj)
    preprocessor.fit()
    x_train, x_val, y_train, y_val = preprocessor.apply_transform()
    #x_train, y_train = pos_sphere, acc_proj # don't even transform the data. 
    # preprocessor = RobustTransform()
    # a = Analyze("./Hyperparams/Initial_Search/Uniform/070820134910.csv") # Uniform Robust 1

   
    hist, model = NN_hyperparam(x_train, y_train, x_val, y_val, params, verbose=1, save_location=save_location)
    #hist, model = NN_hyperparam(x_train, y_train, x_train, y_train, params, verbose=1) # Test on the same data

    plot_metrics(hist)
    compute_error(model, 
                                x_train, y_train,
                                x_val, y_val
                                , preprocessor)

    # Plot NN Results
    if plot_maps:
        map_grid = DHGridDist(planet, planet.radius, degree=175)
        sh_all_gravityModel = SphericalHarmonics(sh_file, degree=max_deg, trajectory=map_grid)
        sh_C20_gravityModel = SphericalHarmonics(sh_file, degree=2, trajectory=map_grid)
        true_grid = Grid(trajectory=map_grid, accelerations=sh_all_gravityModel.load()
        sh_20_grid = Grid(trajectory=map_grid, accelerations=sh_C20_gravityModel.load()
        true_grid -= sh_20_grid #these values are projected
        nn = NN_Base(model, preprocessor, test_traj=map_grid)

        gravityModelMap = SphericalHarmonics(sh_file, degree=100, trajectory=map_grid)
        gravityModelMap.load() 
        C100_grid = Grid(trajectory=map_grid, accelerations=gravityModelMap.load()
        C100_grid -= sh_20_grid

        map_viz = MapVisualization(unit = 'mGal')
        grid = Grid(trajectory=map_grid, accelerations=nn.load(), override=True)
        fig, ax = map_viz.plot_grid_rmse(grid, true_grid,vlim=[0, 40])

        # std = np.std(true_grid.total)
        # mask = true_grid.total > 3*std
        M_params = count_params(nn.model.trainable_weights)
        print("Params: " + str(M_params))

        #map_viz.save(fig, nn.file_directory+"NN_Rel_Error.pdf")
        # coefficient_list.append(M_params)
        # rmse_list.append(np.average(np.sqrt(np.square(grid.total - true_grid.total))))
        # rmse_feat_list.append(np.average(np.sqrt(np.square((grid.total - true_grid.total))),weights=mask))
        plt.show()




if __name__ == '__main__':
    main(1)
    #main(2)
    #main(3)
    
    #main(4)
    #main(5)
    #main(6)
    main(None)
