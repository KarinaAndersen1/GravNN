
import os
import copy
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
#import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Networks.Data import generate_dataset
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

def generate_test_dataset(x, batch_size):
    x = x.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((x,))
    dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset

def time_model(positions, model, batch=True):
    start = time.time()
    if batch:
        output = model.compute_acceleration(positions)
    else:
        for i in range(len(positions)):
            output = model.compute_acceleration(positions[i,:])
    delta = time.time() - start
    try:
        params = model.mesh.vertices.shape[0]* model.mesh.vertices.shape[1] + model.mesh.faces.shape[0]
    except:
        pass

    try:
        params = model.degree*(model.degree+1)
    except:
        pass

    return params, delta

def time_network(positions, config, network, batch=True):
    if batch:
        dataset = generate_test_dataset(positions.astype('float32'), 10000)
        output = network.predict(dataset)
    else:
        for i in range(len(positions)):
            output = network.predict(positions[i,:])
    
    start = time.time()
    output = network.predict(dataset)
    delta = time.time() - start
    params = config['params'][0]
    return params, delta

def time_models_in_df(df_file, column_name, batch=True):
    total_params = []
    time = []

    df = pd.read_pickle(df_file)
    ids = df['id'].values
    
    for model_id in ids:
        tf.keras.backend.clear_session()
        config, model = load_config_and_model(model_id, df_file)
        params, delta = time_network(positions, config, model, batch=batch)
        total_params.append(params)
        time.append(delta)
    df = pd.DataFrame(data=time, index=total_params, columns=[column_name])
    return df


def time_polyhedral(asteroid, batch):
    poly_params = []
    poly_time = []
    poly_3 = Polyhedral(asteroid, asteroid.model_3k)
    poly_6 = Polyhedral(asteroid, asteroid.model_6k)
    poly_12 = Polyhedral(asteroid, asteroid.model_12k)
    poly_25 = Polyhedral(asteroid, asteroid.model_25k)

    models = [poly_3, poly_6, poly_12, poly_25]
    for model in models:
        params, delta = time_model(positions, model, batch=batch)
        poly_params.append(params)
        poly_time.append(delta)
    poly_df = pd.DataFrame(data=poly_time, index=poly_params, columns=['poly_time'])
    return poly_df


def time_spherical_harmonics(planet, batch):
    sh_params = []
    sh_time = []

    sh_10 = SphericalHarmonics(planet.sh_hf_file, 10)
    sh_50 = SphericalHarmonics(planet.sh_hf_file, 50)
    sh_200 = SphericalHarmonics(planet.sh_hf_file, 200)
    sh_400 = SphericalHarmonics(planet.sh_hf_file, 400)

    models = [sh_10, sh_50, sh_200, sh_400]
    for model in models:
        params, delta = time_model(positions, model,batch=batch)
        sh_params.append(params)
        sh_time.append(delta)
    sh_df = pd.DataFrame(data=sh_time, index=sh_params, columns=['sh_time'])
    return sh_df

positions = np.random.uniform(size=(10000,3))*1E4# Must be in meters
#positions = np.random.uniform(size=(10,3))*1E4

def conference_timing():
    earth = Earth()
    asteroid = Eros()

    poly_df = time_polyhedral(asteroid)
    sh_df = time_spherical_harmonics(earth)
    
    nn_df = time_models_in_df('N_10000_rand_study.data', 'nn_time')
    pinn_df = time_models_in_df('N_10000_rand_PINN_study.data', 'pinn_time')


    df = pd.concat([poly_df, sh_df, nn_df, pinn_df])#, nn_CPU_df, pinn_CPU_df])
    df.to_pickle('Data/speed_results_v2.data')

def journal_timing():
    earth = Earth()
    asteroid = Eros()

    batch = True

    poly_df = time_polyhedral(asteroid,batch)
    sh_df = time_spherical_harmonics(earth,batch)
    
    #* Traditional Network (GPU)
    nn_df = time_models_in_df('Data/Dataframes/traditional_nn_df.data', 'nn_time', batch)
    pinn_df = time_models_in_df('Data/Dataframes/pinn_df.data', 'pinn_time', batch)


    df = pd.concat([poly_df, sh_df, nn_df, pinn_df])#, nn_CPU_df, pinn_CPU_df])
    df.to_pickle('Data/Dataframes/speed_results_journal.data')



if __name__ == '__main__':
    journal_timing()