import numpy as np
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from GravNN.Preprocessors.UniformScaler import UniformScaler
from GravNN.Support.transformations import cart2sph


import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

import copy
import numpy as np
from GravNN.Networks.utils import (
    configure_tensorflow,
    set_mixed_precision,
    check_config_combos,
)
from GravNN.Networks.Callbacks import SimpleCallback
from GravNN.Networks.Data import get_preprocessed_data, configure_dataset, compute_input_layer_normalization_constants
from GravNN.Networks.Model import PINNGravityModel
from GravNN.Networks.Networks import load_network
from GravNN.Networks.utils import populate_config_objects, configure_optimizer
from GravNN.Networks.Schedules import get_schedule

tf = configure_tensorflow()


# tf.config.run_functions_eagerly(True)
tf.keras.backend.clear_session()

class NN:
    def __init__(self, config):
        tf.keras.backend.clear_session()

        np.random.seed(config['seed'][0])
        tf.random.set_seed(config['seed'][0])
        # Get data, network, optimizer, and generate model
        compute_input_layer_normalization_constants(config)
        self.config = config
        optimizer = configure_optimizer(config, mixed_precision=None)
        network = load_network(config)
        model = PINNGravityModel(config, network)
        model.compile(optimizer=optimizer, loss="mse")
        self.model = model


    def update(self, rVec, aVec, iterations=5):
        callback = tf.keras.callbacks.EarlyStopping('loss',min_delta=1E-6, patience=2000, restore_best_weights=False, verbose=1)
        history = self.model.fit(
                        x=tf.convert_to_tensor(rVec.astype(np.float32)),
                        y=tf.convert_to_tensor(aVec.astype(np.float32)),
                        batch_size=self.config['batch_size'][0],
                        epochs=iterations,
                        verbose=1,
                        callbacks=[callback]
                    )
        self.model.history = history


