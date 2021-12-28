from __future__ import print_function
import sys
import tensorflow as tf

import yaml
from fedn.utils.kerashelper import KerasHelper
from models.KDD_model import create_seed_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def train(model, settings):
    """
    Helper function to train the model
    :return: model
    """
    print("-- RUNNING TRAINING --", flush=True)
    xtrain = np.array(pd.read_csv('../data/xtrain.csv',  header=None))
    xtrain = xtrain.reshape(xtrain.shape[0], 1, xtrain.shape[1])
    ytrain = np.array(pd.read_csv('../data/ytrain.csv',  header=None))
    _, X, _, y = train_test_split(xtrain, ytrain, test_size=settings['test_size'])

    model.fit(X, y, epochs=settings['epochs'], batch_size=settings['batch_size'], verbose=True)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise (e)

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model()
    model.set_weights(weights)
    model = train(model, settings)
    helper.save_model(model.get_weights(), sys.argv[2])


