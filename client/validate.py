import sys
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
import tensorflow.keras.models as krm
import json
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def validate(model):
    print("-- RUNNING VALIDATION --", flush=True)

    try:
        xtest = np.array(pd.read_csv('../data/inputData.csv', header=None))
        xtest = xtest.reshape(xtest.shape[0], 1, xtest.shape[1])
        ytest = np.array(pd.read_csv('../data/outputData.csv', header=None))

        _, X, _, y = train_test_split(xtest, ytest, test_size=0.15)

        model_score = model.evaluate(X, y, verbose=0)
        print('Testing loss:', model_score[0])
        print('Testing accuracy:', model_score[1])
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)

        clf_report = metrics.classification_report(y.argmax(axis=1),y_pred)
    except Exception as e:
        print("failed to validate the model {}".format(e),flush=True)
        raise
    
    report = { 
                "classification_report": clf_report,
                "loss": model_score[0],
                "accuracy": model_score[1]
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    from fedn.utils.kerashelper import KerasHelper
    from models.KDD_model import create_seed_model

    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])
    model = create_seed_model()
    model.set_weights(weights)
    report = validate(model)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))
