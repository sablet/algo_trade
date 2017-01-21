from __future__ import print_function
from keras.models import Sequential, load_model
# from keras.layers.core import Dense
from keras.layers import LSTM, Conv1D
from keras.optimizers import RMSprop
import numpy as np
import os
import matplotlib.pyplot as plt


class LearningSequence:
    """
    DataFrame predict evaluate
    """
    def __init__(self, features, labels, model_path=None):
        self.outpath = lambda path: os.path.join("exp_outcome", path)
        self.features = features
        self.labels = labels
        if model_path is not None:
            assert os.path.exists(model_path)
            self.model = load_model(model_path)

    def sturcts_layer(self, topology_arr):
        """
        structs Feed Forward Neural Network
        param: topology_arr: List
        """
        n_data, n_feature, n_kinds = self.features['train'].shape
        self.model = Sequential()
        self.model.add(Conv1D(n_kinds, topology_arr[0], input_shape=(n_feature, n_kinds)))
        self.model.add(LSTM(n_kinds))
        self.model.compile(
            loss='mse',
            optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            )
        self.model.summary()

    def inference(self, nb_epoch=10000, verbose=0, save=False):
        print("inference...")
        history = self.model.fit(
            self.features['train'],
            self.labels['train'],
            batch_size=len(self.features['train']),
            nb_epoch=nb_epoch,
            verbose=verbose,
            validation_data=(self.features['valid'], self.labels['valid'])
        )
        for score in history.history.values():
            plt.plot(np.array(score))
        plt.savefig(self.outpath("learning_curve.png"))

        self.learned_label = self.model.predict(self.features['test'], verbose=0)
        if save is True:
            self.model.save(self.outpath("keras_model.h5"))

    def eval_accuracy(self):
        pass

