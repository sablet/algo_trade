from __future__ import print_function
from keras.models import Sequential, load_model
# from keras.layers.core import Dense
from keras.layers import LSTM, Conv2D
from keras.optimizers import RMSprop
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean
from traceback import extract_stack


class LearningSequence:
    """
    predict and evaluate from features amd label dict
    """

    def __init__(self, features, labels, terms, model_path=None):
        """
        predict and evaluate from features amd labels
        :param features: dict
        :param labels: dict
        :param model_path: str
        """
        for asserted_value in [features, labels, terms]:
            assert set(asserted_value.keys()) == {'train', 'valid', 'test'}
        self.features = features
        self.labels = labels
        self.terms = terms
        os.makedirs('log', exist_ok=True)
        self.outpath = lambda path: os.path.join('log', path)
        if model_path is not None:
            assert os.path.exists(model_path)
            self.model = load_model(model_path)
        self.history = None
        self.predicted_labels = None

    def sturcts_layer(self, topology_arr):
        """
        structs Neural Network Layer and display summary
        param: topology_arr: list
        """
        n_data, n_feature, n_kinds = self.features['train'].shape
        self.model = Sequential()
        self.model.add(Conv2D(
            topology_arr[0],
            topology_arr[1],
            topology_arr[2],
            activation='relu',
            input_shape=(None, n_feature, n_kinds)
        ))
        self.model.add(LSTM(n_kinds))
        self.model.compile(
            loss='mse',
            optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        )
        self.model.summary()

    def inference(self, nb_epoch=1000, verbose=0, save=True):
        """
        fitting parameter and get inference values
        :param nb_epoch: int
        :param verbose: int
        :param save: Bool
        """
        print("inference...")
        self.history = self.model.fit(
            self.features['train'],
            self.labels['train'],
            batch_size=len(self.features['train']),
            nb_epoch=nb_epoch,
            verbose=verbose,
            validation_data=(self.features['valid'], self.labels['valid'])
        )
        self.predicted_labels = {key: self.model.predict(
            self.features[key], verbose=0
        ) for key in self.features.keys()}

        if save is True:
            self.model.save(self.outpath("keras_model.h5"))

    def _save_png(self, save):
        """
        plot and save template
        :param save: bool
        :rtype: None
        """
        if save is True:
            plt.savefig(self.outpath(
                extract_stack()[-2][-2].replace("plot_", "") + ".png"))

    def plot_learning_curve(self, later=0, save=False):
        """
        get learning curve and plot
        axis=0: time sries index
        axis=1: enterprise kinds index
        :param later: int
        :param save:
        :return: None
        """
        if type(later) is int:
            pd.DataFrame.from_dict(self.history.history)[
            (len(self.history.history) // 2):].plot()
            self._save_png(save)

    def plot_direction_accuracy(self, data_key='valid', save=False):
        """
        evaluate predicted direction accuracy and plot
        :param data_key: str: valid or test
        :param save: bool
        :rtype: None
        """
        accuracy = (self.labels[data_key] * self.predicted_labels[data_key] > 0)
        print("whole accuracy is {}".format(round(float(accuracy.mean(axis=None)), 3)))
        plt.hist(accuracy.mean(axis=1), bins=20)
        self._save_png(save)

    def plot_profit(self, data_key='valid', kinds='portfolio', save=False):
        """
        evaluate profit time series and plot
        :param kinds: str
        :param data_key: str
        :param save: Bool
        :rtype: None
        """
        profit = np.sign(self.predicted_labels[data_key][1:]) \
                 * self.labels[data_key][1:] + 1
        print("whole profit ration is {}".format(np.around(
            gmean(profit, axis=None), 3
        )))
        if kinds is 'portfolio':
            pd.DataFrame(
                np.multiply.accumulate(profit.mean(axis=1)),
                index=self.terms[data_key][1:],
                columns=[kinds]
            ).plot()
        self._save_png(save)
