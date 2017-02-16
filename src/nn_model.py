from __future__ import print_function
import os
from keras.layers import LSTM, Conv1D
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Nadam
from src.model_template import ModelTemplate
from src.utility import get_out_path, np3to2
import pytest

NN_TYPE = ['FFNN', 'LSTM', 'CNN']


class NnModel(ModelTemplate):
    """
    predict and evaluate from features amd label dict
    """
    def __init__(self, features, labels, terms, kinds=NN_TYPE[0], model_path=None):
        assert kinds in NN_TYPE
        self.n_data, self.n_feature, self.n_label = features['train'].shape
        super().__init__(features, labels, terms, feature_1dim=(kinds in NN_TYPE[:1]))
        if model_path is not None:
            assert os.path.exists(model_path)
            self.model = load_model(model_path)
        self.kinds = kinds
        self.history = None
        self.predicted_labels = None

    def setup_layer(self, topology_arr):
        """
        structs Neural Network Layer and display summary
        param: topology_arr: list
        """
        self.model = Sequential()
        self._characteristic_layer_construct(topology_arr)
        self.model.compile(
            loss='mse',
            optimizer=Nadam()
        )
        self.model.summary()

    def _characteristic_layer_construct(self, topology_arr):
        if self.kinds is NN_TYPE[0]:
            self.model.add(Dense(
                topology_arr[0],
                input_dim=self.n_feature * self.n_label,
                activation='relu'
            ))
            for n_unit in topology_arr[1:]:
                self.model.add(Dense(
                    n_unit,
                    activation='relu'
                ))
            self.model.add(Dense(self.n_label))
        elif self.kinds is NN_TYPE[1]:
            self.model.add(LSTM(
                topology_arr[0],
                input_dim=self.n_feature * self.n_label,
            ))
            for n_unit in topology_arr[1:]:
                self.model.add(LSTM(n_unit))
            self.model.add(LSTM(self.n_label))
        else:
            assert False

    def inference(self, nb_epoch=1000, verbose=0, batchsize=None, save=False):
        """
        fitting parameter and get inference values
        :param batchsize: int
        :param nb_epoch: int
        :param verbose: int
        :param save: Bool
        """
        if batchsize is None:
            batchsize = len(self.features['train'])
        print("inference...")
        self.history = self.model.fit(
            self.features['train'],
            self.labels['train'],
            batch_size=batchsize,
            nb_epoch=nb_epoch,
            verbose=verbose,
            validation_data=(self.features['valid'], self.labels['valid'])
        )
        self.predict_all(verbose=0)
        if save is True:
            self.model.save(
                get_out_path(
                    self.kinds + "_model.h5"
                ))

    def predict_all(self):
        self.predicted_labels = {key: self.model.predict(
            self.features[key], verbose=0
        ) for key in self.features.keys()}

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
