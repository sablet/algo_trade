from __future__ import print_function
import os

from src.plot_and_evaluate import PlotAndEvaluate
from src.shape_utility import np3to2
from keras.layers import LSTM, Conv1D
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from sklearn.linear_model import LinearRegression


class ModelTemplate(PlotAndEvaluate):
    """
    predict by linear model
    """
    def __init__(self, features, labels, terms):
        """
        predict and evaluate from features amd labels
        :param features: dict
        :param labels: dict
        """
        for asserted_value in [features, labels, terms]:
            assert set(asserted_value.keys()) == {'train', 'valid', 'test'}
        self.terms = terms
        self.labels = labels
        self.features = features
        self.predicted_labels = None

    def predict_all(self, verbose=None):
        if verbose is None:
            self.predicted_labels = {key: self.model.predict(
                self.features[key]
            ) for key in self.features.keys()}
        else:
            self.predicted_labels = {key: self.model.predict(
                self.features[key], verbose=0
            ) for key in self.features.keys()}


class LinearModel(ModelTemplate):
    def __init__(self, features, labels, terms):
        super().__init__(features, labels, terms)
        for key in features.keys():
            self.features[key] = np3to2(self.features[key])
        self.model = LinearRegression()

    def inference(self):
        self.model.fit(self.features['train'], self.labels['train'])
        self.predict_all()


class CnnModel(ModelTemplate):
    """
    predict and evaluate from features amd label dict
    """
    def __init__(self, features, labels, terms, model_path=None):
        super().__init__(features, labels, terms)
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
        self.model.add(Conv1D(
            n_kinds,
            topology_arr[0],
            activation='relu',
            input_shape=(n_feature, n_kinds)
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
        self.predict_all()
        if save is True:
            self.model.save(self.outpath("keras_model.h5"))
