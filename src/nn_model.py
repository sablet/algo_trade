from __future__ import print_function
import os
from keras.layers import LSTM, Conv1D
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from src.model_template import ModelTemplate
from src.utility import get_out_path, np3to2

TYPE = ['FFNN', 'LSTM', 'CNN']


class NnModel(ModelTemplate):
    """
    predict and evaluate from features amd label dict
    """
    def __init__(self, features, labels, terms, kinds=TYPE[0], model_path=None):
        assert kinds in TYPE
        self.n_data, self.n_feature, self.n_label = features['train'].shape
        super().__init__(features, labels, terms, feature_1dim=(kinds in TYPE[:1]))
        if model_path is not None:
            assert os.path.exists(model_path)
            self.model = load_model(model_path)
        self.kinds = kinds
        self.history = None
        self.predicted_labels = None

    def layer_stack(self, topology_arr):
        """
        structs Neural Network Layer and display summary
        param: topology_arr: list
        """
        self.model = Sequential()
        self._characteristic_layer_construct(topology_arr)
        self.model.compile(
            loss='mse',
            optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        )
        self.model.summary()

    def _characteristic_layer_construct(self, topology_arr):
        if self.kinds is TYPE[0]:
            for n_unit in topology_arr:
                self.model.add(Dense(
                    n_unit,
                    input_dim=self.n_feature*self.n_label,
                    activation='relu'
                ))
            self.model.add(Dense(self.n_label, activation='relu'))
        else:
            assert False

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
            self.model.save(get_out_path("keras_model.h5"))
