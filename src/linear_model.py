from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.model_template import ModelTemplate
from src.shape_utility import np3to2


class LinearModel(ModelTemplate):
    def __init__(self, features, labels, terms, kinds='linear'):
        super().__init__(features, labels, terms)
        for key in self.features.keys():
            self.features[key] = np3to2(self.features[key])
        if kinds is 'linear':
            self.model = LinearRegression()
        elif kinds is 'svm':
            self.model = SVR(C=1.0, epsilon=0.1)
        else:
            assert False

    def inference(self):
        self.model.fit(self.features['train'], self.labels['train'])
        self.predict_all()
