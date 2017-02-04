from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.model_template import ModelTemplate
from src.utility import np3to2


class LinearModel(ModelTemplate):
    def __init__(self, features, labels, terms):
        super().__init__(features, labels, terms)
        for key in self.features.keys():
            self.features[key] = np3to2(self.features[key])
        self.model = LinearRegression()
    
    def inference(self):
        self.model.fit(self.features['train'], self.labels['train'])
        self.predict_all()
