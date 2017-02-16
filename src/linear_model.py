from sklearn.linear_model import LinearRegression
from src.model_template import ModelTemplate
from src.utility import np3to2


class LinearModel(ModelTemplate):
    def __init__(self, features, labels, terms):
        super().__init__(features, labels, terms)
        self.model = LinearRegression()

    def inference(self):
        self.model.fit(self.features['train'], self.labels['train'])
        self.predict_all()

    def predict_all(self):
        self.predicted_labels = {key: self.model.predict(
            self.features[key]
        ) for key in self.features.keys()}
