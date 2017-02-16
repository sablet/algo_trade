from sklearn.linear_model import LinearRegression
from src.model_template import ModelTemplate
from src.utility import np3to2


class LinearModel(ModelTemplate):
    def __init__(self, features, labels, terms, kinds='svr'):
        super().__init__(features, labels, terms)
        self.model = LinearRegression()

    def inference(self):
        self.model.fit(self.features['train'], self.labels['train'])
        self.predict_all()
