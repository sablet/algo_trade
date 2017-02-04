import pytest
from src import get_data, linear_model, preprocess
import matplotlib.pyplot as plt


@pytest.fixture()
def test_getdata():
    term_dict = {
        'train': ['2016-11-01', '2016-11-15'],
        'valid': ['2016-11-16', '2016-11-22'],
        'test' : ['2016-11-23', '2016-11-30']
    }
    panel = get_data.symbols2daily_values(
        kinds='apple_and_amazon',
        symbols=['aapl', 'amzn'])
    features, labels, terms = preprocess.panel2get_batch(
        panel,
        term_dict,
        feature_term=3)
    return features, labels, terms


# @pytest.mark.skip('time loss')
def test_linear(test_getdata):
    d1, d2, t = test_getdata
    l1 = linear_model.LinearModel(d1, d2, t)
    assert len(l1.features['train'].shape) == 2
    l1.inference()
    l1.plot_direction_accuracy(save=True)
    # plt.show()
    l1.plot_profit(save=True)
    # plt.show()


@pytest.mark.skip('progress')
def test_ffnn(test_getdata):
    d1, d2, t = test_getdata
    l1 = linear_model.LinearModel(d1, d2, t)
    assert len(l1.features['train'].shape) == 2
    l1.inference()
    l1.plot_direction_accuracy()
    plt.show()
    l1.plot_profit()
    plt.show()
