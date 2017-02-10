import pytest
from src import get_data, linear_model, preprocess, nn_model
import matplotlib.pyplot as plt
NN_TYPE = nn_model.NN_TYPE


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


@pytest.mark.skip()
@pytest.fixture()
def real_getdata():
    term_dict = {
        'train': ['2011-01-01', '2014-12-31'],
        'valid': ['2015-01-01', '2015-12-31'],
        'test' : ['2016-01-01', '2016-12-31']
    }
    panel = get_data.symbols2daily_values(kinds='sandp500')
    features, labels, terms = preprocess.panel2get_batch(
        panel,
        term_dict,
        feature_term=6)
    assert not labels['train'][0] in features['train']
    return features, labels, terms


@pytest.mark.skip('time loss')
def test_linear(test_getdata):
    d1, d2, t = test_getdata
    l1 = linear_model.LinearModel(d1, d2, t)
    assert len(l1.features['train'].shape) == 2
    l1.inference()
    l1.plot_direction_accuracy(save=True)
    l1.plot_profit(save=True)


@pytest.mark.skip()
def test_ffnn(test_getdata):
    d1, d2, t = test_getdata
    l1 = nn_model.NnModel(d1, d2, t, kinds=NN_TYPE[0])
    l1.layer_stack([3, 5])
    l1.inference()
    l1.plot_learning_curve(save=True)
    l1.plot_direction_accuracy(save=True)
    l1.plot_profit(save=True)


# fail test
# @pytest.mark.skip()
def test_lstm(test_getdata):
    d1, d2, t = test_getdata
    l1 = nn_model.NnModel(d1, d2, t, kinds=NN_TYPE[1])
    l1.layer_stack([3, 5])
    l1.inference()
    l1.plot_learning_curve(save=True)
    l1.plot_direction_accuracy(save=True)
    l1.plot_profit(save=True)


@pytest.mark.skip()
def test_real_ffnn(real_getdata):
    d1, d2, t = real_getdata
    l1 = nn_model.NnModel(d1, d2, t, kinds=NN_TYPE[0])
    l1.layer_stack([10000, 3000])
    l1.inference(nb_epoch=20, batchsize=128, verbose=1)
    l1.plot_learning_curve(save=True)
    l1.plot_direction_accuracy(save=True)
    l1.plot_profit(save=True)


