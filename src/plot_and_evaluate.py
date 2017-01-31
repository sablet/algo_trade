from traceback import extract_stack
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gmean


class PlotAndEvaluate(object):
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
        profit = (np.e ** self.labels[data_key][1:]) ** np.sign(
            self.predicted_labels[data_key][1:])
        print("whole profit ration is {}".format(np.around(
            gmean(profit, axis=None), 5
        )))
        if kinds is 'portfolio':
            pd.DataFrame(
                np.multiply.accumulate(profit.mean(axis=1)),
                index=self.terms[data_key][1:],
                columns=[kinds]
                ).plot()
        self._save_png(save)