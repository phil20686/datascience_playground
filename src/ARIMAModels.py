import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms

import matplotlib.pyplot as plt
import matplotlib as mpl


class ARIMAModels(object):

    def __init__(self, data, max_lag = 20):
        """

        Args:
            data (pd.Series):
        """
        self._data = data
        self.max_lag = max_lag

    def create_timeseries_plots(self, figsize=(10, 8), style='bmh', data=None):
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # mpl.rcParams['font.family'] = 'Ubuntu Mono'
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))

            data = self._data if data is None else data
            data.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(data, lags=self.max_lag, ax=acf_ax, alpha=0.05)
            smt.graphics.plot_pacf(data, lags=self.max_lag, ax=pacf_ax, alpha=0.05)
            sm.qqplot(data, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            scs.probplot(data, sparams=(data.mean(), data.std()), plot=pp_ax)

            plt.tight_layout()
        return fig, [ts_ax, acf_ax, pacf_ax, qq_ax, pp_ax]

    def fit_model(self, order):
        """

        Args:
            order (tuple[int]): a three tuple for the arima orders

        Returns:

        """
        model = smt.ARIMA(self._data, order).fit(
            method='mle', trend='nc'
        )
        return model

    def find_best_model_aic(self, generator_i, gen_j, gen_k):
        """
        The parameters are generators over the range of parameters that you want to search.
        Args:
            generator_i:
            gen_j:
            gen_k:

        Returns:

        """
        best_aic = np.inf
        best_order = None
        best_mdl = None
        for i in generator_i:
            for j in gen_j:
                for k in gen_k:
                    try :
                        model = self.fit_model((i,j,k))
                        temp_aic = model.aic
                    except :
                        continue
                    else :
                        if temp_aic < best_aic:
                            best_aic = temp_aic
                            best_order = (i,j,k)
                            best_mdl = model
        return best_mdl, best_order

    def test_residuals(self, model):
        """
        This will perform a box ljung test on the residuals and
        Args:
            model:

        Returns:

        """
        return sms.diagnostic.acorr_ljungbox(model.resid, lags=self.max_lag, boxpierce=False)

    def forecast_plots(self, model, steps=None):
        steps = steps or self.max_lag
        forecast, err95, ci95 = model.forecast(steps=steps)

        new_index = pd.date_range(self._data.index[-1], periods=steps, freq='D')
        fc_95 = pd.DataFrame(np.column_stack([forecast, ci95]), index=new_index, columns=
                     ['forecast', 'lower_95', 'upper_95'])

        plt.style.use('bmh')
        fig = plt.figure(figsize=(15, 10))
        ax = plt.gca()
        ts = self._data
        ts.plot(ax=ax, label=self._data.name)
        # in sample prediction
        pred = model.predict(0, len(ts.index)-1)
        pred.plot(ax=ax, style='r-', label='In-sample prediction')
        styles = ['b-', '0.2', '0.75', '0.2', '0.75']
        fc_95.plot(ax=ax, style=styles)
        plt.fill_between(fc_95.index, fc_95.lower_95,
                         fc_95.upper_95, color='gray', alpha=0.7)
        fig.legend(loc='best', fontsize=10)
        return fig

