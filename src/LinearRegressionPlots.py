import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot


class _Residuals(object):

    def __init__(self, model_fit):
        self.model_fit = model_fit

    @property
    def residuals(self):
        return self.model_fit.resid

    @property
    def normalised_residuals(self):
        return self.model_fit.get_influence().resid_studentized_internal

    @property
    def residuals_abs_sqrt(self):
        return np.sqrt(np.abs(self.normalised_residuals))

    @property
    def absolute_residuals(self):
        return np.abs(self.residuals)

    @property
    def leverage(self):
        return self.model_fit.get_influence().hat_matrix_diag

    @property
    def cooks_distance(self):
        return self.model_fit.get_influence().cooks_distance[0]


class LinearRegressionPlotPack(object):

    def __init__(self, data, model_string, dependent_variable=None):
        """

        Args:
            data (pd.DataFrame):
            model_string (str):
            dependent_variable (None|str):
        """
        self._data = data.dropna()
        self._model_string = model_string
        self.dependent_variable = dependent_variable

        self.model = smf.ols(formula=self._model_string, data=self._data)
        self.model_fit = self.model.fit()
        self.fitted_values = self.model_fit.fittedvalues
        self.residuals = _Residuals(self.model_fit)

    def get_dependent_variable(self):
        return self.dependent_variable or self._model_string.split('~')[0].strip()


    def get_summary(self):
        return self.model_fit.summary()

    def _get_dependent_variables(self):
        return self._data.loc[:, [x for x in self._data.columns if x != self.get_dependent_variable()]]

    def _test_data_for_collinearity(self, df_dependent_variables):
        return pd.Series(
            [
                variance_inflation_factor(df_dependent_variables.values, i) for i in
                   range(df_dependent_variables.shape[1])
            ], index=df_dependent_variables.columns
        ).sort_values(0, ascending=False)

    def test_for_multicollinearity(self):
        """
        If the variance inflation factor is > 5 then there is a multicollinearity problem.
        Returns:
            pd.Series : A series with the variance inflation factors
        """
        df_dependent_variables = self._get_dependent_variables()
        return self._test_data_for_collinearity(df_dependent_variables)

    def reduce_data_to_avoid_collinearity(self, max_value=5):
        """
        while there are any collinearity estimates this will remove the highest and retest.
        Returns:
            pd.Series: The final collinearity estimates.
        """
        data = self._get_dependent_variables()
        collinearity = self._test_data_for_collinearity(data)

        while collinearity.max() > max_value:
            data = data.loc[:, collinearity.index[1:]]
            collinearity = self._test_data_for_collinearity(data)
        return collinearity


    def _common_formatting(self):
        plt.style.use('seaborn')  # pretty matplotlib plots

        plt.rc('font', size=14)
        plt.rc('figure', titlesize=18)
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=18)

    def actual_vs_predicted(self, ax):
        sns.residplot(
            self.fitted_values,
            self.get_dependent_variable(),
            data=self._data,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax
        )
        ax.set_title('Residuals vs Fitted')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')

        #annotate the largest residuals with their index values.
        abs_resid_top_3 = self.residuals.absolute_residuals.sort_values(ascending=False)[:3]
        for i in abs_resid_top_3.index:
            ax.annotate(i, xy=(self.fitted_values[i], self.residuals.residuals[i]))

    def qq_plot(self, ax):
        QQ = ProbPlot(self.residuals.normalised_residuals)
        QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax)
        ax.set_title('Normal Q-Q')
        ax.set_xlabel('Theoretical Quintiles')
        ax.set_ylabel('Standardised Residuals')

        # annotations
        abs_norm_resid_top_3 = np.flip(np.argsort(np.abs(self.residuals.normalised_residuals)), 0)[:3]

        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residuals.normalised_residuals[i])
            )

    def scale_location_plot(self, ax):
        ax.scatter(self.fitted_values, self.residuals.residuals_abs_sqrt, alpha=0.5)
        sns.regplot(
            self.fitted_values, self.residuals.residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}
        )

        ax.set_title('Scale-Location')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('$\sqrt{|Standardized Residuals|}$')

        #annotations
        abs_sq_norm_resid_top_3 = np.flip(np.argsort(self.residuals.residuals_abs_sqrt), 0)[:3]

        for i in abs_sq_norm_resid_top_3:
            ax.annotate(i, xy=(self.fitted_values[i], self.residuals.residuals_abs_sqrt[i]))

    def leverage_plot(self, ax):
        ax.scatter(self.residuals.leverage, self.residuals.normalised_residuals)
        sns.regplot(self.residuals.leverage, self.residuals.normalised_residuals,
                    scatter=False, ci=False, lowess=True, line_kws = {'color': 'red', 'lw': 1, 'alpha': 0.8}
                    )
        ax.set_xlim(0, 0.2)
        ax.set_ylim(-3, 5)
        ax.set_title('Residuals vs Leverage')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardised Residuals')

        #annotations
        leverage_top_3 = np.flip(np.argsort(self.residuals.cooks_distance), 0)[:3]

        for i in leverage_top_3:
            ax.annotate(i, xy=(self.residuals.leverage[i], self.residuals.normalised_residuals[i]))

        def graph(formula, x_range, label=None):
            x = x_range
            y = formula(x)
            plt.plot(x, y, label=label, lw=1, ls='--', color='red')

        p = len(self.model_fit.params)  # number of model parameters

        graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
              np.linspace(0.001, 0.200, 50),
              'Cook\'s distance')  # 0.5 line
        graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
              np.linspace(0.001, 0.200, 50))  # 1 line
        plt.legend(loc='upper right')

    def create_all_plots(self):
        self._common_formatting()
        print(self.get_summary())

        ser_vif = self.test_for_multicollinearity()
        if ser_vif.max() > 5.0:
            logging.getLogger(__name__).warning("This regression has a server multicollinearity problem: \n{}".format(str(ser_vif)))
            print(ser_vif)

        fig, ax = plt.subplots()
        self.actual_vs_predicted(ax)
        fig.show()

        fig, ax = plt.subplots()
        self.qq_plot(ax)
        fig.show()

        fig, ax = plt.subplots()
        self.scale_location_plot(ax)
        fig.show()

        fig, ax = plt.subplots()
        self.leverage_plot(ax)
        fig.show()