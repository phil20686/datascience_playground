import pandas as pd
import numpy as np
from src.LinearRegressionPlots import LinearRegressionPlotPack

if __name__ == "__main__":

    from sklearn.datasets import load_boston
    boston = load_boston()

    print(boston.keys())
    print(boston.DESCR)

    df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df['PRICE'] = boston['target']

    print(df['RM'].head())
    print(df.head())

    # formula = "PRICE ~ {}".format(" + ".join(boston['feature_names']))
    formula = "np.log(PRICE) ~ CRIM + ZN + INDUS + C(CHAS) + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT"
    plotter = LinearRegressionPlotPack(df, formula, dependent_variable='PRICE')

    ser_reduced_data_estimates = plotter.reduce_data_to_avoid_collinearity(max_value=10)

    formula = "np.log(PRICE) ~ LSTAT + DIS + RAD + ZN + CRIM + C(CHAS)"
    df_reduced_data = df.loc[:, ser_reduced_data_estimates.index.tolist()+['PRICE']]
    plotter = LinearRegressionPlotPack(df_reduced_data, formula, dependent_variable='PRICE')

    plotter.create_all_plots()