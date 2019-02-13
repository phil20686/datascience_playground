from src.ARIMAModels import ARIMAModels
from src.quandle_data_sources import QuandlDataSources, Datahandler
import pandas as pd
from datetime import date

if __name__ == "__main__":
    series = [
        QuandlDataSources.Equities.appl,
    ]
    df = Datahandler.get(series, start_date=date(2014, 1, 1), end_date=date.today())
    df.columns = pd.MultiIndex.from_tuples([tuple(x.strip() for x in name.split('-')) for name in df.columns])

    ser_raw =df.loc[:, [(QuandlDataSources.Equities.appl, 'Adj_Close')]].squeeze().pct_change().dropna()
    print(ser_raw.head())

    arima = ARIMAModels(ser_raw)
    arima.create_timeseries_plots()[0].show()

    #find the best model
    mdl, order = arima.find_best_model_aic(range(4,6,1), range(1), range(3,5,1))
    print(order)
    print(mdl.summary())
    print(arima.test_residuals(mdl))

    arima.create_timeseries_plots(data=mdl.resid)[0].show()
    arima.forecast_plots(mdl).show()




