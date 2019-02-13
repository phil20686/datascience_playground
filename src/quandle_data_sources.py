import requests
import pyarrow, pandas

_port = 9090

class QuandlDataSources(object):

    class Currencies(object):
        usdaud = "FRED/DEXUSAL"
        canusd = "FRED/DEXCAUS"
        usdgbp = "FRED/DEXUSUK"
        cnyusd = "FRED/DEXUSUK"
        usdeur = "FRED/DEXUSEU"
        jpyusd = "FRED/DEXJPUS"

    class Equities(object):

        spy = "EOD/SPY"
        appl = "EOD/AAPL"

class Datahandler(object):

    @classmethod
    def get(cls, list_quandle_series, start_date, end_date):
        r = requests.get('http://localhost:{}/v1.0/data/multiple'.format(_port),
                         params={'list_series': list_quandle_series,
                                 'start_date': start_date,
                                 'end_date': end_date})

        if r.status_code == 200:
            return pyarrow.deserialize_pandas(r.content)
        else :
            raise Exception("unable to acquire data: HTTP code: ".format(r.status_code))