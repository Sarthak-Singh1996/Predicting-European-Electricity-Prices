
import pandas as pd

class Indicator:
    # Simple moving average
    def _sma(self,
             df: pd.DataFrame,
             price: str,
             ndays: int) -> pd.DataFrame:
        col_name = "sma_" + str(ndays)
        sma = pd.Series(
            df[price].rolling(ndays).mean(),
            name=col_name,
        )
        df = df.join(sma)
        return df

    # Exponentially-weighted moving average
    def _ewma(self,
              df: pd.DataFrame,
              price: str,
              ndays: int) -> pd.DataFrame:
        ewma = pd.Series(
            df[price].ewm(span=ndays, min_periods=ndays - 1).mean(),
            name="ewma_" + str(ndays),
        )
        df = df.join(sma)
        return df

    # Bolinger bands
    def _bbands(self,
                df: pd.DataFrame,
                price: str,
                window: int) -> pd.DataFrame:
        ma = df[price].rolling(window).mean()
        std = df[price].rolling(window).std()
        df["middle_band"] = ma
        df["upper_band"] = ma + 2 * std
        df["lower_band"] = ma + 2 * std
        return df

def run():
    # Load data
    url = "https://raw.githubusercontent.com/Tobias-Neubert94/adam_monk_II/master/adam_monk_II/data/Price_data.gzip"
    df = pd.read_parquet(url)

    # Add indicators
    ind = Indicator()
    df = ind._sma(df, "Future Price", 30)
    df = ind._bbands(df, "Future Price", window=30)
    return df

df = run()
# end
