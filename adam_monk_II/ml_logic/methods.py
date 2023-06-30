import pandas as pd
import numpy as np

def shift_weather_features(
    X: pd.DataFrame,
    date_df,
    fill_col: str,
    period:int
    ) -> pd.DataFrame:
    """
    Inputs:
        - X: feature matrix
        - date_df: date column from X
        - fill_col: column name where shifted values will replace current values
        - period: which direction and period the shift applies on.
    Returns:
        pd.DataFrame
    """
    X = date_df.join(X)
    X["shifted"] = X.groupby(
        [X["date"].dt.month, X["date"].dt.day]
    )[fill_col].shift(period)
    X[fill_col] = np.where(
        X[fill_col].isnull(),
        X["shifted"],
        X[fill_col]
    )
    X.drop(columns=["date", "shifted"], inplace=True)
    return X

def average_weather_features(
    X: pd.DataFrame,
    measures: List[str]
) -> pd.DataFrame:
    """
    Inputs:
        - measures: list of weather measures to be averaged across the dataframe
    Output:
        pd.DataFrame
    """
    col_list = []
    for measure in measures:
        col = f"{measure}_germany"
        X[col] = X[
            [c for c in X.columns if measure in c]
        ].mean(axis=1)
        df.drop(columns=[
            c for c in X.columns if measure in c and "germany" not in c
        ], inplace=True)
        col_list.append(col)
    return X
