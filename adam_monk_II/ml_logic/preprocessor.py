import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

from adam_monk_II.ml_logic.methods import format_columns, shift_weather_features
from taxifare.utils import simple_time_and_memory_tracker

@simple_time_and_memory_tracker

def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    # Rename columns
    names = [
    "date",
    "temperature_berlin", "temperature_cologne", "temperature_frankfurt", "temperature_hamburg", "temperature_munich",
    "prep_berlin", "prep_cologne", "prep_frankfurt", "prep_hamburg", "prep_munich",
    "snow_berlin", "snow_cologne", "snow_frankfurt", "snow_hamburg", "snow_munich",
    "windspeed_berlin", "windspeed_cologne", "windspeed_frankfurt", "windspeed_hamburg", "windspeed_munich",
    "irradiation_berlin", "irradiation_cologne", "irradiation_frankfurt", "irradiation_hamburg", "irradiation_munich",
    "future_price",
    "gen_biomass",
    "gen_ff_browncoallignite",
    "gen_ff_coalderivedgas",
    "gen_fossilgas",
    "gen_fossilhardcoal",
    "gen_fossiloil",
    "gen_geothermal",
    "gen_hydropumpedstorage",
    "gen_hydrorunofriver",
    "gen_hydrowaterreservoir",
    "gen_nuclear",
    "gen_other",
    "gen_otherrenewable",
    "gen_solar",
    "gen_waste",
    "gen_windoffshore",
    "gen_windonshore",
    ]
    X.columns = names

    # Reorder columns
    X = X[[c for c in X.columns if c != "future_price"] + ["future_price"]]

    # Define date_df for shift_weather_features() method
    date_df = X[["date"]]

    # Filter dataset for years 2015+ as generation data is not available as far back as 2003
    X = X[X.date.dt.year >= 2015]

    X = X.drop(columns="date")

    def create_sklearn_preprocessor(X, date_df) -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 43)
        into a preprocessed one of fixed shape (_, 65).

        """
        snow_imputer = SimpleImputer(strategy="constant", fill_value=0)
        snow_columns = [c for c in X.columns if "snow" in c]
        snow_preproc = ColumnTransformer(
             [
                  ("snow_imputer", snow_imputer, snow_columns)
             ]
        )

        # Berlin Irradiation Imputer
        berlin_irradiation_imputer = FunctionTransformer(
             lambda X: shift_weather_features(X, date_df, "irradiation_berlin", 1)
        )

        # Windspeed Munich Imputer
        windspeed_munich_imputer = FunctionTransformer(
             lambda X: shift_weather_features(X, date_df, "windspeed_munich", -1)
        )

        # Merge three imputers
        preproc_imputers = ColumnTransformer(
            [
                ("snow_imputer", snow_imputer, snow_columns),
                ("irradiation_berlin", berlin_irradiation_imputer, ["irradiation_berlin"]),
                ("windspeed_munich", windspeed_munich_imputer, ["windspeed_munich"])
            ],
            remainder="passthrough"
        )

        # Add weather mean imputer
        weather_mean_imputer = FunctionTransformer(
             lambda X: average_weather_features(
            X, ["windspeed", "irradiation", "temperature"]
            )
        )

        # Final preprocessor
        final_preprocessor = make_pipeline(
            preproc_imputers,
            weather_mean_imputer,
            StandardScaler(),
            PCA(),
        )

        return final_preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
