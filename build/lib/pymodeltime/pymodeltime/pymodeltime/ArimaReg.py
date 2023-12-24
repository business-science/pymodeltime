
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional, Union
from typing import Optional, Union
from typing import Optional, Union, List

class ArimaReg:
    def __init__(self, seasonal_period: Optional[Union[str, int]] = None,
                 non_seasonal_ar: Optional[int] = None,
                 non_seasonal_differences: Optional[int] = None,
                 non_seasonal_ma: Optional[int] = None,
                 seasonal_ar: Optional[int] = None,
                 seasonal_differences: Optional[int] = None,
                 seasonal_ma: Optional[int] = None,
                 auto_arima: bool = False,
                 trend: Optional[str] = None,
                 uses_date: bool = False):  # Add the uses_date attribute

        self.seasonal_period = seasonal_period
        self.non_seasonal_ar = non_seasonal_ar
        self.non_seasonal_differences = non_seasonal_differences
        self.non_seasonal_ma = non_seasonal_ma
        self.seasonal_ar = seasonal_ar
        self.seasonal_differences = seasonal_differences
        self.seasonal_ma = seasonal_ma
        self.auto_arima = auto_arima
        self.trend = trend
        self.model = None
        self.exog_columns = None
        self.uses_date = uses_date  # Initialize the attribute

        # Dynamically generate the description
        non_seasonal_part = "ARIMA({},{},{})".format(
            self.non_seasonal_ar or 0,
            self.non_seasonal_differences or 0,
            self.non_seasonal_ma or 0
        )

        if self.seasonal_period and (self.seasonal_ar or self.seasonal_differences or self.seasonal_ma):
            seasonal_part = "({},{},{})[{}]".format(
                self.seasonal_ar or 0,
                self.seasonal_differences or 0,
                self.seasonal_ma or 0,
                self.seasonal_period
            )
            self.description = non_seasonal_part + seasonal_part
        else:
            self.description = non_seasonal_part

    # Add other methods of the ArimaReg class here...

    ##

    def fit(self, data: pd.DataFrame, target_column: str, date_column: Optional[str] = None, regressors: Optional[List[str]] = None):
        self.target_column = target_column
        y = data[target_column].astype(float)
        self.exog_columns = regressors if regressors else []

        # Drop date column if it's specified and not used as a regressor
        if date_column and date_column not in self.exog_columns:
            data = data.drop(columns=[date_column], errors='ignore')

        if self.auto_arima:
            # Using pmdarima's auto_arima
            X_exog = data[self.exog_columns].astype(float) if self.exog_columns else None
            self.model = pm.auto_arima(y, X=X_exog, seasonal=True, m=self.seasonal_period or 1,
                                       # ... other parameters ...
                                       trace=True, suppress_warnings=True)
        else:
            # Using statsmodels' ARIMA
            X_exog = data[self.exog_columns].astype(float) if self.exog_columns else None
            order = (self.non_seasonal_ar or 1, self.non_seasonal_differences or 0, self.non_seasonal_ma or 1)
            seasonal_order = (self.seasonal_ar or 1, self.seasonal_differences or 0, self.seasonal_ma or 1, self.seasonal_period or 1)
            self.model = ARIMA(endog=y, exog=X_exog, order=order, seasonal_order=seasonal_order, trend=self.trend).fit()
    def predict(self, X_new: pd.DataFrame) -> np.array:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        # Only use the exogenous columns that were used in training
        if self.exog_columns is not None:
            X_exog = X_new[self.exog_columns].astype(float)
        else:
            X_exog = None

        if self.auto_arima:
            n_periods = len(X_new)
            return self.model.predict(n_periods=n_periods, X=X_exog)
        else:
            if X_exog is not None:
                return self.model.get_forecast(steps=len(X_new), exog=X_exog).predicted_mean
            else:
                return self.model.get_forecast(steps=len(X_new)).predicted_mean

    def forecast(self, horizon, actual_data):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        # Handling the horizon
        if 'years' in horizon:
            num_years = int(horizon.split()[0])
            periods = num_years * 12  # for monthly data
        elif 'months' in horizon:
            periods = int(horizon.split()[0])
        elif isinstance(horizon, int):
            periods = horizon
        else:
            raise ValueError("Unsupported time unit in horizon. Please use 'months' or 'years'.")

        # Creating a date index for the forecast period
        last_date = pd.to_datetime(actual_data['date'].iloc[-1])
        forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]

        # Forecasting
        start = len(actual_data)
        end = start + periods - 1
        forecast_values = self.model.predict(start=start, end=end)

        # Creating the forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted': forecast_values
        })

        return forecast_df

    ##
    def calibrate(self, exog_data, target_data=None):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        # Extract target values if provided
        y_test = target_data[self.target_column] if target_data is not None else None

        # Ensure 'date' column is in the target_data
        if target_data is not None and 'date' not in target_data.columns:
            raise KeyError("'date' column not found in target_data.")

        # Prepare exogenous data
        X_exog = exog_data[self.exog_columns].astype(float) if self.exog_columns else None

        # Ensure the exogenous data shape matches the training shape
        if X_exog is not None and X_exog.shape[1] != len(self.exog_columns):
            raise ValueError(f"The shape of provided exogenous data does not match the training shape. Expected {len(self.exog_columns)} columns, got {X_exog.shape[1]}.")

        # Make predictions using the model
        if self.auto_arima:
            predictions = self.model.predict(n_periods=len(X_exog), X=X_exog)
        else:
            predictions = self.model.get_forecast(steps=len(X_exog), exog=X_exog).predicted_mean

        # Compute residuals if target data is provided
        residuals = y_test - predictions if y_test is not None else [None] * len(predictions)

        # Store calibration data using 'date' from target_data
        self.calibration_data = pd.DataFrame({
            'date': target_data['date'],
            'actual': y_test,
            'predicted': predictions,
            'residuals': residuals
        })



    ##
    def get_model_description(self):
        # Constructing a description based on the available attributes
        description = "ARIMA"
        if self.auto_arima:
            description += " (Auto)"
        if self.seasonal_period:
            description += f", Seasonal Period: {self.seasonal_period}"
        if self.non_seasonal_ar is not None:
            description += f", Non-seasonal AR: {self.non_seasonal_ar}"
        if self.non_seasonal_differences is not None:
            description += f", Non-seasonal Differences: {self.non_seasonal_differences}"
        if self.non_seasonal_ma is not None:
            description += f", Non-seasonal MA: {self.non_seasonal_ma}"
        if self.seasonal_ar is not None:
            description += f", Seasonal AR: {self.seasonal_ar}"
        if self.seasonal_differences is not None:
            description += f", Seasonal Differences: {self.seasonal_differences}"
        if self.seasonal_ma is not None:
            description += f", Seasonal MA: {self.seasonal_ma}"

        return description