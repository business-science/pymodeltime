from prophet import Prophet
import pandas as pd
class ProphetReg:
    def __init__(self, growth=None, changepoint_num=None, changepoint_range=0.8,
                 seasonality_yearly=True, seasonality_weekly=True, seasonality_daily=True,
                 season=None, prior_scale_changepoints=0.05, prior_scale_seasonality=10.0,
                 prior_scale_holidays=10.0, logistic_cap=None, logistic_floor=None,
                 regressors_prior_scale=1e4, regressors_standardize="auto", regressors_mode=None,
                 interval_width=0.95):  # Add this line
        self.growth = growth if growth is not None else 'linear'
        self.changepoint_num = changepoint_num
        self.changepoint_range = changepoint_range
        self.seasonality_yearly = seasonality_yearly
        self.seasonality_weekly = seasonality_weekly
        self.seasonality_daily = seasonality_daily
        self.interval_width = interval_width  # Add this line
        self.season = season if season is not None else 'additive'
        self.prior_scale_changepoints = prior_scale_changepoints
        self.prior_scale_seasonality = prior_scale_seasonality
        self.prior_scale_holidays = prior_scale_holidays
        self.logistic_cap = logistic_cap
        self.logistic_floor = logistic_floor
        self.regressors_prior_scale = regressors_prior_scale
        self.regressors_standardize = regressors_standardize
        self.regressors_mode = regressors_mode
        self.regressors = None
        self.model = None
        self.description = "PROPHET"

    def fit(self, df: pd.DataFrame, target_column='value', date_column='date', regressors=None):
        self.target_column = target_column
        self.regressors = regressors
        df = df.rename(columns={date_column: 'ds', target_column: 'y'})
        self.model = Prophet(
            growth=self.growth,
            changepoints=self.changepoint_num,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.seasonality_yearly,
            weekly_seasonality=self.seasonality_weekly,
            daily_seasonality=self.seasonality_daily,
            seasonality_mode=self.season,
            changepoint_prior_scale=self.prior_scale_changepoints,
            seasonality_prior_scale=self.prior_scale_seasonality,
            holidays_prior_scale=self.prior_scale_holidays,
            interval_width=self.interval_width
        )
        if self.growth == 'logistic':
            df['cap'] = self.logistic_cap
            df['floor'] = self.logistic_floor
        if self.regressors:
            for regressor in self.regressors:
                self.model.add_regressor(regressor, prior_scale=self.regressors_prior_scale,
                                         standardize=self.regressors_standardize, mode=self.regressors_mode)
        self.model.fit(df)



    ##
    def predict(self, future_data):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        forecast = self.model.predict(future_data)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={'ds': 'date', 'yhat': 'predicted', 'yhat_lower': 'conf_lo', 'yhat_upper': 'conf_hi'})


    ##
    def get_model_description(self):
        # Return a string that describes the Prophet model's configuration
        return "PROPHET"  # Adjust this based on the model's parameters

    ##
    def calibrate(self, exog_data, target_data=None):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        # Ensure 'ds' column is present for Prophet
        if 'ds' not in exog_data.columns:
            raise KeyError("'ds' column not found in exog_data DataFrame.")

        # Create a future DataFrame for prediction
        future = self.model.make_future_dataframe(periods=0)  # No additional future periods
        future = future.head(len(exog_data))  # Truncate to the length of exog_data

        # Assign regressor values to the future DataFrame
        for regressor in self.regressors:
            if regressor in exog_data.columns:
                future[regressor] = exog_data[regressor].values
            else:
                raise KeyError(f"Regressor '{regressor}' not found in exog_data DataFrame.")

        forecast = self.model.predict(future)

        # Extract the relevant forecast data
        predictions = forecast['yhat']

        # Compute residuals if the target column is present in target_data
        if target_data is not None and self.target_column in target_data.columns:
            residuals = target_data[self.target_column] - predictions
            actual = target_data[self.target_column]
        else:
            residuals = [None] * len(predictions)
            actual = [None] * len(predictions)

        # Store the calibration results
        self.calibration_data = pd.DataFrame({
            'date': exog_data['date'],
            'actual': actual,
            'predicted': predictions,
            'residuals': residuals
        })


