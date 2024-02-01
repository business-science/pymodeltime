import pandas as pd
import mlforecast
from mlforecast.utils import PredictionIntervals
from mlforecast.core import Lags, LagTransforms, DateFeature
import lightgbm as lgb
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression


def mae(y_true, y_pred):
    return (y_true - y_pred).abs().mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

def mape(y_true, y_pred):
    return ((y_true - y_pred).abs() / y_true.abs()).mean()

def smape(y_true, y_pred):
    denominator = (y_true.abs() + y_pred.abs())
    diff = (y_true - y_pred).abs() / denominator
    diff[denominator == 0] = 0  # handle the case where the denominator is zero
    return 2 * diff.mean()





class MLForecastWrapper:
    def __init__(self, models, freq, target_transforms, lags, lag_transforms, date_features, target_col):
        self.models = models
        self.model_id = None
        self.freq = freq
        self.target_transforms = target_transforms
        self.lags = lags
        self.lag_transforms = lag_transforms
        self.date_features = date_features
        self.target_col = target_col  # Store the target column name

    #


    def fit(self, df, unique_id_col, ds_col, n_windows=None, h=None, generate_intervals=True):
        # Renaming columns to match MLForecast's expected format
        df = df.rename(columns={unique_id_col: 'unique_id', ds_col: 'ds', self.target_col: 'y'})

        # Set default values for n_windows and h if not provided
        n_windows = n_windows or 4  # Default value, adjust as needed
        h = h or 24  # Default value, adjust as needed

        # Initializing MLForecast
        self.mlf = mlforecast.MLForecast(
            models=self.models,
            freq=self.freq,
            target_transforms=self.target_transforms,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features
        )

        # Fitting the model with custom prediction intervals
        self.mlf.fit(
            df=df,
            prediction_intervals=PredictionIntervals(n_windows=n_windows, h=h) if generate_intervals else None
        )


    ##
    def get_model_details(self):
          model_details = {
              '.model_id': self.model_id,
              '.model': '<fit[+]>',
              '.model_desc': ", ".join([model.__class__.__name__ for model in self.models.values()]),
              '.type': 'ML Model'
          }
          return model_details

    def predict_with_intervals(self, h, levels):
        # Method to generate predictions with confidence intervals
        if self.mlf is None:
            raise ValueError("Model not trained. Call fit() before predicting.")
        try:
            return self.mlf.predict(h=h, level=levels)
        except Exception as e:
            print(f"Error in generating predictions with intervals: {e}")
            return None
    ##
    def predict_future(self, h, levels=None):
        """
        Generate future predictions with optional confidence intervals.
        """
        if self.mlf is None:
            raise ValueError("Model not trained. Call fit() before predicting.")

        try:
            if levels:
                predictions = self.mlf.predict(h=h, level=levels)
            else:
                predictions = self.mlf.predict(h=h)
            return predictions
        except Exception as e:
            print(f"Error in generating future predictions: {e}")
            return None




    def predict(self, h, levels=None):
        if self.mlf is None:
            raise ValueError("Model not trained. Call fit() before predicting.")

        # Generate predictions with or without confidence intervals
        try:
            if levels:
                predictions = self.mlf.predict(h=h, level=levels)
            else:
                predictions = self.mlf.predict(h=h)
            return predictions
        except Exception as e:
            print(f"Error in generating predictions: {e}")
            return None



    def calibrate(self, test_data, target_col):
        # Rename 'date' to 'ds' and target column to 'actual'
        test_data = test_data.rename(columns={'date': 'ds', target_col: 'actual'})

        # Convert 'ds' to date only (without time)
        test_data['ds'] = test_data['ds'].dt.date

        # Generate forecasts
        forecasts = self.predict(h=len(test_data), levels=[0.95])

        # Ensure 'ds' in forecasts is also date only
        if 'ds' in forecasts.columns:
            forecasts['ds'] = pd.to_datetime(forecasts['ds']).dt.date

        # Merge forecasts with test data
        test_data = test_data.merge(forecasts, how='left', on=['unique_id', 'ds'])

        # Prepare calibration data for each model
        self.calibration_data = {}
        for model_name in self.models.keys():
            calibration_data = test_data[['unique_id', 'ds', 'actual', model_name]].copy()
            calibration_data['predicted'] = calibration_data[model_name]
            calibration_data['residuals'] = calibration_data['actual'] - calibration_data['predicted']

            # Store calibration data for each model
            self.calibration_data[model_name] = calibration_data[['unique_id', 'ds', 'actual', 'predicted', 'residuals']]


    ##
    def get_model_details(self):
          model_details = {
              '.model': '<fit[+]>',
              '.model_desc': ", ".join([model.__class__.__name__ for model in self.models.values()]),
              '.type': 'ML Model'
          }
          return model_details


    ##



    def evaluate(self, test_df, unique_id_col, date_col, target_col, metrics):
          # Internally rename 'date' to 'ds'
          test_df = test_df.rename(columns={unique_id_col: 'unique_id', date_col: 'ds'})

          # Generate forecasts
          forecasts = self.predict(h=len(test_df), levels=[0.95])
          test_df = test_df.merge(forecasts, how='left', on=['unique_id', 'ds'])

          # Prepare results
          results = {}
          unique_ids = test_df['unique_id'].unique()
          for unique_id in unique_ids:
              unique_id_df = test_df[test_df['unique_id'] == unique_id]
              for metric in metrics:
                  metric_name = metric.__name__
                  for model_name in self.models.keys():  # Iterate over model names
                      key = (unique_id, metric_name)
                      if key not in results:
                          results[key] = {'unique_id': unique_id, 'metric': metric_name}
                      actual_values = unique_id_df[target_col]
                      predicted_values = unique_id_df.get(model_name)
                      if predicted_values is not None:
                          results[key][model_name] = metric(actual_values, predicted_values)

          # Convert the results dictionary to DataFrame
          return pd.DataFrame(results.values())


