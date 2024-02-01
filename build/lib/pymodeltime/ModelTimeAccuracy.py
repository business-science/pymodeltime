from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg
from .AutoGluonTabularWrapper import AutoGluonTabularWrapper
from .MLForecastWrapper import MLForecastWrapper



# Additional Functions for MAPE and SMAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


class ModelTimeAccuracy:
    current_model_id = 0  # Class variable to track model ID

    def __init__(self, model_time_table, new_data, target_column, metrics=None):
        self.model_time_table = model_time_table
        self.new_data = new_data
        self.target_column = target_column
        self.metrics = metrics if metrics is not None else ['mae', 'rmse', 'r2', 'mape', 'smape']
        ModelTimeAccuracy.current_model_id = 0  # Reset the counter for each instance


    ##
    def calculate_accuracy(self):
        results = []

        for model in self.model_time_table.models:
            forecast_data = self.generate_forecast_data(model)

            if isinstance(model, MLForecastWrapper):
                horizon = len(forecast_data)
                predictions_df = model.predict(horizon, levels=[0.95])

                # Calculate accuracy for each sub-model within MLForecastWrapper
                for sub_model_name in model.models.keys():
                    if sub_model_name in predictions_df.columns:
                        self._calculate_and_append_metrics(
                            model, sub_model_name,
                            predictions_df[sub_model_name], results, forecast_data
                        )
            else:
                self._handle_non_mlforecast_model(model, forecast_data, results)

        return pd.DataFrame(results)

    ##
    def _calculate_and_append_metrics(self, model, sub_model_name, predictions, results, forecast_data):
        predictions = np.array(predictions).astype(float)
        actual = np.array(forecast_data[self.target_column]).astype(float)

        # Generate a unique model ID for each sub-model in MLForecastWrapper

        model_id = ModelTimeAccuracy.current_model_id
        ModelTimeAccuracy.current_model_id += 1
        #model_id = getattr(model, 'id', 'N/A')
        if isinstance(model, MLForecastWrapper):
            model_id = f"{model_id}_{sub_model_name}"

        model_accuracy = {
            'model_id': model_id,
            'model_desc': sub_model_name,
            'mae': mean_absolute_error(actual, predictions) if 'mae' in self.metrics else None,
            'rmse': np.sqrt(mean_squared_error(actual, predictions)) if 'rmse' in self.metrics else None,
            'r2': r2_score(actual, predictions) if 'r2' in self.metrics else None,
            'mape': mean_absolute_percentage_error(actual, predictions) if 'mape' in self.metrics else None,
            'smape': symmetric_mean_absolute_percentage_error(actual, predictions) if 'smape' in self.metrics else None,
        }
        results.append(model_accuracy)


    def _handle_non_mlforecast_model(self, model, forecast_data, results):
        predictions_df = model.predict(forecast_data)
        if isinstance(predictions_df, pd.DataFrame):
            numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                predictions = predictions_df[numeric_cols[0]]
            else:
                raise ValueError("No numeric prediction columns found in predictions DataFrame.")
        else:
            predictions = predictions_df

        predictions = np.array(predictions).astype(float)
        actual = np.array(forecast_data[self.target_column]).astype(float)

        model_accuracy = {
            'model_id': getattr(model, 'id', 'N/A'),
            'model_desc': self._get_model_description(model),
            'mae': mean_absolute_error(actual, predictions) if 'mae' in self.metrics else None,
            'rmse': np.sqrt(mean_squared_error(actual, predictions)) if 'rmse' in self.metrics else None,
            'r2': r2_score(actual, predictions) if 'r2' in self.metrics else None,
            'mape': mean_absolute_percentage_error(actual, predictions) if 'mape' in self.metrics else None,
            'smape': symmetric_mean_absolute_percentage_error(actual, predictions) if 'smape' in self.metrics else None,
        }
        results.append(model_accuracy)




    ##
    def generate_forecast_data(self, model):
        forecast_data = self.new_data.copy()
        if isinstance(model, ProphetReg):
            forecast_data = forecast_data.rename(columns={'date': 'ds'})
        elif isinstance(model, MLForecastWrapper):
            # Exclude or transform non-numerical columns for MLForecastWrapper
            # For example, drop the 'unique_id' column if it's not needed
            forecast_data = forecast_data.drop(columns=['unique_id'], errors='ignore')
        # Add similar conditions for other model types if necessary
        return forecast_data

    def _get_model_description(self, model):
        if isinstance(model, MLModelWrapper):
            return model.model_name  # Assuming MLModelWrapper has a 'model_name' attribute
        elif isinstance(model, AutoGluonTabularWrapper):
            return model.get_actual_model_name()  # For AutoGluonTabularWrapper
        elif isinstance(model, MLForecastWrapper):
            # Implement logic to return a description for MLForecastWrapper
            model_names = [m.__class__.__name__ for m in model.models.values()]
            return ', '.join(model_names)
        elif isinstance(model, H2OAutoMLWrapper):
            return self._get_model_type(model)  # For H2OAutoMLWrapper
        else:
            return getattr(model, 'description', 'N/A')  # Default case



    def get_actual_model_name(self):
        # Assuming there's a method or attribute in AutoGluonTabularWrapper that gives the actual model name
        return self.actual_model_name if hasattr(self, 'actual_model_name') else 'AutoGluonTabular'


    ##
    def _get_model_type(self, model):
        """ Utility function to get the type of the model. """
        if isinstance(model, AutoGluonTabularWrapper):
            return 'AutoGluonTabular'
        elif isinstance(model, ProphetReg):
            return 'Prophet'
        elif isinstance(model, ArimaReg):
            return 'ARIMA'
        elif isinstance(model, H2OAutoMLWrapper):
            return 'H2O AutoML' if model.model is not None else 'H2O AutoML (Untrained)'
        elif isinstance(model, MLModelWrapper):
            return model.model.__class__.__name__  # Get the class name of the underlying model

        elif isinstance(model, MLForecastWrapper):
            return 'MLForecast Wrapper'  # New case for MLForecastWrapper
        else:
            return 'Unknown Model'

    def generate_forecast_data(self, model):
        forecast_data = self.new_data.copy()
        if isinstance(model, ProphetReg):
            forecast_data = forecast_data.rename(columns={'date': 'ds'})
        return forecast_data