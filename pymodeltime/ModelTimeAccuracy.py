from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np



from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg


class ModelTimeAccuracy:
    def __init__(self, model_time_table, new_data, target_column, metrics=None):
        self.model_time_table = model_time_table
        self.new_data = new_data
        self.target_column = target_column
        self.metrics = metrics if metrics is not None else ['mae', 'rmse', 'r2']

    def calculate_accuracy(self):
        results = []

        for model in self.model_time_table.models:
            forecast_data = self.generate_forecast_data(model)
            predictions_df = model.predict(forecast_data)

            if isinstance(predictions_df, pd.DataFrame):
                predictions_column = 'predicted' if 'predicted' in predictions_df.columns else predictions_df.columns[0]
                predictions = predictions_df[predictions_column]
            else:
                predictions = predictions_df

            predictions = np.array(predictions).astype(float)
            actual = np.array(self.new_data[self.target_column]).astype(float)

            model_accuracy = {
                'model_id': getattr(model, 'id', 'N/A'),
                'model_desc': self._get_model_description(model),
                'mae': mean_absolute_error(actual, predictions) if 'mae' in self.metrics else None,
                'rmse': np.sqrt(mean_squared_error(actual, predictions)) if 'rmse' in self.metrics else None,
                'r2': r2_score(actual, predictions) if 'r2' in self.metrics else None
            }
            results.append(model_accuracy)

        return pd.DataFrame(results)

    def _get_model_description(self, model):
        if isinstance(model, MLModelWrapper):
            return model.model_name  # Assuming MLModelWrapper has a 'model_name' attribute
        else:
            return getattr(model, 'description', 'N/A')

    def generate_forecast_data(self, model):
        forecast_data = self.new_data.copy()
        if isinstance(model, ProphetReg):
            forecast_data = forecast_data.rename(columns={'date': 'ds'})
        return forecast_data
