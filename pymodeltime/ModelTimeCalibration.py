from ipywidgets import interact
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd

from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg


# Insert the definitions of ArimaReg, ProphetReg, MLModelWrapper, and H2OAutoMLWrapper here
class ModelTimeCalibration:
    def __init__(self, model_time_table, new_data, target_column):
        self.model_time_table = model_time_table
        self.new_data = new_data
        self.target_column = target_column  # New parameter for target column

    def calibrate(self):
        for model in self.model_time_table.models:
            if isinstance(model, ProphetReg):
                self._calibrate_prophet(model)
            elif isinstance(model, ArimaReg):
                self._calibrate_arima(model)
            elif isinstance(model, MLModelWrapper):
                self._calibrate_ml_model(model)
            elif isinstance(model, H2OAutoMLWrapper):
                self._calibrate_h2o_automl(model)

    def _calibrate_prophet(self, model):
        prophet_data = self.new_data.copy()
        prophet_data['ds'] = prophet_data['date']
        regressor_columns = model.regressors if model.regressors else []
        prophet_exog_data = prophet_data[['date', 'ds'] + regressor_columns].dropna()
        predictions = model.predict(prophet_exog_data)
        predictions['predicted'] = predictions['predicted'] 
        merged_data = self._merge_and_calculate_residuals(predictions, 'predicted')
        model.calibration_data = merged_data

    def _calibrate_arima(self, model):
        predictions = model.predict(self.new_data)
        predictions = predictions.to_frame(name='predicted')
        merged_data = self._merge_and_calculate_residuals(predictions, 'predicted')
        model.calibration_data = merged_data

    def _calibrate_ml_model(self, model):
        X = self.new_data[model.feature_names]
        predictions = model.predict(X)
        calibration_data = pd.DataFrame({
            'date': self.new_data['date'],
            self.target_column: self.new_data[self.target_column],
            'predicted': predictions
        })
        model.calibration_data = self._calculate_residuals(calibration_data)

    def _calibrate_h2o_automl(self, model):
        predictions = model.predict(self.new_data)
        predictions['predicted'] = predictions['predict']  # Ensure 'predicted' column
        calibration_data = pd.DataFrame({
            'date': self.new_data['date'],
            self.target_column: self.new_data[self.target_column],
            'predicted': predictions['predicted']
        })
        model.calibration_data = self._calculate_residuals(calibration_data)

    def _calculate_residuals(self, df):
        df['residuals'] = df[self.target_column] - df['predicted']
        return df

    def _merge_and_calculate_residuals(self, predictions, pred_column):
        predictions['date'] = self.new_data['date'].values
        merged_data = pd.merge(self.new_data, predictions[['date', pred_column]], on='date', how='left')
        merged_data['residuals'] = merged_data[self.target_column] - merged_data[pred_column]
        return merged_data[['date', self.target_column, pred_column, 'residuals']]

    def get_calibration_results(self):
        calibration_results = []
        for model in self.model_time_table.models:
            if isinstance(model, MLModelWrapper):
                model_desc = model.description  # Accessing the correct attribute
            else:
                model_desc = getattr(model, 'description', 'Custom Model')

            model_details = {
                '.model_id': getattr(model, 'id', 'N/A'),
                '.model': '<fit[+]>',
                '.model_desc': model_desc,
                '.type': 'ML Model' if isinstance(model, MLModelWrapper) else 'Custom',
                '.calibration_data': "<tibble [{} × 4]>".format(len(model.calibration_data)) if hasattr(model, 'calibration_data') else "None"
            }
            calibration_results.append(model_details)
        return pd.DataFrame(calibration_results)

    @property
    def description(self):
        return self.model_name  #
    def display_calibration_results(self):
        model_ids = [model.id for model in self.model_time_table.models]
        interact(self._view_calibration, model_id=model_ids)

    def _view_calibration(self, model_id):
        model = self.model_time_table.get_model_by_id(model_id)
        if model and hasattr(model, 'calibration_data'):
            display(model.calibration_data)
        else:
            print("Calibration data not available for this model.")