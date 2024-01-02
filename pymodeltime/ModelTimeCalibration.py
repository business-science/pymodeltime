
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

from .AutoGluonTabularWrapper import AutoGluonTabularWrapper



class ModelTimeCalibration:
    def __init__(self, model_time_table, new_data, target_column):
        self.model_time_table = model_time_table
        self.new_data = new_data
        self.target_column = target_column  # New parameter for target column

    def calibrate(self):
        for model in self.model_time_table.models:

            if isinstance(model, AutoGluonTabularWrapper):
                X = self.new_data.drop(columns=[self.target_column])
                y = self.new_data[self.target_column]
                model.calibrate(X, y)

            elif isinstance(model, ProphetReg):
                self._calibrate_prophet(model)
            elif isinstance(model, ArimaReg):
                self._calibrate_arima(model)
            elif isinstance(model, MLModelWrapper):
                self._calibrate_ml_model(model)
            elif isinstance(model, H2OAutoMLWrapper):
                self._calibrate_h2o_automl(model)

    ##
    def _calibrate_auto_gluon_tabular(self, model):
        print("Starting calibration for AutoGluonTabular model...")  # Debug print

        if model.predictor is None:
            print("AutoGluonTabular model is not trained. Skipping calibration.")
            return

        X = self.new_data.drop(columns=[self.target_column])
        y = self.new_data[self.target_column]

        try:
            predictions = model.predict(X)
            residuals = y - predictions

            calibration_data = pd.DataFrame({
                'actual': y,
                'predicted': predictions,
                'residuals': residuals
            })

            model.calibration_data = calibration_data
            print("Calibration data set successfully for AutoGluonTabular model.")
        except Exception as e:
            print(f"Error during prediction in AutoGluonTabular model: {e}")

        if hasattr(model, 'calibration_data'):
            print("Calibration data:", model.calibration_data.head())  # Display first few rows of calibration data
        else:
            print("Calibration data NOT set for AutoGluonTabular model.")






    def update_model_calibration_data(self, model_id, calibration_data):
        """
        Update the calibration data of a specific model in the model time table.

        Parameters:
        model_id: The ID of the model to update.
        calibration_data: The new calibration data for the model.
        """
        for model in self.model_time_table.models:
            if getattr(model, 'model_id', None) == model_id:
                model.calibration_data = calibration_data
                print(f"Calibration data updated for model ID {model_id}.")
                return
        print(f"No model found with ID {model_id} to update.")

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

    ##
    def _get_model_type(self, model):
        """ Utility function to get the type of the model. """
        if isinstance(model, AutoGluonTabularWrapper):
            return model.get_best_model() if hasattr(model, 'get_best_model') else 'AutoGluonTabular'
        elif isinstance(model, ProphetReg):
            return 'Prophet'
        elif isinstance(model, ArimaReg):
            return 'ARIMA'
        elif isinstance(model, H2OAutoMLWrapper):
            return model.model.model_id if model.model is not None else 'H2O AutoML'
        elif isinstance(model, MLModelWrapper):
            return model.model.__class__.__name__  # Get the class name of the underlying model
        else:
            return 'Unknown Model'

    def get_calibration_results(self):
        calibration_results = []
        for model in self.model_time_table.models:
            # Retrieve the actual model name for AutoGluonTabular models
            if isinstance(model, H2OAutoMLWrapper):
                model_desc = self._get_model_type(model)
            elif isinstance(model, AutoGluonTabularWrapper) and hasattr(model, 'actual_model_name'):
                model_desc = self._get_model_type(model)
            elif isinstance(model, MLModelWrapper):
                model_desc = model.description
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

    ##

    def _view_calibration(self, model_id):
        model = self.model_time_table.get_model_by_id(model_id)
        if model and hasattr(model, 'calibration_data'):
            display(model.calibration_data)
        else:
            print("Calibration data not available for this model.")
