
from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg
from autogluon.tabular import TabularPredictor
from .AutoGluonTabularWrapper import AutoGluonTabularWrapper
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

import h2o

class ModelTimeRefit:
    def __init__(self, modeltime_table, verbose=False, parallel=False, max_workers=None):
        self.models = modeltime_table.models if hasattr(modeltime_table, 'models') else [modeltime_table]
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers

    def _refit_auto_gluon_tabular(self, model, data, target_column):
        try:
            model.fit(data)
            model.refit_full(data, target_column)
        except Exception as e:
            if self.verbose:
                print(f"Error refitting AutoGluonTabular model {model}: {e}")

    ##
    def _refit_model(self, model, data, target_column):
        try:
            if isinstance(model, AutoGluonTabularWrapper):
                # Handle refitting for AutoGluonTabularWrapper
                self._refit_auto_gluon_tabular(model, data, target_column)

            elif isinstance(model, ArimaReg):
                # Fit and refit for ArimaReg
                model.fit(data, target_column=target_column, date_column='date')

            elif isinstance(model, ProphetReg):
                # Fit and refit for ProphetReg
                model.fit(data, target_column=target_column, date_column='date')

            elif isinstance(model, MLModelWrapper):
                # Fit and refit for MLModelWrapper
                X = data[model.feature_names]
                y = data[target_column]
                model.fit(X, y)

            elif isinstance(model, H2OAutoMLWrapper):
                # Fit and refit for H2OAutoMLWrapper, and update with the new leader model
                new_leader = self._refit_h2o_automl(model, data, target_column)
                model.model = new_leader  # Update the model in the wrapper

            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            if self.verbose:
                print(f"Model successfully refitted: {model}")
            return model

        except Exception as e:
            if self.verbose:
                print(f"Error refitting model {model}: {e}")
            return None
   
   ##
 

    
    
    ##
    def _refit_h2o_automl(self, h2o_model, data, target_column, max_models=2, seed=1, max_runtime_secs=3600):
        h2o.init()
        h2o_data = h2o.H2OFrame(data)
        predictors = [col for col in h2o_data.columns if col != target_column]

        automl = H2OAutoML(max_models=max_models, seed=seed, max_runtime_secs=max_runtime_secs)
        automl.train(x=predictors, y=target_column, training_frame=h2o_data)

        # Print the leaderboard immediately after training
        lb = automl.leaderboard
        print("H2O AutoML Leaderboard after refitting:")
        lb.head(rows=lb.nrows).as_data_frame().to_csv("h2o_automl_leaderboard.csv")
        lb.head(rows=lb.nrows)  # Display all rows of the leaderboard

        return automl.leader


    def forecast_h2o_automl(self, h2o_model, actual_data):
        h2o_data = h2o.H2OFrame(actual_data)
        h2o_predictions = h2o_model.model.predict(h2o_data)
        predictions = h2o_predictions.as_data_frame()
        predictions.rename(columns={'predict': 'predicted'}, inplace=True)
        return predictions




    def refit_models(self, new_data, target_column):
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self.models = list(executor.map(lambda m: self._refit_model(m, new_data, target_column), self.models))
        else:
            self.models = [self._refit_model(model, new_data, target_column) for model in self.models]
        return self.models

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

    ##

    def forecast(self, actual_data, target_column):
        actual_data = self._filter_actual_data(actual_data, target_column)
        self.forecast_results = {}

        for model in self.models:
            if model is None:
                continue

            try:
                predictions = None
                model_type = self._get_model_type(model)  # Retrieve model type

                if isinstance(model, AutoGluonTabularWrapper):
                    X_test = actual_data.drop(columns=[target_column])
                    predictions = model.predict(X_test)
                    predictions = predictions.to_frame(name='predicted') if isinstance(predictions, pd.Series) else predictions

                elif isinstance(model, ProphetReg):
                    prophet_data = actual_data.rename(columns={'date': 'ds'})
                    predictions = model.predict(prophet_data)
                    predictions.rename(columns={'yhat': 'predicted'}, inplace=True)

                elif isinstance(model, ArimaReg):
                    predictions = model.predict(actual_data)
                    predictions = predictions.to_frame(name='predicted')

                elif isinstance(model, H2OAutoMLWrapper):
                    h2o_data = h2o.H2OFrame(actual_data.drop(columns=[target_column]))
                    h2o_predictions = model.model.predict(h2o_data)
                    predictions = h2o_predictions.as_data_frame()
                    predictions.rename(columns={'predict': 'predicted'}, inplace=True)

                elif isinstance(model, MLModelWrapper):
                    X = actual_data.drop(columns=[target_column])
                    predictions_raw = model.predict(X)
                    # Ensure predictions are in a DataFrame
                    if isinstance(predictions_raw, np.ndarray):
                        predictions = pd.DataFrame(predictions_raw, columns=['predicted'])
                    elif isinstance(predictions_raw, pd.Series):
                        predictions = predictions_raw.to_frame(name='predicted')
                    else:
                        predictions = predictions_raw

                # Verify the length of predictions matches actual_data
                if len(predictions) != len(actual_data):
                    raise ValueError("Mismatch in length of predictions and actual data.")

                # Aligning predictions with actual data
                aligned_forecast = pd.DataFrame({
                    'date': actual_data['date'],
                    target_column: actual_data[target_column],
                    'predicted': predictions['predicted'].values,
                    'residuals': actual_data[target_column] - predictions['predicted'],
                    'model_type': model_type  # Use model type
                })

                self.forecast_results[model] = aligned_forecast

            except Exception as e:
                print(f"Error in forecasting with model {model}: {e}")
                self.forecast_results[model] = None

        return self.forecast_results




    ##
    def _filter_actual_data(self, data, target_column):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        # Convert 'date' column to Timestamps if it exists
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])

        filtered_data = data.dropna(subset=[target_column])
        latest_actual_date = filtered_data['date'].max()
        return filtered_data[filtered_data['date'] <= latest_actual_date]



    def get_model_summary(self):
        model_summary = []
        for model in self.models:
            calibration_data_status = 'Available' if model in self.forecast_results else 'Not Available'
            model_summary.append({
                '.model_id': id(model),
                '.model': type(model).__name__,
                '.model_desc': str(model),
                '.type': 'Refitted',
                '.calibration_data': calibration_data_status
            })
        return pd.DataFrame(model_summary)
