import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg

class ModelTimeRefit:
    def __init__(self, modeltime_table, verbose=False, parallel=False, max_workers=None):
        self.models = modeltime_table.models if hasattr(modeltime_table, 'models') else [modeltime_table]
        self.verbose = verbose
        self.parallel = parallel
        self.max_workers = max_workers

    def _refit_model(self, model, data, target_column):
        try:
            if isinstance(model, (ArimaReg, ProphetReg)):
                model.fit(data, target_column=target_column, date_column='date')
            elif isinstance(model, MLModelWrapper):
                X = data[model.feature_names]
                y = data[target_column]
                model.fit(X, y)
            elif isinstance(model, H2OAutoMLWrapper):
                self._refit_h2o_automl(model, data, target_column)
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
    def _refit_h2o_automl(self, h2o_model, data, target_column, max_models=5, seed=1, max_runtime_secs=3600):
        h2o.init()
        h2o_data = h2o.H2OFrame(data)
        predictors = [col for col in h2o_data.columns if col != target_column]

        automl = H2OAutoML(max_models=max_models, seed=seed, max_runtime_secs=max_runtime_secs)
        automl.train(x=predictors, y=target_column, training_frame=h2o_data)

        h2o_model.model = automl.leader

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
    def forecast(self, actual_data, target_column):
        actual_data = self._filter_actual_data(actual_data, target_column)
        self.forecast_results = {}

        for model in self.models:
            if model is None:
                continue

            try:
                if isinstance(model, ProphetReg):
                    # Prophet forecasting
                    prophet_data = actual_data.rename(columns={'date': 'ds'})
                    predictions = model.predict(prophet_data)
                    predictions.rename(columns={'ds': 'date'}, inplace=True)
                    model_type = 'Prophet'
                elif isinstance(model, ArimaReg):
                    # ARIMA forecasting
                    predictions = model.predict(actual_data)
                    predictions = predictions.to_frame(name='predicted')
                    model_type = 'ARIMA'
                elif isinstance(model, H2OAutoMLWrapper):
                    # H2O AutoML forecasting
                    h2o_data = h2o.H2OFrame(actual_data)
                    h2o_predictions = model.model.predict(h2o_data)
                    predictions = h2o_predictions.as_data_frame()
                    predictions.rename(columns={'predict': 'predicted'}, inplace=True)
                    model_type = 'H2O AutoML'
                elif isinstance(model, MLModelWrapper):
                    # Other ML models forecasting
                    X = actual_data[model.feature_names]
                    predictions = model.predict(X)
                    if not isinstance(predictions, pd.DataFrame):
                        predictions = pd.DataFrame(predictions, columns=['predicted'])
                    model_type = model.model_name
                else:
                    raise ValueError(f"Unsupported model type: {type(model)}")

                aligned_forecast = actual_data[['date', target_column]].copy()
                aligned_forecast['predicted'] = predictions['predicted'].values
                aligned_forecast['residuals'] = aligned_forecast[target_column] - aligned_forecast['predicted']
                aligned_forecast['model_id'] = model_type
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

