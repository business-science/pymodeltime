from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg

class ModelTimeTable:
    def __init__(self, *models):
        self.models = list(models)
        self.model_descriptions = {}
        self._validate_models()
        self._assign_model_ids()


    def _validate_models(self):
        for model in self.models:
            if not hasattr(model, 'predict'):
                raise ValueError("All objects must be fitted models.")

    ##
    def _assign_model_ids(self):
        for i, model in enumerate(self.models, start=1):
            model.id = i
            self.model_descriptions[model.id] = self._get_model_description(model)



    def add_model(self, model):
        if hasattr(model, 'predict'):
            self.models.append(model)
            self._assign_model_ids()
        else:
            raise ValueError("Added object must be a fitted model.")

    def remove_model(self, model_id):
        self.models = [m for m in self.models if m.id != model_id]
        self._assign_model_ids()

    def update_model(self, model_id, new_model):
        if not hasattr(new_model, 'predict'):
            raise ValueError("New model must be a fitted model.")
        for i, model in enumerate(self.models):
            if model.id == model_id:
                self.models[i] = new_model
                break
        self._assign_model_ids()

    def get_forecast_details(self, model_id):
        model = self.get_model_by_id(model_id)
        if hasattr(model, 'calibration_data'):
            calibration_data = model.calibration_data
            if 'actual' in calibration_data and 'prediction' in calibration_data:
                calibration_data['residuals'] = calibration_data['actual'] - calibration_data['prediction']
                return calibration_data[['date', 'actual', 'prediction', 'residuals']]
            else:
                raise ValueError(f"Expected columns 'actual' and 'prediction' are missing in the calibration data for model ID {model_id}.")
        else:
            raise ValueError(f"No calibration data available for model ID {model_id}.")

    def get_model_by_id(self, model_id):
        for model in self.models:
            if model.id == model_id:
                return model
        raise ValueError(f"Model with ID {model_id} not found.")

    ##
    def _get_model_description(self, model):
        if isinstance(model, ArimaReg):
            desc = "ARIMA({},{},{})".format(
                model.non_seasonal_ar or 0,
                model.non_seasonal_differences or 0,
                model.non_seasonal_ma or 0
            )
            if model.seasonal_period:
                desc += "({},{},{})[{}]".format(
                    model.seasonal_ar or 0,
                    model.seasonal_differences or 0,
                    model.seasonal_ma or 0,
                    model.seasonal_period
                )
            return desc
        elif isinstance(model, ProphetReg):
            return "PROPHET"
        elif isinstance(model, MLModelWrapper):
            return model.model_name  # Assuming MLModelWrapper has a 'model_name' attribute
        elif isinstance(model, H2OAutoMLWrapper):
            return "H2O AutoML"  # or any other descriptive string for H2O AutoML
        else:
            return str(model)  # Fallback for other types



    ##
    def fit_models(self, train_data):
        for model in self.models:
            if isinstance(model, ArimaReg):
                model.fit(train_data[['date']], train_data['value'])
            elif isinstance(model, ProphetReg):
                prophet_df = train_data.rename(columns={'date': 'ds', 'value': 'y'})
                model.fit(prophet_df)


    def generate_forecast_data(self, model):
        if isinstance(model, ProphetReg):
            future_df = model.model.make_future_dataframe(periods=self.h) if self.h is not None else self.new_data
        elif isinstance(model, ArimaReg):
            future_df = self.new_data  # or generate appropriate data frame for ARIMA model
        else:
            raise ValueError("Unsupported model type for forecasting.")
        return future_df

    ##
    def print_calibration_results(self):
        print("# A tibble: {} × 5".format(len(self.models)))
        print("  .model_id .model   .model_desc             .type .calibration_data")

        for model in self.models:
            model_id = model.id
            model_desc = self.model_descriptions.get(model_id, "Unknown")
            model_type = "Test" if hasattr(model, 'calibration_data') and model.calibration_data is not None else "None"
            calibration_data_summary = "<tibble [{} × 4]>".format(len(model.calibration_data)) if model_type == "Test" else "None"

            print(f"      {model_id:<8} <fit[+]>   {model_desc:<20} {model_type:<6} {calibration_data_summary}")



