
class MLModelWrapper:
    _id_counter = 1  # Class-level counter for generating unique model IDs

    def __init__(self, model, feature_names, model_name):
        self.model = model
        self.feature_names = feature_names
        self.model_id = MLModelWrapper._id_counter
        self.model_name = model_name
        MLModelWrapper._id_counter += 1

    @property
    def description(self):
        return self.model_name  # Use model_name as the description


    ##
    def calibrate(self, X, y):
        # Generate predictions
        predictions = self.predict(X)

        # Prepare the calibration DataFrame
        calibration_data = X.copy()
        calibration_data['actual'] = y
        calibration_data['predicted'] = predictions
        calibration_data['residuals'] = calibration_data['actual'] - calibration_data['predicted']

        # Store calibration data in the wrapper
        self.calibration_data = calibration_data

        return calibration_data
    def fit(self, X, y):
        self.model.fit(X[self.feature_names], y)

    def predict(self, X):
        return self.model.predict(X[self.feature_names])

    def get_model_details(self):
        return {
            '.model_id': self.model_id,
            '.model': '<fit[+]>',
            '.model_desc': self.model_name,
            '.type': 'ML Model'
        }

    ##
    def forecast_with_date_id(self, data):
        # Check if 'date' and 'id' columns exist
        if 'date' not in data.columns or 'id' not in data.columns:
            raise ValueError("Data must contain 'date' and 'id' columns")

        # Extract 'date' and 'id' before prediction
        date_series = data['date']
        id_series = data['id']

        # Prepare data for prediction (excluding 'date' and 'id')
        X = data[self.feature_names]

        # Generate predictions
        predictions = self.predict(X)

        # Combine 'date', 'id', and 'predictions' into a DataFrame
        forecast_df = pd.DataFrame({
            'prediction': predictions,
            'date': date_series,
            'id': id_series
        })

        return forecast_df

