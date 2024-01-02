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
        # Convert 'date' column to datetime.date format if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date']).dt.date

        # Identify potential grouping column
        group_column = self._find_group_column(data)

        if group_column:
            return self._forecast_grouped_data(data, group_column)
        else:
            return self._forecast_individual(data)


    def _find_group_column(self, data):
        # Heuristic to find a potential grouping column
        # This could be refined based on your specific data and requirements
        for col in data.columns:
            if data[col].dtype == 'object' or col == 'Dept':
                return col
        return None


    ##
    def _forecast_grouped_data(self, data, group_column):
        grouped = data.groupby(group_column)
        all_forecasts = []

        for _, group in grouped:
            # Debugging: Check 'date' type in each group
            print(f"Group '{group_column}' 'date' type:", group['date'].dtype)

            forecast_df = self._forecast_individual(group)
            all_forecasts.append(forecast_df)

        return pd.concat(all_forecasts, ignore_index=True)



    ##
    def _forecast_individual(self, data):
        # Check if 'date' and 'id' columns exist
        if 'date' not in data.columns or 'id' not in data.columns:
            raise ValueError("Data must contain 'date' and 'id' columns")

        # Standardize 'date' to Timestamp format
        data['date'] = pd.to_datetime(data['date'])

        # Extract 'date' and 'id' before prediction
        date_series = data['date']
        id_series = data['id']

        # Prepare data for prediction
        X = data[self.feature_names]

        # Generate predictions
        predictions = self.predict(X)

        return pd.DataFrame({
            'prediction': predictions,
            'date': date_series,
            'id': id_series
        })





