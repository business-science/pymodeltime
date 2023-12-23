import h2o
from h2o.automl import H2OAutoML

class H2OAutoMLWrapper:
    def __init__(self, automl_model, target_column):
        self.model = automl_model
        self.target_column = target_column  # Store the target column
        h2o.init()  # Initialize H2O

    def predict(self, new_data):
        # Convert date to string in pandas, if present
        if 'date' in new_data.columns:
            new_data['date'] = new_data['date'].astype(str)

        # Convert the Pandas DataFrame to an H2O frame
        h2o_new_data = h2o.H2OFrame(new_data)

        # Exclude 'id' and 'date' from features used for prediction
        predictors = [col for col in h2o_new_data.columns if col not in ['id', 'date']]

        # Make predictions
        preds = self.model.predict(h2o_new_data[predictors])

        # Return predictions as a pandas DataFrame
        predictions_df = preds.as_data_frame()
        predictions_df.index = new_data.index  # Align indices
        if 'id' in new_data.columns:
            predictions_df['id'] = new_data['id']
        if 'date' in new_data.columns:
            predictions_df['date'] = new_data['date']

        return predictions_df

    @property
    def description(self):
        return "H2O AutoML"