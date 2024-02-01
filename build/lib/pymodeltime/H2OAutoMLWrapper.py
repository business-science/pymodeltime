import h2o
from h2o.automl import H2OAutoML

class H2OAutoMLWrapper:
    def __init__(self, automl_model, target_column):
        self.model = automl_model
        self.target_column = target_column  # Store the target column
        h2o.init()  # Initialize H2O

    def predict(self, new_data, dept=None):
        # Convert date to string in pandas, if present
        if 'date' in new_data.columns:
            new_data['date'] = new_data['date'].astype(str)

        # If a department is specified, filter the data for that department
        if dept is not None and 'Dept' in new_data.columns:
            new_data = new_data[new_data['Dept'] == dept]

        # Convert the Pandas DataFrame to an H2O frame
        h2o_new_data = h2o.H2OFrame(new_data)

        # Exclude 'id', 'Dept', and 'date' from features used for prediction
        predictors = [col for col in h2o_new_data.columns if col not in ['id', 'Dept', 'date']]

        # Make predictions
        preds = self.model.predict(h2o_new_data[predictors])

        # Return predictions as a pandas DataFrame
        predictions_df = preds.as_data_frame()
        predictions_df.index = new_data.index  # Align indices

        # Add back 'id', 'Dept', and 'date' columns if present in the original data
        if 'id' in new_data.columns:
            predictions_df['id'] = new_data['id']
        if 'Dept' in new_data.columns:
            predictions_df['Dept'] = new_data['Dept']
        if 'date' in new_data.columns:
            predictions_df['date'] = new_data['date']

        return predictions_df

    ##
    def get_actual_model_name(self):
        # Assuming model is the H2OAutoML object
        if self.model is not None:
            return self.model.leader.model_id
        else:
            return "Model not trained"
    def predict_for_dept(self, new_data, dept):
        """ Predicts for a specific department """
        # Filter data for the specific department
        dept_data = new_data[new_data['Dept'] == dept]
        if dept_data.empty:
            print(f"No data available for Department: {dept}")
            return pd.DataFrame()

        # Convert the Pandas DataFrame to an H2O frame
        h2o_dept_data = h2o.H2OFrame(dept_data)

        # Exclude 'id' and 'date' from features used for prediction
        predictors = [col for col in h2o_dept_data.columns if col not in ['id', 'date', 'Dept']]

        # Make predictions
        preds = self.model.predict(h2o_dept_data[predictors])

        # Return predictions as a pandas DataFrame
        predictions_df = preds.as_data_frame()
        predictions_df.index = dept_data.index  # Align indices
        if 'id' in dept_data.columns:
            predictions_df['id'] = dept_data['id']
        if 'date' in dept_data.columns:
            predictions_df['date'] = dept_data['date']
        predictions_df['Dept'] = dept  # Add department information

        return predictions_df
    @property
    def description(self):
        return self.get_actual_model_name()

