
#from AutoGluonTabularWrapper import AutoGluonTabularWrapper
import pandas as pd
from autogluon.tabular import TabularPredictor



class AutoGluonTabularWrapper:
    _id_counter = 1

    def __init__(self, target_column, model_name="AutoGluonTabular"):
        self.target_column = target_column
        self.predictor = None
        self.model_id = AutoGluonTabularWrapper._id_counter
        self.model_name = model_name
        self.train_data = None
        #AutoGluonTabularWrapper._id_counter += 1

    def fit(self, train_data, time_limit=3600):
        """ Train the AutoGluon model and calibrate it """
        self.train_data = train_data
        try:
            self.predictor = TabularPredictor(label=self.target_column).fit(train_data, time_limit=time_limit)
            # Perform calibration after model fitting
            self.calibrate(train_data.drop(columns=[self.target_column]), train_data[self.target_column])
        except Exception as e:
            print(f"Error occurred during fitting AutoGluon model: {e}")
            self.predictor = None

    def calibrate(self, X, y):
        """ Generate predictions and calculate residuals for model calibration """
        # Check if the model is trained
        if self.predictor is None:
            print("AutoGluonTabular model is not trained. Skipping calibration.")
            return

        try:
            # Generate predictions
            predictions = self.predict(X)
        except Exception as e:
            print(f"Error during prediction in AutoGluonTabular model: {e}")
            return

        # Calculate residuals and store calibration data
        calibration_data = pd.DataFrame({'actual': y, 'predicted': predictions})
        calibration_data['residuals'] = calibration_data['actual'] - calibration_data['predicted']
        self.residual_std = calibration_data['residuals'].std()
        self.calibration_data = calibration_data

        # Store the actual model name
        best_model_name = self.get_best_model() if hasattr(self, 'get_best_model') else 'AutoGluonTabular'
        self.actual_model_name = best_model_name

        # Optionally, print confirmation
        print("Calibration completed for AutoGluonTabular model.")
    ##

    def predict_quantiles(self, test_data, quantiles=[0.05, 0.95]):
        """
        Predict specified quantiles for the test data.
        Tries to use AutoGluon's quantile prediction if available, otherwise, uses a normal distribution assumption.
        """
        if self.predictor is None:
            raise Exception("Model not trained. Call fit() before predicting.")

        # If AutoGluon supports direct quantile prediction
        if hasattr(self.predictor, 'predict_quantile'):
            quantile_predictions = {}
            for q in quantiles:
                quantile_predictions[q] = self.predictor.predict_quantile(test_data, quantile=q)
            quantile_df = pd.DataFrame(quantile_predictions)
            quantile_df.columns = [f'quantile_{int(q*100)}' for q in quantiles]
            return quantile_df

        # Fallback to original method using normal distribution assumption
        else:
            if self.residual_std is None:
                raise Exception("Model not calibrated. Call calibrate() before predicting.")
            mean_predictions = self.predict(test_data)
            lower_bound = mean_predictions - 1.96 * self.residual_std
            upper_bound = mean_predictions + 1.96 * self.residual_std
            return pd.DataFrame({
                'lower': lower_bound,
                'upper': upper_bound,
                'predicted': mean_predictions
            })

    ##
    def predict_for_dept(self, new_data, dept):
        """ Predicts for a specific department """
        # Filter data for the specific department
        dept_data = new_data[new_data['Dept'] == dept]
        if dept_data.empty:
            return pd.DataFrame()
        return self.predict(dept_data)
    def predict(self, new_data):
        """ Make predictions using the trained model """
        return self.predictor.predict(new_data)

    def predict_with_model(self, new_data, model_name):
        """Make predictions using a specific model"""
        return self.predictor.predict(new_data, model=model_name)

    def evaluate(self, test_data):
        """ Evaluate the model on test data """
        return self.predictor.evaluate(test_data)

    def get_model_summary(self):
        """ Get a summary of the trained model """
        return self.predictor.leaderboard()

    def get_best_model(self):
        """ Get the best performing model """
        return self.predictor.get_model_best()

    @property
    def description(self):
        return self.model_name  # Use model_name as the description



    def get_actual_model_name(self):
        # Assuming there is a way to get the actual model name from AutoGluon
        # This could be a specific method or an attribute in the AutoGluonTabularWrapper class
        # For example, if the actual model name is stored in an attribute named 'actual_model_name'
        return getattr(self, 'actual_model_name', 'AutoGluonTabular')

    def refit_full(self, new_data, target_column):
        # Initialize and train the predictor on the new data
        self.predictor = TabularPredictor(label=target_column).fit(new_data)

        # Refit models on the entire dataset
        refit_models = self.predictor.refit_full()

        # Set the best model to the refit version
        best_model_name = max(refit_models, key=lambda k: refit_models[k])
        self.predictor.set_model_best(model=best_model_name, save_trainer=True)


    def get_model_details(self):
        return {
            '.model_id': self.model_id,
            '.model': '<fit[+]>' if self.predictor is not None else '<unfit>',
            '.model_desc': self.model_name,
            '.type': 'AutoGluon Tabular Model'
        }

