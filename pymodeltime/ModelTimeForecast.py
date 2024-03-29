from prophet import Prophet
import pandas as pd
import h2o

from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg
from .ModelTimeTable import ModelTimeTable
from .AutoGluonTabularWrapper import AutoGluonTabularWrapper
from .MLForecastWrapper import MLForecastWrapper



class ModelTimeForecast:
    def __init__(self, model_container, actual_data, target_column, future_data=None, forecast_horizon=None,
                 new_data=None, conf_interval=0.95, conf_by_id=False, actual_data_cutoff=None,
                 conf_method="conformal_default", keep_data=False, arrange_index=True):
        # Initialize the ModelTimeForecast class with model container and data parameters
        self.models = model_container.models if isinstance(model_container, ModelTimeTable) else model_container
        self.actual_data = actual_data
        self.future_data = future_data
        self.target_column = target_column
        self.actual_data_cutoff = actual_data_cutoff
        self.forecast_horizon = forecast_horizon
        self.new_data = new_data
        self.conf_interval = conf_interval
        self.conf_by_id = conf_by_id
        self.conf_method = conf_method
        self.keep_data = keep_data
        self.arrange_index = arrange_index
        self.model_id_counter = 1


    def forecast(self):
        forecast_results = []

        # Check if 'Dept' column exists for grouping
        if 'Dept' in self.actual_data.columns:
            grouped_data = self.actual_data.groupby('Dept')
        else:
            # If no 'Dept' column, treat the entire dataset as one group
            grouped_data = [('All', self.actual_data)]

        for dept, group in grouped_data:
            dept_new_data = self.new_data[self.new_data['Dept'] == dept] if self.new_data is not None and 'Dept' in self.new_data.columns else self.new_data
            dept_future_data = self.future_data[self.future_data['Dept'] == dept] if self.future_data is not None and 'Dept' in self.future_data.columns else self.future_data

            for model in self.models:
                # Process new data predictions
                if dept_new_data is not None:
                    dept_forecast_results = self._predict_new_data(model, dept_new_data, dept)
                    forecast_results.extend(dept_forecast_results)

                # Handle future forecasts for ProphetReg
                if isinstance(model, ProphetReg):
                    if dept_future_data is not None:
                        dept_forecast_results = self._prophet_future_forecast(model, dept_future_data)
                        forecast_results.extend(dept_forecast_results)
                    elif self.forecast_horizon:
                        dept_future_data_prophet = self._generate_future_forecast_data(model)
                        dept_forecast_results = self._prophet_future_forecast(model, dept_future_data_prophet)
                        forecast_results.extend(dept_forecast_results)
                else:
                    # Handle future forecasts for other models
                    if dept_future_data is not None:
                        dept_forecast_results = self._predict_future_data(model, dept_future_data, dept)
                        forecast_results.extend(dept_forecast_results)
                    elif self.forecast_horizon:
                        dept_future_data = self._generate_future_forecast_data(model, dept)
                        dept_forecast_results = self._predict_future_data(model, dept_future_data, dept)
                        forecast_results.extend(dept_forecast_results)

            # Process actual data for each department or the entire dataset
            dept_actual_data_results = self._process_actual_data(group, dept)
            forecast_results.extend(dept_actual_data_results)


        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)
        if 'date' in forecast_df.columns:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        forecast_df.drop_duplicates(inplace=True)
        # Ensure 'date' column in forecast results is consistent
        # if 'date' in forecast_df.columns and not pd.api.types.is_datetime64_any_dtype(forecast_df['date']):
        #      forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date


        return forecast_df



    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        model_desc = self._get_model_type(model)
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment for the next mode
        #print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        #print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': model_desc,
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data, dept):
        #print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Get the actual model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        results = []




        if isinstance(model, AutoGluonTabularWrapper):
            if future_data is None:
                print("Error: future_data is None")
                return []

            X_new = future_data.drop(columns=[self.target_column], errors='ignore')

            if model is None:
                print("Error: model is None")
                return []

            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})
                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = future_data['date'].values  # Change to future_data
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                #model_type = self._get_model_type(model)

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                         'Dept': dept,  # Include Dept in the result
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        ##
        elif isinstance(model, H2OAutoMLWrapper):
            # Initialize H2O
            h2o.init()

            # Handle future forecasts for H2O AutoML
            if future_data is not None:
                # Filter data for the specific department if 'Dept' column exists
                if 'Dept' in future_data.columns and dept is not None:
                    future_data_for_prediction = future_data[future_data['Dept'] == dept]
                else:
                    future_data_for_prediction = future_data

                # Convert to H2OFrame for prediction
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction.drop(columns=[self.target_column], errors='ignore'))

                # Make predictions
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Add back date and department info
                predictions['date'] = future_data_for_prediction['date'].values
                predictions['Dept'] = dept if 'Dept' in future_data.columns else 'All'

                # Add confidence intervals (or adjust based on H2O model output)
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct forecast results
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': self.model_id_counter,
                        'model_desc': model.model.model_id if model.model is not None else 'H2O AutoML',
                        'key': 'future',
                        'date': row['date'],
                        'Dept': row['Dept'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

                # Increment model_id_counter for the next model
                self.model_id_counter += 1


        elif isinstance(model, ArimaReg):
            # Ensure the future_data DataFrame is properly formatted for ARIMA
            if 'date' not in future_data.columns:
                raise KeyError("Expected 'date' column in future_data for ARIMA")

            # Perform prediction
            predictions = model.predict(future_data)

            # Convert Series to DataFrame if necessary
            if isinstance(predictions, pd.Series):
                predictions = predictions.to_frame(name='predicted')

            # Ensure the 'date' column from future_data is included in predictions
            predictions = predictions.set_index(pd.to_datetime(future_data['date']).dt.date)
            predictions.reset_index(inplace=True)

            # Construct the result list with confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],  # 'index' contains the date
                     'Dept': dept,  # Include Dept in the result
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)


        ##
        elif isinstance(model, MLModelWrapper):
            # Ensure that feature_names are in the future_data
            if not all(name in future_data.columns for name in model.feature_names):
                raise ValueError("Missing required feature columns in future_data for MLModelWrapper")

            X = future_data[model.feature_names]
            predictions = model.predict(X)

            # Convert predictions to DataFrame if necessary
            if not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions, columns=['predicted'])

            # Ensure the 'date' column from future_data is included in predictions
            predictions = predictions.set_index(pd.to_datetime(future_data['date']).dt.date)
            predictions.reset_index(inplace=True)
             # Ensure 'date' is of type Timestamp
            # predictions['date'] = pd.to_datetime(future_data['date'])



            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            #model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': getattr(model, 'model_name', 'ML Model'),
                    'key': 'future',
                    'date': row['date'],
                     'Dept': dept,  # Include Dept in the result
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

      
        return results



    #
    def _filter_dept_data(self, data, dept):
        """Helper method to filter data by department if 'Dept' column exists."""
        return data[data['Dept'] == dept] if data is not None and 'Dept' in data.columns else data

    def _process_actual_data(self, group, dept):
        """
        Process actual data for forecasting, filtering out any future dates.
        This method now takes additional arguments 'group' and 'dept'.
        """
        if not hasattr(self, 'target_column') or self.target_column not in group.columns:
            raise ValueError("Target column not set or not found in actual data.")

        filtered_actual_data = group.dropna(subset=[self.target_column])

        latest_actual_date = filtered_actual_data['date'].max()
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        # Include 'Dept' in results if it exists in the group DataFrame
        include_dept = 'Dept' in group.columns

        return [{
            'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
            'date': row['date'], 'value': row[self.target_column],
            'conf_lo': None, 'conf_hi': None,
            'Dept': row['Dept'] if include_dept else dept
        } for _, row in filtered_actual_data.iterrows()] or []





    def _process_forecast_data(self, forecast_results):
        """
        Process new data and future forecasts.
        """
        for model in self.models:
            self._validate_model_predict_method(model)

            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            if self.forecast_horizon:
                forecast_results.extend(self._predict_future_data(model))

    def _validate_model_predict_method(self, model):
        """
        Validate if the model has a 'predict' method.
        """
        if not hasattr(model, 'predict'):
            raise AttributeError(f"The model {model} does not have a 'predict' method.")



    ##
    def _generate_forecast_data(self, model, data):
        """
        Generate forecast data for the model.
        """
        if isinstance(model, ProphetReg):
            #print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            #print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model, new_data, dept):
        #print(f"Processing predictions for model: {type(model).__name__} and department: {dept}")
        forecast_data = self._generate_forecast_data(model, new_data)
        # model_desc = getattr(model, 'description', 'No description available')
        model_desc = self._get_model_type(model)

        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        results = []


        ##
        if isinstance(model, AutoGluonTabularWrapper):
            # Filter new data for the specific department if Dept column exists and department is specified
            if 'Dept' in new_data.columns and dept is not None:
                X_new = new_data[new_data['Dept'] == dept].drop(columns=[self.target_column], errors='ignore')
            else:
                X_new = new_data.drop(columns=[self.target_column], errors='ignore')

            if not X_new.empty:
                try:
                    predictions_raw = model.predict(X_new)
                    quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})
                    predictions_df = predictions_raw.to_frame(name='predicted')
                    predictions_df['date'] = X_new['date'].values
                    predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                    for _, row in predictions_df.iterrows():
                        predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                        result = {
                            'model_id': model_id, 'model_desc': model_desc, 'key': 'prediction',
                            'date': row['date'], 'Dept': dept if 'Dept' in new_data.columns else 'All',
                            'value': predicted_value, 'conf_lo': row.get('lower', None), 'conf_hi': row.get('upper', None)
                        }
                        results.append(result)

                except Exception as e:
                    print(f"Error in predicting with AutoGluonTabularWrapper: {e}")

        ##
        elif isinstance(model, MLForecastWrapper):
            # Using model's own id for MLForecastWrapper sub-models
            model_id = getattr(model, 'id', None) or self.model_id_counter
            model_desc = self._get_model_type(model)

            # Ensure 'ds' column is present and properly formatted
            if 'date' in new_data.columns:
                new_data = new_data.rename(columns={'date': 'ds'})
            new_data['ds'] = pd.to_datetime(new_data['ds']).dt.date

            # Generate predictions
            predictions = model.predict(len(new_data), levels=[0.95])

            # Iterate over each sub-model in MLForecastWrapper and append results

            results = []
            for sub_model_name in model.models.keys():
                sub_model_predictions = predictions[[sub_model_name, f'{sub_model_name}-lo-0.95', f'{sub_model_name}-hi-0.95']]

                # Assign a unique model_id for each sub-model
                sub_model_id = f"{getattr(model, 'id', None) or self.model_id_counter}_{sub_model_name}"
                self.model_id_counter += 1  # Increment counter for next sub-model

                for idx, row in sub_model_predictions.iterrows():
                    result = {
                        'model_id': sub_model_id,
                        'model_desc': f"{sub_model_name}",
                        'key': 'prediction',
                        'date': new_data.iloc[idx]['ds'],
                        'Dept': dept,
                        'value': row[sub_model_name],
                        'conf_lo': row[f'{sub_model_name}-lo-0.95'],
                        'conf_hi': row[f'{sub_model_name}-hi-0.95']
                    }
                    results.append(result)

            return results



        elif isinstance(model, ArimaReg):
            # Handling ARIMA model
            predictions = model.predict(forecast_data)
            if isinstance(predictions, pd.Series):
                predictions = predictions.to_frame(name='predicted')
            predictions['date'] = forecast_data['date'].values  # Aligning dates

            # Calculating confidence intervals
            error_margin = predictions['predicted'] * 0.05  # 5% error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Process predictions to create results
            for _, row in predictions.iterrows():
                results.append({
                    'model_id': model.id,
                    'model_desc': model_desc,
                    'key': 'prediction',
                    'date': row['date'],
                    'Dept': dept,  # Include Dept in the result
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })

        ##
        elif isinstance(model, ProphetReg):

            # Handling ProphetReg model
            forecast_data_prophet = forecast_data.rename(columns={'date': 'ds'})
            predictions = model.predict(forecast_data_prophet)
            predictions['date'] = forecast_data_prophet['ds'].values  # Aligning dates

            # Process predictions to create results
            for _, row in predictions.iterrows():
                results.append({
                    'model_id': model.id,
                    'model_desc': model_desc,
                    'key': 'prediction',
                    'date': row['date'],
                    'Dept': dept,  # Include Dept in the result
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        ##
        elif isinstance(model, H2OAutoMLWrapper):
            # Initialize H2O
            h2o.init()

            # Handle predictions for grouped data (by Dept) if 'Dept' column exists
            if 'Dept' in forecast_data.columns:
                unique_depts = forecast_data['Dept'].unique()
                for dept in unique_depts:
                    # Filter data for the current department
                    dept_forecast_data = forecast_data[forecast_data['Dept'] == dept]

                    # Predict for the specific department
                    h2o_predictions = model.model.predict(h2o.H2OFrame(dept_forecast_data))
                    predictions = h2o_predictions.as_data_frame()
                    predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                    predictions['date'] = dept_forecast_data['date'].values  # Align dates

                    # Add confidence intervals
                    error_margin = predictions['predicted'] * 0.05  # Example error margin
                    predictions['conf_lo'] = predictions['predicted'] - error_margin
                    predictions['conf_hi'] = predictions['predicted'] + error_margin

                    # Process predictions to create results
                    for _, row in predictions.iterrows():
                        results.append({
                            'model_id': model_id,
                            'model_desc': model_desc,
                            'key': 'prediction',
                            'date': row['date'],
                            'Dept': dept,  # Specify the department
                            'value': row['predicted'],
                            'conf_lo': row['conf_lo'],
                            'conf_hi': row['conf_hi']
                        })

            else:
                # Handle predictions for non-grouped data
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Align dates

                # Add confidence intervals
                error_margin = predictions['predicted'] * 0.05
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'Dept': 'All',  # Use 'All' for non-grouped data
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })





        ##
        elif isinstance(model, MLModelWrapper):
           # print(f"Processing predictions for model: {type(model).__name__} and department: {dept}")

            # Check if forecast_data is empty
            if forecast_data.empty:
                print(f"No data available for predictions for department: {dept}")
                return []

            # Check if all feature names are present
            missing_features = [feature for feature in model.feature_names if feature not in forecast_data.columns]
            if missing_features:
                print(f"Missing features in forecast_data: {missing_features}")
                return []

            X = forecast_data[model.feature_names]
            predictions = model.predict(X)

            # Check if predictions are empty
            if predictions.size == 0:
                print(f"No predictions returned for model {type(model).__name__} and department {dept}")
                return []

            # Convert predictions to DataFrame if necessary
            if not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions, columns=['predicted'])

            # Standardizing date format
            predictions['date'] = pd.to_datetime(forecast_data['date'].values)

            # Calculating confidence intervals
            error_margin = predictions['predicted'] * 0.05  # 5% error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Process predictions to create results
            for _, row in predictions.iterrows():
                results.append({
                    'model_id': model_id,
                    'model_desc': getattr(model, 'model_name', 'ML Model'),
                    'key': 'prediction',
                    'date': row['date'],
                    'Dept': dept,  # Include Dept in the result
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })


        # Add handling for other model types if necessary...

        return results





    ##
    def _create_future_dataframe(self, periods):
        """
        Create a future DataFrame with the specified number of periods.
        """
        # Ensure the 'date' column is a datetime object
        self.actual_data['date'] = pd.to_datetime(self.actual_data['date'])

        # Get the last date in the data
        last_date_in_data = self.actual_data['date'].max()

        # Calculate the start date for the future date range
        start_date = last_date_in_data + pd.Timedelta(days=1)

        # Create the future date range
        future_dates = pd.date_range(start=start_date, periods=periods, freq='M')
        return pd.DataFrame({'date': future_dates})



    ##
    def _parse_forecast_horizon(self, horizon_str):
          """
          Parse the forecast horizon string to determine the number of periods for forecasting.
          Supported units are 'day', 'week', 'month', 'quarter', and 'year'.
          """
          number, unit = horizon_str.split()
          number = int(number)
          freq = pd.infer_freq(self.actual_data['date'])

          # Default to daily frequency if unable to infer
          if freq is None:
              print("Unable to infer frequency of the date column. Defaulting to daily frequency.")
              freq = 'D'

          # Convert the forecast horizon to the appropriate number of periods
          if unit in ['day', 'days']:
              periods = number
          elif unit in ['week', 'weeks']:
              periods = number * 7  # 7 days in a week
          elif unit in ['month', 'months']:
              periods = number  # Assuming frequency inferred is monthly
          elif unit in ['quarter', 'quarters']:
              periods = number * 3  # 3 months in a quarter
          elif unit in ['year', 'years']:
              periods = number * 12  # 12 months in a year
          else:
              raise ValueError(f"Unsupported time unit in forecast horizon: {unit}")

          return periods


   ##
    def _generate_future_forecast_data(self, model, dept=None):
            """
            Generate future forecast data for the model.
            Adjusted to handle both ungrouped and grouped data.
            """
            if not self.forecast_horizon:
                return None

            periods = self._parse_forecast_horizon(self.forecast_horizon)
            future_data = self._create_future_dataframe(periods)

            # Adjustments for Prophet and ARIMA models
            if isinstance(model, ProphetReg):
                future_data = future_data.rename(columns={'date': 'ds'})
            elif isinstance(model, ArimaReg):
                # Additional handling for ARIMA if needed
                pass

            # Include 'Dept' column if the data is grouped
            if dept is not None:
                future_data['Dept'] = dept

            return future_data


    ##
    def _get_model_type(self, model):
        """ Utility function to get the type of the model. """
        if isinstance(model, AutoGluonTabularWrapper):
            return 'AutoGluonTabular'
        elif isinstance(model, ProphetReg):
            return 'Prophet'
        elif isinstance(model, ArimaReg):
            return 'ARIMA'
        elif isinstance(model, H2OAutoMLWrapper):
            return 'H2O AutoML' if model.model is not None else 'H2O AutoML (Untrained)'
        elif isinstance(model, MLModelWrapper):
            return model.model.__class__.__name__  # Get the class name of the underlying model
        elif isinstance(model, MLForecastWrapper):
            return 'MLForecast Wrapper'  # New case for MLForecastWrapper
        else:
            return 'Unknown Model'

    def _calculate_confidence_interval(self, model, predicted_value):
        """
        Calculate confidence interval for a prediction.
        Handles different input types for ARIMA and Prophet models.
        """
        if isinstance(model, ProphetReg):
            raise ValueError("Prophet model should already include confidence intervals.")

        if isinstance(model, ArimaReg):
            # ARIMA model: Use predicted value to calculate confidence interval
            interval_width = 0.1 * abs(predicted_value)  # Adjust interval width as needed
            conf_lo = predicted_value - interval_width
            conf_hi = predicted_value + interval_width
        else:
            conf_lo = conf_hi = None

        return {'conf_lo': conf_lo, 'conf_hi': conf_hi}



         # Helper function to check if a value is numeric
    def is_numeric(value):
            return isinstance(value, (int, float))

