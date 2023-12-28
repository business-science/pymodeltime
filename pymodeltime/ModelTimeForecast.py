from prophet import Prophet
import pandas as pd
from .MLModelWrapper import MLModelWrapper
from .H2OAutoMLWrapper import H2OAutoMLWrapper
from .ArimaReg import ArimaReg
from .ProphetReg import ProphetReg
from .ModelTimeTable import ModelTimeTable
from .AutoGluonTabularWrapper import AutoGluonTabularWrapper



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
        

           ##
    ##
    def forecast(self):
        forecast_results = []

        # Existing code for processing new data predictions
        for model in self.models:
            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            # Handling future forecasts with special treatment for Prophet
            if isinstance(model, ProphetReg):
                if self.future_data is not None:
                    future_data_prophet = self.future_data.rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
                elif self.forecast_horizon:
                    future_data_prophet = self._generate_future_forecast_data(model).rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
            else:
                if self.future_data is not None:
                    future_forecast_data = self.future_data
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))
                elif self.forecast_horizon:
                    future_forecast_data = self._generate_future_forecast_data(model)
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))

        # Process actual data
        if self.actual_data is not None:
            actual_data_results = self._process_actual_data()
            forecast_results.extend(actual_data_results)

        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)

        # Ensure 'date' column in forecast results is consistent
        if 'date' in forecast_df.columns and forecast_df['date'].dtype != 'datetime.date':
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date  # Convert to datetime.date

        # Sort and remove duplicates
        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        #print("Final Forecast DataFrame:", forecast_df.tail(30))
        # Diagnostics: Inspect forecast_df before dropping duplicates
#         print("Inspecting forecast_df before drop_duplicates:")
#         print(forecast_df.dtypes)  # Print the data types of each column
#         print(forecast_df.head())  # Print the first few rows of the DataFrame

#         # Optional: Check each column for unhashable types
#         for column in forecast_df.columns:
#             if any(isinstance(x, pd.Series) for x in forecast_df[column]):
#                 print(f"Column {column} contains unhashable types.")

        forecast_df.drop_duplicates(inplace=True)

        return forecast_df






    ##
    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': 'PROPHET',
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data):
        print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Use _get_model_type to get the actual model type
        #model_id = getattr(model, 'id', 'N/A')  # Get the model's existing ID
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

                # Prepare a copy of the future_data DataFrame without the target column for prediction
                if model.target_column in future_data.columns:
                    future_data_for_prediction = future_data.drop(columns=[model.target_column], errors='ignore')
                else:
                    future_data_for_prediction = future_data

                # Convert the prepared DataFrame to H2OFrame
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction)

                # Make predictions using the H2O AutoML model
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Ensure the index of predictions matches future_data's index
                predictions.index = future_data.index

                # Add 'date' from future_data to predictions
                predictions['date'] = future_data['date']

                # Add confidence intervals
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct results without 'id'
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

    
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

            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

        # ... [handling for other models, if needed] ...

        return results
        
    
    
    def _process_actual_data(self):
        """
        Process actual data for forecasting, filtering out any future dates.
        """
        # Ensure target_column is set
        if not hasattr(self, 'target_column') or self.target_column not in self.actual_data.columns:
            raise ValueError("Target column not set or not found in actual data.")

        # Filter out rows where the target value is NaN (or apply other criteria)
        filtered_actual_data = self.actual_data.dropna(subset=[self.target_column])

        # Find the latest date in the filtered data
        latest_actual_date = filtered_actual_data['date'].max()

        # Additional filtering if needed
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        #print("Filtered actual data (tail):", filtered_actual_data.tail())

        # Ensure a list is returned even if filtered_actual_data is empty
        return [{'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
                'date': row['date'], 'value': row[self.target_column], 'conf_lo': None, 'conf_hi': None}
                for _, row in filtered_actual_data.iterrows()] or []




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
            print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model):
        print(f"Processing predictions for model: {type(model).__name__}")
        forecast_data = self._generate_forecast_data(model, self.new_data)
        model_desc = getattr(model, 'description', 'No description available')

         # Generate a unique model_id for each model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        # Initialize an empty list for results
        results = []

        ##
        if isinstance(model, AutoGluonTabularWrapper):
            X_new = self.new_data.drop(columns=[self.target_column], errors='ignore')
            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})

                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = self.new_data['date'].values
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                model_type = self._get_model_type(model)  # Use _get_model_type to get the actual model name

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_type,  # Use model_type for the model description
                        'key': 'prediction',
                        'date': row['date'],
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        


       

        
        ##
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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        elif isinstance(model, H2OAutoMLWrapper):
                # Handling H2OAutoMLWrapper model
                h2o.init()
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Ensure date is aligned

                # Using a similar approach for confidence interval estimation
                error_margin = predictions['predicted'] * 0.05  # 5% error margin as an example
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                model_id = getattr(model, 'id', 'H2O_AutoML')  # Use a default id if not available

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })
        elif isinstance(model, MLModelWrapper):
                # Handling MLModelWrapper models
                X = forecast_data[model.feature_names]
                predictions = model.predict(X)

                # Convert predictions to DataFrame if necessary
                if not isinstance(predictions, pd.DataFrame):
                    predictions = pd.DataFrame(predictions, columns=['predicted'])

                # Aligning dates
                predictions['date'] = forecast_data['date'].values

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
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })

        # Add handling for other model types if necessary...

        return results
    
  

   
    ##
    def _create_future_dataframe(self, periods):
        """
        Create a DataFrame for future dates.
        """
        last_date_in_data = self.actual_data['date'].max()
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods, freq='M')
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
    def _generate_future_forecast_data(self, model):
        """
        Generate future forecast data for the model.
        """
        if not self.forecast_horizon:
            return None

        periods = self._parse_forecast_horizon(self.forecast_horizon)
        future_data = self._create_future_dataframe(periods)

        # For Prophet model, ensure the 'ds' column is present
        if isinstance(model, ProphetReg):
            future_data = future_data.rename(columns={'date': 'ds'})
        return future_data

        ##
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
from prophet import Prophet
from MLModelWrapper import MLModelWrapper
from H2OAutoMLWrapper import H2OAutoMLWrapper
from ArimaReg import ArimaReg
from ProphetReg import ProphetReg
from ModelTimeTable import ModelTimeTable
from AutoGluonTabularWrapper import AutoGluonTabularWrapper

import pandas as pd

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
        

           ##
    ##
    def forecast(self):
        forecast_results = []

        # Existing code for processing new data predictions
        for model in self.models:
            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            # Handling future forecasts with special treatment for Prophet
            if isinstance(model, ProphetReg):
                if self.future_data is not None:
                    future_data_prophet = self.future_data.rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
                elif self.forecast_horizon:
                    future_data_prophet = self._generate_future_forecast_data(model).rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
            else:
                if self.future_data is not None:
                    future_forecast_data = self.future_data
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))
                elif self.forecast_horizon:
                    future_forecast_data = self._generate_future_forecast_data(model)
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))

        # Process actual data
        if self.actual_data is not None:
            actual_data_results = self._process_actual_data()
            forecast_results.extend(actual_data_results)

        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)

        # Ensure 'date' column in forecast results is consistent
        if 'date' in forecast_df.columns and forecast_df['date'].dtype != 'datetime.date':
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date  # Convert to datetime.date

        # Sort and remove duplicates
        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        #print("Final Forecast DataFrame:", forecast_df.tail(30))
        # Diagnostics: Inspect forecast_df before dropping duplicates
#         print("Inspecting forecast_df before drop_duplicates:")
#         print(forecast_df.dtypes)  # Print the data types of each column
#         print(forecast_df.head())  # Print the first few rows of the DataFrame

#         # Optional: Check each column for unhashable types
#         for column in forecast_df.columns:
#             if any(isinstance(x, pd.Series) for x in forecast_df[column]):
#                 print(f"Column {column} contains unhashable types.")

        forecast_df.drop_duplicates(inplace=True)

        return forecast_df






    ##
    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': 'PROPHET',
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data):
        print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Use _get_model_type to get the actual model type
        #model_id = getattr(model, 'id', 'N/A')  # Get the model's existing ID
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

                # Prepare a copy of the future_data DataFrame without the target column for prediction
                if model.target_column in future_data.columns:
                    future_data_for_prediction = future_data.drop(columns=[model.target_column], errors='ignore')
                else:
                    future_data_for_prediction = future_data

                # Convert the prepared DataFrame to H2OFrame
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction)

                # Make predictions using the H2O AutoML model
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Ensure the index of predictions matches future_data's index
                predictions.index = future_data.index

                # Add 'date' from future_data to predictions
                predictions['date'] = future_data['date']

                # Add confidence intervals
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct results without 'id'
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

    
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

            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

        # ... [handling for other models, if needed] ...

        return results
        
    
    
    def _process_actual_data(self):
        """
        Process actual data for forecasting, filtering out any future dates.
        """
        # Ensure target_column is set
        if not hasattr(self, 'target_column') or self.target_column not in self.actual_data.columns:
            raise ValueError("Target column not set or not found in actual data.")

        # Filter out rows where the target value is NaN (or apply other criteria)
        filtered_actual_data = self.actual_data.dropna(subset=[self.target_column])

        # Find the latest date in the filtered data
        latest_actual_date = filtered_actual_data['date'].max()

        # Additional filtering if needed
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        #print("Filtered actual data (tail):", filtered_actual_data.tail())

        # Ensure a list is returned even if filtered_actual_data is empty
        return [{'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
                'date': row['date'], 'value': row[self.target_column], 'conf_lo': None, 'conf_hi': None}
                for _, row in filtered_actual_data.iterrows()] or []




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
            print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model):
        print(f"Processing predictions for model: {type(model).__name__}")
        forecast_data = self._generate_forecast_data(model, self.new_data)
        model_desc = getattr(model, 'description', 'No description available')

         # Generate a unique model_id for each model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        # Initialize an empty list for results
        results = []

        ##
        if isinstance(model, AutoGluonTabularWrapper):
            X_new = self.new_data.drop(columns=[self.target_column], errors='ignore')
            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})

                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = self.new_data['date'].values
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                model_type = self._get_model_type(model)  # Use _get_model_type to get the actual model name

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_type,  # Use model_type for the model description
                        'key': 'prediction',
                        'date': row['date'],
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        


       

        
        ##
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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        elif isinstance(model, H2OAutoMLWrapper):
                # Handling H2OAutoMLWrapper model
                h2o.init()
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Ensure date is aligned

                # Using a similar approach for confidence interval estimation
                error_margin = predictions['predicted'] * 0.05  # 5% error margin as an example
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                model_id = getattr(model, 'id', 'H2O_AutoML')  # Use a default id if not available

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })
        elif isinstance(model, MLModelWrapper):
                # Handling MLModelWrapper models
                X = forecast_data[model.feature_names]
                predictions = model.predict(X)

                # Convert predictions to DataFrame if necessary
                if not isinstance(predictions, pd.DataFrame):
                    predictions = pd.DataFrame(predictions, columns=['predicted'])

                # Aligning dates
                predictions['date'] = forecast_data['date'].values

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
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })

        # Add handling for other model types if necessary...

        return results
    
  

   
    ##
    def _create_future_dataframe(self, periods):
        """
        Create a DataFrame for future dates.
        """
        last_date_in_data = self.actual_data['date'].max()
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods, freq='M')
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
    def _generate_future_forecast_data(self, model):
        """
        Generate future forecast data for the model.
        """
        if not self.forecast_horizon:
            return None

        periods = self._parse_forecast_horizon(self.forecast_horizon)
        future_data = self._create_future_dataframe(periods)

        # For Prophet model, ensure the 'ds' column is present
        if isinstance(model, ProphetReg):
            future_data = future_data.rename(columns={'date': 'ds'})
        return future_data

        ##
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
from prophet import Prophet
from MLModelWrapper import MLModelWrapper
from H2OAutoMLWrapper import H2OAutoMLWrapper
from ArimaReg import ArimaReg
from ProphetReg import ProphetReg
from ModelTimeTable import ModelTimeTable
from AutoGluonTabularWrapper import AutoGluonTabularWrapper

import pandas as pd

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
        

           ##
    ##
    def forecast(self):
        forecast_results = []

        # Existing code for processing new data predictions
        for model in self.models:
            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            # Handling future forecasts with special treatment for Prophet
            if isinstance(model, ProphetReg):
                if self.future_data is not None:
                    future_data_prophet = self.future_data.rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
                elif self.forecast_horizon:
                    future_data_prophet = self._generate_future_forecast_data(model).rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
            else:
                if self.future_data is not None:
                    future_forecast_data = self.future_data
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))
                elif self.forecast_horizon:
                    future_forecast_data = self._generate_future_forecast_data(model)
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))

        # Process actual data
        if self.actual_data is not None:
            actual_data_results = self._process_actual_data()
            forecast_results.extend(actual_data_results)

        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)

        # Ensure 'date' column in forecast results is consistent
        if 'date' in forecast_df.columns and forecast_df['date'].dtype != 'datetime.date':
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date  # Convert to datetime.date

        # Sort and remove duplicates
        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        #print("Final Forecast DataFrame:", forecast_df.tail(30))
        # Diagnostics: Inspect forecast_df before dropping duplicates
#         print("Inspecting forecast_df before drop_duplicates:")
#         print(forecast_df.dtypes)  # Print the data types of each column
#         print(forecast_df.head())  # Print the first few rows of the DataFrame

#         # Optional: Check each column for unhashable types
#         for column in forecast_df.columns:
#             if any(isinstance(x, pd.Series) for x in forecast_df[column]):
#                 print(f"Column {column} contains unhashable types.")

        forecast_df.drop_duplicates(inplace=True)

        return forecast_df






    ##
    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': 'PROPHET',
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data):
        print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Use _get_model_type to get the actual model type
        #model_id = getattr(model, 'id', 'N/A')  # Get the model's existing ID
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

                # Prepare a copy of the future_data DataFrame without the target column for prediction
                if model.target_column in future_data.columns:
                    future_data_for_prediction = future_data.drop(columns=[model.target_column], errors='ignore')
                else:
                    future_data_for_prediction = future_data

                # Convert the prepared DataFrame to H2OFrame
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction)

                # Make predictions using the H2O AutoML model
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Ensure the index of predictions matches future_data's index
                predictions.index = future_data.index

                # Add 'date' from future_data to predictions
                predictions['date'] = future_data['date']

                # Add confidence intervals
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct results without 'id'
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

    
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

            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

        # ... [handling for other models, if needed] ...

        return results
        
    
    
    def _process_actual_data(self):
        """
        Process actual data for forecasting, filtering out any future dates.
        """
        # Ensure target_column is set
        if not hasattr(self, 'target_column') or self.target_column not in self.actual_data.columns:
            raise ValueError("Target column not set or not found in actual data.")

        # Filter out rows where the target value is NaN (or apply other criteria)
        filtered_actual_data = self.actual_data.dropna(subset=[self.target_column])

        # Find the latest date in the filtered data
        latest_actual_date = filtered_actual_data['date'].max()

        # Additional filtering if needed
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        #print("Filtered actual data (tail):", filtered_actual_data.tail())

        # Ensure a list is returned even if filtered_actual_data is empty
        return [{'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
                'date': row['date'], 'value': row[self.target_column], 'conf_lo': None, 'conf_hi': None}
                for _, row in filtered_actual_data.iterrows()] or []




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
            print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model):
        print(f"Processing predictions for model: {type(model).__name__}")
        forecast_data = self._generate_forecast_data(model, self.new_data)
        model_desc = getattr(model, 'description', 'No description available')

         # Generate a unique model_id for each model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        # Initialize an empty list for results
        results = []

        ##
        if isinstance(model, AutoGluonTabularWrapper):
            X_new = self.new_data.drop(columns=[self.target_column], errors='ignore')
            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})

                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = self.new_data['date'].values
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                model_type = self._get_model_type(model)  # Use _get_model_type to get the actual model name

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_type,  # Use model_type for the model description
                        'key': 'prediction',
                        'date': row['date'],
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        


       

        
        ##
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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        elif isinstance(model, H2OAutoMLWrapper):
                # Handling H2OAutoMLWrapper model
                h2o.init()
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Ensure date is aligned

                # Using a similar approach for confidence interval estimation
                error_margin = predictions['predicted'] * 0.05  # 5% error margin as an example
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                model_id = getattr(model, 'id', 'H2O_AutoML')  # Use a default id if not available

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })
        elif isinstance(model, MLModelWrapper):
                # Handling MLModelWrapper models
                X = forecast_data[model.feature_names]
                predictions = model.predict(X)

                # Convert predictions to DataFrame if necessary
                if not isinstance(predictions, pd.DataFrame):
                    predictions = pd.DataFrame(predictions, columns=['predicted'])

                # Aligning dates
                predictions['date'] = forecast_data['date'].values

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
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })

        # Add handling for other model types if necessary...

        return results
    
  

   
    ##
    def _create_future_dataframe(self, periods):
        """
        Create a DataFrame for future dates.
        """
        last_date_in_data = self.actual_data['date'].max()
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods, freq='M')
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
    def _generate_future_forecast_data(self, model):
        """
        Generate future forecast data for the model.
        """
        if not self.forecast_horizon:
            return None

        periods = self._parse_forecast_horizon(self.forecast_horizon)
        future_data = self._create_future_dataframe(periods)

        # For Prophet model, ensure the 'ds' column is present
        if isinstance(model, ProphetReg):
            future_data = future_data.rename(columns={'date': 'ds'})
        return future_data

        ##
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
from prophet import Prophet
from MLModelWrapper import MLModelWrapper
from H2OAutoMLWrapper import H2OAutoMLWrapper
from ArimaReg import ArimaReg
from ProphetReg import ProphetReg
from ModelTimeTable import ModelTimeTable
from AutoGluonTabularWrapper import AutoGluonTabularWrapper

import pandas as pd

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
        

           ##
    ##
    def forecast(self):
        forecast_results = []

        # Existing code for processing new data predictions
        for model in self.models:
            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            # Handling future forecasts with special treatment for Prophet
            if isinstance(model, ProphetReg):
                if self.future_data is not None:
                    future_data_prophet = self.future_data.rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
                elif self.forecast_horizon:
                    future_data_prophet = self._generate_future_forecast_data(model).rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
            else:
                if self.future_data is not None:
                    future_forecast_data = self.future_data
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))
                elif self.forecast_horizon:
                    future_forecast_data = self._generate_future_forecast_data(model)
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))

        # Process actual data
        if self.actual_data is not None:
            actual_data_results = self._process_actual_data()
            forecast_results.extend(actual_data_results)

        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)

        # Ensure 'date' column in forecast results is consistent
        if 'date' in forecast_df.columns and forecast_df['date'].dtype != 'datetime.date':
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date  # Convert to datetime.date

        # Sort and remove duplicates
        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        #print("Final Forecast DataFrame:", forecast_df.tail(30))
        # Diagnostics: Inspect forecast_df before dropping duplicates
#         print("Inspecting forecast_df before drop_duplicates:")
#         print(forecast_df.dtypes)  # Print the data types of each column
#         print(forecast_df.head())  # Print the first few rows of the DataFrame

#         # Optional: Check each column for unhashable types
#         for column in forecast_df.columns:
#             if any(isinstance(x, pd.Series) for x in forecast_df[column]):
#                 print(f"Column {column} contains unhashable types.")

        forecast_df.drop_duplicates(inplace=True)

        return forecast_df






    ##
    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': 'PROPHET',
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data):
        print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Use _get_model_type to get the actual model type
        #model_id = getattr(model, 'id', 'N/A')  # Get the model's existing ID
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

                # Prepare a copy of the future_data DataFrame without the target column for prediction
                if model.target_column in future_data.columns:
                    future_data_for_prediction = future_data.drop(columns=[model.target_column], errors='ignore')
                else:
                    future_data_for_prediction = future_data

                # Convert the prepared DataFrame to H2OFrame
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction)

                # Make predictions using the H2O AutoML model
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Ensure the index of predictions matches future_data's index
                predictions.index = future_data.index

                # Add 'date' from future_data to predictions
                predictions['date'] = future_data['date']

                # Add confidence intervals
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct results without 'id'
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

    
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

            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

        # ... [handling for other models, if needed] ...

        return results
        
    
    
    def _process_actual_data(self):
        """
        Process actual data for forecasting, filtering out any future dates.
        """
        # Ensure target_column is set
        if not hasattr(self, 'target_column') or self.target_column not in self.actual_data.columns:
            raise ValueError("Target column not set or not found in actual data.")

        # Filter out rows where the target value is NaN (or apply other criteria)
        filtered_actual_data = self.actual_data.dropna(subset=[self.target_column])

        # Find the latest date in the filtered data
        latest_actual_date = filtered_actual_data['date'].max()

        # Additional filtering if needed
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        #print("Filtered actual data (tail):", filtered_actual_data.tail())

        # Ensure a list is returned even if filtered_actual_data is empty
        return [{'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
                'date': row['date'], 'value': row[self.target_column], 'conf_lo': None, 'conf_hi': None}
                for _, row in filtered_actual_data.iterrows()] or []




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
            print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model):
        print(f"Processing predictions for model: {type(model).__name__}")
        forecast_data = self._generate_forecast_data(model, self.new_data)
        model_desc = getattr(model, 'description', 'No description available')

         # Generate a unique model_id for each model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        # Initialize an empty list for results
        results = []

        ##
        if isinstance(model, AutoGluonTabularWrapper):
            X_new = self.new_data.drop(columns=[self.target_column], errors='ignore')
            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})

                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = self.new_data['date'].values
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                model_type = self._get_model_type(model)  # Use _get_model_type to get the actual model name

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_type,  # Use model_type for the model description
                        'key': 'prediction',
                        'date': row['date'],
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        


       

        
        ##
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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        elif isinstance(model, H2OAutoMLWrapper):
                # Handling H2OAutoMLWrapper model
                h2o.init()
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Ensure date is aligned

                # Using a similar approach for confidence interval estimation
                error_margin = predictions['predicted'] * 0.05  # 5% error margin as an example
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                model_id = getattr(model, 'id', 'H2O_AutoML')  # Use a default id if not available

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })
        elif isinstance(model, MLModelWrapper):
                # Handling MLModelWrapper models
                X = forecast_data[model.feature_names]
                predictions = model.predict(X)

                # Convert predictions to DataFrame if necessary
                if not isinstance(predictions, pd.DataFrame):
                    predictions = pd.DataFrame(predictions, columns=['predicted'])

                # Aligning dates
                predictions['date'] = forecast_data['date'].values

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
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })

        # Add handling for other model types if necessary...

        return results
    
  

   
    ##
    def _create_future_dataframe(self, periods):
        """
        Create a DataFrame for future dates.
        """
        last_date_in_data = self.actual_data['date'].max()
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods, freq='M')
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
    def _generate_future_forecast_data(self, model):
        """
        Generate future forecast data for the model.
        """
        if not self.forecast_horizon:
            return None

        periods = self._parse_forecast_horizon(self.forecast_horizon)
        future_data = self._create_future_dataframe(periods)

        # For Prophet model, ensure the 'ds' column is present
        if isinstance(model, ProphetReg):
            future_data = future_data.rename(columns={'date': 'ds'})
        return future_data

        ##
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
from prophet import Prophet
from MLModelWrapper import MLModelWrapper
from H2OAutoMLWrapper import H2OAutoMLWrapper
from ArimaReg import ArimaReg
from ProphetReg import ProphetReg
from ModelTimeTable import ModelTimeTable
from AutoGluonTabularWrapper import AutoGluonTabularWrapper

import pandas as pd

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
        

           ##
    ##
    def forecast(self):
        forecast_results = []

        # Existing code for processing new data predictions
        for model in self.models:
            if self.new_data is not None:
                forecast_results.extend(self._predict_new_data(model))

            # Handling future forecasts with special treatment for Prophet
            if isinstance(model, ProphetReg):
                if self.future_data is not None:
                    future_data_prophet = self.future_data.rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
                elif self.forecast_horizon:
                    future_data_prophet = self._generate_future_forecast_data(model).rename(columns={'date': 'ds'})
                    forecast_results.extend(self._prophet_future_forecast(model, future_data_prophet))
            else:
                if self.future_data is not None:
                    future_forecast_data = self.future_data
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))
                elif self.forecast_horizon:
                    future_forecast_data = self._generate_future_forecast_data(model)
                    forecast_results.extend(self._predict_future_data(model, future_forecast_data))

        # Process actual data
        if self.actual_data is not None:
            actual_data_results = self._process_actual_data()
            forecast_results.extend(actual_data_results)

        # Combine and sort DataFrame from results, then remove duplicates
        forecast_df = pd.DataFrame(forecast_results)

        # Ensure 'date' column in forecast results is consistent
        if 'date' in forecast_df.columns and forecast_df['date'].dtype != 'datetime.date':
            forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date  # Convert to datetime.date

        # Sort and remove duplicates
        forecast_df.sort_values(by=['key', 'model_id', 'date'], inplace=True)
        #print("Final Forecast DataFrame:", forecast_df.tail(30))
        # Diagnostics: Inspect forecast_df before dropping duplicates
#         print("Inspecting forecast_df before drop_duplicates:")
#         print(forecast_df.dtypes)  # Print the data types of each column
#         print(forecast_df.head())  # Print the first few rows of the DataFrame

#         # Optional: Check each column for unhashable types
#         for column in forecast_df.columns:
#             if any(isinstance(x, pd.Series) for x in forecast_df[column]):
#                 print(f"Column {column} contains unhashable types.")

        forecast_df.drop_duplicates(inplace=True)

        return forecast_df






    ##
    def _prophet_future_forecast(self, model, future_data):
        # Specialized method for Prophet future forecasts
        print("Original future_data:", future_data.head())  # Debug: Check the initial future_data

        # Ensure the data is in the correct format for Prophet
        if 'ds' not in future_data.columns:
            if 'date' in future_data.columns:
                future_data = future_data.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Missing 'ds' column for Prophet future forecast.")

        print("Processed future_data for Prophet:", future_data.head())  # Debug: Check the processed future_data

        # Direct prediction using Prophet model
        prophet_future_forecast = model.predict(future_data)

        # Extracting forecast results with confidence intervals
        forecast_results = []
        for i, row in prophet_future_forecast.iterrows():
            forecast_results.append({
                'model_id': model.id,
                'model_desc': 'PROPHET',
                'key': 'future',
                'date': row['date'],  # Using 'ds' from prophet_future_forecast
                'value': row['predicted'],  # Use 'yhat' for predicted value
                'conf_lo': row['conf_lo'],  # Lower bound of confidence interval
                'conf_hi': row['conf_hi']  # Upper bound of confidence interval
            })

        return forecast_results

    ##
    def _predict_future_data(self, model, future_data):
        print(f"Processing future predictions for model: {type(model).__name__}")
        model_desc = self._get_model_type(model)  # Use _get_model_type to get the actual model type
        #model_id = getattr(model, 'id', 'N/A')  # Get the model's existing ID
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

                # Prepare a copy of the future_data DataFrame without the target column for prediction
                if model.target_column in future_data.columns:
                    future_data_for_prediction = future_data.drop(columns=[model.target_column], errors='ignore')
                else:
                    future_data_for_prediction = future_data

                # Convert the prepared DataFrame to H2OFrame
                h2o_future_data = h2o.H2OFrame(future_data_for_prediction)

                # Make predictions using the H2O AutoML model
                h2o_predictions = model.model.predict(h2o_future_data)
                predictions = h2o_predictions.as_data_frame()

                # Ensure the index of predictions matches future_data's index
                predictions.index = future_data.index

                # Add 'date' from future_data to predictions
                predictions['date'] = future_data['date']

                # Add confidence intervals
                error_margin = predictions['predict'] * 0.05  # Example error margin
                predictions['conf_lo'] = predictions['predict'] - error_margin
                predictions['conf_hi'] = predictions['predict'] + error_margin

                # Construct results without 'id'
                for _, row in predictions.iterrows():
                    result = {
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'future',
                        'date': row['date'],
                        'value': row['predict'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    }
                    results.append(result)

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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

    
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

            # Adding confidence intervals
            error_margin = predictions['predicted'] * 0.05  # Example error margin
            predictions['conf_lo'] = predictions['predicted'] - error_margin
            predictions['conf_hi'] = predictions['predicted'] + error_margin

            # Get the model name from the MLModelWrapper instance
            model_desc = getattr(model, 'model_name', 'ML Model Description Not Available')

            # Construct the result list
            for _, row in predictions.iterrows():
                result = {
                    'model_id': model_id,
                    'model_desc': model_desc,
                    'key': 'future',
                    'date': row['date'],
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                }
                results.append(result)

        # ... [handling for other models, if needed] ...

        return results
        
    
    
    def _process_actual_data(self):
        """
        Process actual data for forecasting, filtering out any future dates.
        """
        # Ensure target_column is set
        if not hasattr(self, 'target_column') or self.target_column not in self.actual_data.columns:
            raise ValueError("Target column not set or not found in actual data.")

        # Filter out rows where the target value is NaN (or apply other criteria)
        filtered_actual_data = self.actual_data.dropna(subset=[self.target_column])

        # Find the latest date in the filtered data
        latest_actual_date = filtered_actual_data['date'].max()

        # Additional filtering if needed
        filtered_actual_data = filtered_actual_data[filtered_actual_data['date'] <= latest_actual_date]

        #print("Filtered actual data (tail):", filtered_actual_data.tail())

        # Ensure a list is returned even if filtered_actual_data is empty
        return [{'model_id': 'Actual', 'model_desc': 'ACTUAL', 'key': 'actual',
                'date': row['date'], 'value': row[self.target_column], 'conf_lo': None, 'conf_hi': None}
                for _, row in filtered_actual_data.iterrows()] or []




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
            print("Original columns before processing for Prophet:", data.columns)
            if 'date' in data.columns:
                forecast_data = data.rename(columns={'date': 'ds'})
            elif 'ds' in data.columns:
                forecast_data = data
            else:
                raise KeyError("The DataFrame must contain a 'date' or 'ds' column for Prophet models.")
            print("Columns after processing for Prophet:", forecast_data.columns)

            # Additional check for 'ds' column
            if 'ds' not in forecast_data.columns:
                raise KeyError("Failed to create 'ds' column for Prophet model.")
        else:
            forecast_data = data

        return forecast_data


    ##
    def _predict_new_data(self, model):
        print(f"Processing predictions for model: {type(model).__name__}")
        forecast_data = self._generate_forecast_data(model, self.new_data)
        model_desc = getattr(model, 'description', 'No description available')

         # Generate a unique model_id for each model type
        model_id = self.model_id_counter
        self.model_id_counter += 1  # Increment the counter for the next model
        # Initialize an empty list for results
        results = []

        ##
        if isinstance(model, AutoGluonTabularWrapper):
            X_new = self.new_data.drop(columns=[self.target_column], errors='ignore')
            try:
                predictions_raw = model.predict(X_new)

                quantiles_df = model.predict_quantiles(X_new) if hasattr(model, 'predict_quantiles') else pd.DataFrame({'lower': [None] * len(predictions_raw), 'upper': [None] * len(predictions_raw)})

                predictions_df = predictions_raw.to_frame(name='predicted')
                predictions_df['date'] = self.new_data['date'].values
                predictions_df = pd.concat([predictions_df.reset_index(drop=True), quantiles_df.reset_index(drop=True)], axis=1)

                model_type = self._get_model_type(model)  # Use _get_model_type to get the actual model name

                for _, row in predictions_df.iterrows():
                    predicted_value = row['predicted'] if isinstance(row['predicted'], (int, float)) else row['predicted'].iloc[0]
                    result = {
                        'model_id': model_id,
                        'model_desc': model_type,  # Use model_type for the model description
                        'key': 'prediction',
                        'date': row['date'],
                        'value': predicted_value,
                        'conf_lo': row.get('lower', None),
                        'conf_hi': row.get('upper', None)
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error in predicting with AutoGluonTabularWrapper: {e}")
                return []

        


       

        
        ##
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
                    'value': row['predicted'],
                    'conf_lo': row['conf_lo'],
                    'conf_hi': row['conf_hi']
                })
        elif isinstance(model, H2OAutoMLWrapper):
                # Handling H2OAutoMLWrapper model
                h2o.init()
                h2o_predictions = model.model.predict(h2o.H2OFrame(forecast_data))
                predictions = h2o_predictions.as_data_frame()
                predictions = predictions[['predict']].rename(columns={'predict': 'predicted'})
                predictions['date'] = forecast_data['date'].values  # Ensure date is aligned

                # Using a similar approach for confidence interval estimation
                error_margin = predictions['predicted'] * 0.05  # 5% error margin as an example
                predictions['conf_lo'] = predictions['predicted'] - error_margin
                predictions['conf_hi'] = predictions['predicted'] + error_margin

                model_id = getattr(model, 'id', 'H2O_AutoML')  # Use a default id if not available

                # Process predictions to create results
                for _, row in predictions.iterrows():
                    results.append({
                        'model_id': model_id,
                        'model_desc': model_desc,
                        'key': 'prediction',
                        'date': row['date'],
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })
        elif isinstance(model, MLModelWrapper):
                # Handling MLModelWrapper models
                X = forecast_data[model.feature_names]
                predictions = model.predict(X)

                # Convert predictions to DataFrame if necessary
                if not isinstance(predictions, pd.DataFrame):
                    predictions = pd.DataFrame(predictions, columns=['predicted'])

                # Aligning dates
                predictions['date'] = forecast_data['date'].values

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
                        'value': row['predicted'],
                        'conf_lo': row['conf_lo'],
                        'conf_hi': row['conf_hi']
                    })

        # Add handling for other model types if necessary...

        return results
    
  

   
    ##
    def _create_future_dataframe(self, periods):
        """
        Create a DataFrame for future dates.
        """
        last_date_in_data = self.actual_data['date'].max()
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=periods, freq='M')
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
    def _generate_future_forecast_data(self, model):
        """
        Generate future forecast data for the model.
        """
        if not self.forecast_horizon:
            return None

        periods = self._parse_forecast_horizon(self.forecast_horizon)
        future_data = self._create_future_dataframe(periods)

        # For Prophet model, ensure the 'ds' column is present
        if isinstance(model, ProphetReg):
            future_data = future_data.rename(columns={'date': 'ds'})
        return future_data

        ##
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
