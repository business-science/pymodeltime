### pymodeltime

- [Univariate Regression - Open in Colab](https://colab.research.google.com/drive/1CK6Zu_4lBYTkihyEQ1U_4wiViw2qjFBr?usp=sharing)
- [Demand Forecasting - Open in Colab](https://colab.research.google.com/drive/1OPxgPBLiIQpIE1T1ZnHaIOI005GOKIgb?usp=sharing)


### Installation


```
!pip install git+https://github.com/business-science/pymodeltime.git
```

### Usage Examples

#### Importing Necessary Modules

```python
from pymodeltime import ArimaReg, ProphetReg
from pymodeltime import ModelTimeTable, ModelTimeAccuracy, ModelTimeCalibration, ModelTimeForecast, ModelTimeRefit
from pymodeltime import MLModelWrapper, H2OAutoMLWrapper
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
```

##### Create instances of the ML models 
```
rf_model = RandomForestRegressor()
ml_rf_wrapper = MLModelWrapper(rf_model, feature_columns, "Random Forest")
##Machine Learning Model Integration
xgb_model = XGBRegressor()
ml_xgb_wrapper = MLModelWrapper(xgb_model, feature_columns, "XGBoost")

# Fit the models
ml_rf_wrapper.fit(train_data, y_train)
ml_xgb_wrapper.fit(train_data, y_train)
```
#####  AutoGluon models 
```
from pymodeltime import AutoGluonTabularWrapper

auto_gluon_wrapper = AutoGluonTabularWrapper(target_column="GDP")
auto_gluon_wrapper.fit(train_data)
quantile_predictions = auto_gluon_wrapper.predict_quantiles(test_data)
```
#### H2O AutoML Integration


```
h2o.init()

# Convert to H2OFrame and train
h2o_train = h2o.H2OFrame(train_data)
automl = H2OAutoML(max_models=10, seed=1, max_runtime_secs=300)
automl.train(x=feature_columns, y='GDP', training_frame=h2o_train)

# Extract the leaderboard
lb = automl.leaderboard
lb_df = lb.as_data_frame()
print(lb_df.head())

# Get the best model
best_h2o_model = automl.leader
h2o_automl_wrapper = H2OAutoMLWrapper(best_h2o_model, target_column='GDP')
```

#### Prophet Model for Multivariate Time Series
```
prophet_model_multi = ProphetReg(seasonality_yearly=True, seasonality_weekly=True,
                                 seasonality_daily=False, changepoint_range=0.5,
                                 prior_scale_seasonality=5.0, season='multiplicative',
                                 interval_width=0.95)

prophet_model_multi.fit(train_data, target_column='GDP', date_column='date', regressors=feature_columns)
```
#### Model Calibration and Accuracy Evaluation
```
modeltime_table = ModelTimeTable(h2o_automl_wrapper, ml_xgb_wrapper, ml_rf_wrapper, prophet_model_multi)
```
#### Model Calibration
```
model_time_calibrator = ModelTimeCalibration(modeltime_table, test_data, target_column='GDP')
model_time_calibrator.calibrate()
calibration_results_df = model_time_calibrator.get_calibration_results()
print(calibration_results_df)
```
#### Model Accuracy
```
modeltime_accuracy = ModelTimeAccuracy(modeltime_table, test_data, target_column='GDP')
accuracy_results = modeltime_accuracy.calculate_accuracy()
print(accuracy_results)

```
