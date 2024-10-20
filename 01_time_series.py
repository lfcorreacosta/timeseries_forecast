# BUSINESS SCIENCE UNIVERSITY
# WALMART FORECASTING DAILY DEMAND  
# PART 1: FORECAST ANALYSIS
# ----

# GOAL: Forecast Daily Demand for Walmart Products for N-Days into the Future

# ABOUT THE DATA

# These data are from the the M5 Forecasting Competition. 

# LIBRARIES
import os

import numpy as np
import pandas as pd
import plotly.express as px

import pycaret.regression as reg

from utils.timeseries_helpers import (
    extend_single_timeseries_frame,
    make_timeseries_features,
    forecast_single_timeseries
)


# DATA IMPORT ----

walmart_sales_raw_df = pd.read_csv("data/walmart_sales.csv")

walmart_sales_raw_df.tail(3)

unique_ids = walmart_sales_raw_df['item_id'].unique()
unique_ids

# 1.0 THE FORECASTING PROCESS ----

# EXTEND THE DATA --- 

# Select a single id
id = unique_ids[1]

# Extend a signle timeseries
single_timeseries_extended_df = extend_single_timeseries_frame(
    data = walmart_sales_raw_df,
    id   = id,
    h    = 60
)
single_timeseries_extended_df

# Visualize the extended timeseries
px.line(
    single_timeseries_extended_df,
    y          = 'value',
    line_group = 'item_id',
    color      = 'key',
)


# FEATURE ENGINEERING ----

single_timeseries_feat_df = make_timeseries_features(single_timeseries_extended_df)

# Split into Future and Actual
future_df = single_timeseries_feat_df \
    .query("key == 'FUTURE'") 
    
actual_df = single_timeseries_feat_df \
    .query("key == 'ACTUAL'")


# MODELING ----

df = actual_df \
    .drop(['item_id', 'key'], axis=1) 

# Numeric Columns
float_cols = df \
    .drop(['value'], axis=1) \
    .select_dtypes(include="float64").columns.to_list() 

int_cols = df.select_dtypes(include="int64").columns.to_list()

numeric_columns = [*float_cols, *int_cols]

# * Setup the Regressor ----
exp = reg.setup(
    data        = df, 
    target      = 'value', 
    train_size  = 0.8,
    session_id  = 123,
    
    numeric_features = numeric_columns,
    silent=True
)

# * Make A Machine Learning Model ----
xgb_model = reg.create_model(
    estimator = 'xgboost',
    cross_validation=True,
    fold=5,
    
    # xgboost args
    eta=0.2,
    gamma=0.1,
    max_depth=6,
)


# CHECK MODEL ----

reg.plot_model(xgb_model)


# FEATURE IMPORTANCE -----

# Basic Feature Importance
reg.plot_model(xgb_model, plot = 'feature')


# FINALIZE MODEL ----

xgb_model_finalized = reg.finalize_model(xgb_model)


# MAKE PREDICTIONS & UPDATE FUTURE DF ----

# Make predictions
predictions_df = reg.predict_model(
    xgb_model_finalized, 
    data      = future_df
)

predictions_df['Label']

# Update the future_df with the predictions

future_df['value'] = predictions_df['Label']

future_df

# Combine the future and actual data

combined_df = pd.concat([actual_df, future_df], axis = 0)

# Visualize the Forecast
px.line(
    combined_df,
    y          = 'value',
    line_group = 'item_id',
    color      = 'key',
)

# 2.0 SIMPLIFYING THE FORECASTING PROCESS ----

unique_ids

reg.models()

# * Xgboost ----
forecast_test = forecast_single_timeseries(
    data = walmart_sales_raw_df,
    id   = unique_ids[11],
    h    = 180,
    
    estimator='xgboost',
    eta=0.2,
    max_depth=6,
)

px.line(
    forecast_test,
    y          = 'value',
    line_group = 'item_id',
    color      = 'key',
)


# * Random Forest ----
forecast_test = forecast_single_timeseries(
    data = walmart_sales_raw_df,
    id   = unique_ids[11],
    h    = 180,
    
    estimator='rf',
)

px.line(
    forecast_test,
    y          = 'value',
    line_group = 'item_id',
    color      = 'key',
)

# * Elastic Net ----
forecast_test = forecast_single_timeseries(
    data = walmart_sales_raw_df,
    id   = unique_ids[11],
    h    = 180,
    
    estimator='en',
)

px.line(
    forecast_test,
    y          = 'value',
    line_group = 'item_id',
    color      = 'key',
)
