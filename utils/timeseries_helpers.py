
import numpy as np
import pandas as pd
import pycaret.regression as reg

def extend_single_timeseries_frame(data, id, h = 60):
    """Extends a single time series by h periods.

    Args:
        data (DataFramne): A sample of the walmart sales data
        id (string): The id of the item to extend from the item_id column
        h (int, optional): The number of periods to extend. Defaults to 60.

    Returns:
        DataFrame: 
            - The time series is stored as the index
            - The item_id is stored as a column
            - The value is stored as a column
            - The key is stored as a column (ACTUAL or FUTURE)
    """
    
    # Filter to a single item

    _filtered_df = data \
        [data['item_id'] == id] \
        .copy()

    # Convert to datetime
    _filtered_df['date'] = pd.to_datetime(_filtered_df['date'])

    # Reindex to date
    _filtered_df = _filtered_df.set_index('date')

    # Extend Timeseries

    timeseries = _filtered_df.index

    new_timeseries = pd.date_range(
        start   = timeseries[-1] + pd.Timedelta(days = 1),
        periods = h
    )

    # Make future dataframe

    _ids  = np.repeat(id, h)
    _vals = np.repeat(np.nan, h)

    new_df = pd.DataFrame(
        dict(
            item_id = _ids,
            value = _vals
        ),
    ) \
        .set_index(new_timeseries) \
        .assign(key = "FUTURE") 

    # Make actual dataframe

    old_df = pd.DataFrame(
        dict(
            item_id = _filtered_df['item_id'],
            value = _filtered_df['value'] 
        )
    ) \
        .set_index(_filtered_df.index) \
        .assign(key = "ACTUAL") 
    
    # Combine old and new df
    ret_df = pd.concat([old_df, new_df], axis = 0) 
    
    return ret_df


def extend_all_timeseries_frame(data, h = 60):
    """Extends all time series in the walmart sales data by h periods.

    Args:
        data (DataFrame): The walmart sales data that will be extended
        h (int, optional): The number of periods to extend the data. Defaults to 60.

    Returns:
        DataFrame: A dataframe with all time series extended by h periods. A key column is added to indicate whether the data is actual or future. The time series is also reindexed to a datetime index.
        
            - The time series is stored as the index
            - The item_id is stored as a column
            - The value is stored as a column
            - The key is stored as a column (ACTUAL or FUTURE)  
    """
    
    unique_ids = data['item_id'].unique()
    
    _list_of_dfs = []
    
    for id in unique_ids:
        
        df = extend_single_timeseries_frame(
            data = data,
            id   = id,
            h    = h
        )
        
        _list_of_dfs.append(df)
        
    ret_df = pd.concat(_list_of_dfs, axis = 0)
    
    return ret_df


def make_timeseries_features(data):
    """Makes time series features from a dataframe with a datetime index.
    
    Args:
        data (DataFrame): A dataframe with a datetime index.
        
    Returns:
        DataFrame: A dataframe with the following features:
        
        - index.num: The index as a number
        - year: The year
        - quarter: The quarter of the year
        - month: The month of the year
        - mday: The day of the month
        - week: The week of the year
        - wday: The day of the week
    
    """
    
    data['index.num']   = data.index.astype(np.int64) // 10**9
    data['year']        = data.index.year
    data['quarter']     = data.index.quarter
    data['month']       = data.index.month
    data['mday']        = data.index.day
    data['week']        = data.index.week
    data['wday']        = data.index.dayofweek
    
    return data


def forecast_single_timeseries(data, id, h = 60, estimator='xgboost', **kwargs):
    """Forecasts a single timeseries
    
    Args:
        data (DataFrame): The walmart sales data that will be extended
        id (int): The id of the timeseries to forecast
        h (int, optional): The number of periods to extend the data. Defaults to 60.
        estimator (str, optional): The estimator to use. Defaults to 'xgboost'.
        **kwargs: Additional arguments to pass to the model
    
    Returns:
        DataFrame: A dataframe with the timeseries forecasted. A key column is added to indicate whether the data is actual or future. The time series is also reindexed to a datetime index.
            
            - The time series is stored as the index
            - The item_id is stored as a column
            - The value is stored as a column
            - The key is stored as a column (ACTUAL or FUTURE)  
    """
    
    # Extend the timeseries
    single_timeseries_extended_df = extend_single_timeseries_frame(
        data = data,
        id   = id,
        h    = h,        
    )
    
    # Make features
    single_timeseries_feat_df = make_timeseries_features(single_timeseries_extended_df)
    
    # Split into Future and Actual
    future_df = single_timeseries_feat_df \
        .query("key == 'FUTURE'") 
        
    actual_df = single_timeseries_feat_df \
        .query("key == 'ACTUAL'")
        
    # * Modeling ----
    
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
        estimator = estimator,
        cross_validation=False,
        
        # xgboost args
        **kwargs
        
    )
    
    xgb_model_finalized = reg.finalize_model(xgb_model)
    
    # Make predictions
    predictions_df = reg.predict_model(
        xgb_model_finalized, 
        data      = future_df
    )
    
    # Update the future_df with the predictions
    future_df['value'] = predictions_df['Label']
    
    # Combine the future and actual data
    combined_df = pd.concat([actual_df, future_df], axis = 0)
    
    return combined_df

