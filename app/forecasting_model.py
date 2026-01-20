import pandas as pd
import numpy as np
from train_test import data_prep_train,data_prep_test,data_prep_test_act
from functions import get_block_number, db_connection
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

engine = db_connection()

def create_lgbm_forecaster(window_length,linear_lambda):
    return make_reduction(LGBMRegressor(boosting_type='gbdt',random_state=42,verbosity=-1,
                                        linear_tree=True,
                                        linear_lambda=linear_lambda,
                                        objective='regression',
                                        metric='rmse',
                                        n_jobs=-1),
                        window_length=window_length, 
                        strategy="recursive")



def direct_forecast_sktime_lgbm_JM(df_train,df_test, wl,ll,model_id):

    results_96 = [] # Initialising list to store model predictions
    target = ['demand']
    features = [    'hi','normal_holiday','special_day', 
                    'is_day_before_holiday','is_day_after_holiday',
                    'nh_dow_interaction', 'sd_dow_interaction',
                    'minute_sin', 'minute_cos',
                    'hour_sin', 'hour_cos',
                    'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos']
    # Training Data
    train_df = df_train[target + features].copy()
    # print(f"Training from {train_df.index.min()} to {train_df.index.max()}")
    # Test Data
    test_df = df_test[features].copy()  
    # print(f"Predicting from {test_df.index.min()} to {test_df.index.max()}")

    # X_Train and y_Train
    X_train = train_df.drop(columns=target).copy()
    y_train = train_df[target].copy()
    
    X_test = test_df.copy()
    
    forecaster = create_lgbm_forecaster(window_length=wl,linear_lambda=ll)
    forecaster.fit(y_train, X_train)
    fh = ForecastingHorizon(X_test.index, is_relative=False)
    
    y_pred_org = forecaster.predict(fh,X_test)
    y_pred = pd.Series(y_pred_org.values.ravel(), index=X_test.index)
    results_96.extend(zip(X_test.index[-96:],y_pred[-96:]))


# Prepare results with block numbers
    results_96_df = pd.DataFrame(results_96, columns=['datetime', 'forecasted_demand'])
    results_96_df['datetime'] = pd.to_datetime(results_96_df.datetime)
    results_96_df['date'] = results_96_df['datetime'].dt.date
    results_96_df['time'] = results_96_df['datetime'].dt.time
    results_96_df['block'] = results_96_df['time'].apply(get_block_number)
    results_96_df['model_id'] = model_id
    results_96_df_final = results_96_df[['datetime','block', 'forecasted_demand']]
    # results_96_df_final = results_96_df[['datetime','date','time','block', 'forecasted_demand','model_id']]
    # Convert start_time and end_time to datetime.time
#     start_time = datetime.strptime(start_time, '%H:%M:%S').time()
#     end_time = datetime.strptime(end_time, '%H:%M:%S').time()

# # Fix the filtering condition
#     filtered_df = results_96_df[
#         (results_96_df['datetime'].dt.time >= start_time) &
#         (results_96_df['datetime'].dt.time <= end_time)
#     ]
    return results_96_df_final


from sqlalchemy import text
import pandas as pd

def get_forecasted_weather(input_date, sp_id):
    ip_date = pd.to_datetime(input_date)
    test_start = ip_date + pd.Timedelta(hours=8, minutes=15) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23, minutes=45)
    query_weather = f'''
        SELECT datetime, date, "time", block, temp, humidity
        FROM "AEML".t_actual_weather
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        AND sp_id = '{sp_id}'
        ORDER BY datetime ASC
    '''
    w = pd.read_sql(query_weather, con=engine)
    w['datetime'] = pd.to_datetime(w['datetime'])
    w['date'] = pd.to_datetime(w['date'])
    return w[-96:]

def get_weather_data(pred_date, sp_id, start_time, end_time, temp_change=0, hum_change=0):
    ip_date = pd.to_datetime(pred_date)
    test_start = ip_date + pd.Timedelta(hours=8, minutes=15) - pd.DateOffset(days=1)
    test_end = ip_date + pd.Timedelta(hours=23, minutes=45)
    
    query_weather = f'''
        SELECT datetime, date, "time", block, temp, humidity
        FROM "AEML".t_actual_weather
        WHERE datetime >= '{test_start}' 
        AND datetime <= '{test_end}'
        AND sp_id = '{sp_id}'
        ORDER BY datetime ASC
    '''
    
    w = pd.read_sql(query_weather, con=engine)
    w['datetime'] = pd.to_datetime(w['datetime'])
    w['date'] = pd.to_datetime(w['date'])
    w['time'] = pd.to_datetime(w['time'], format='%H:%M:%S').dt.time

    start_time = pd.to_datetime(start_time, format='%H:%M:%S').time()
    end_time = pd.to_datetime(end_time, format='%H:%M:%S').time()

    # Apply the mask and force update
    mask = (w["time"] >= start_time) & (w["time"] <= end_time)
    w.loc[mask, "temp"] = w.loc[mask, "temp"] + temp_change
    w.loc[mask, "humidity"] = w.loc[mask, "humidity"] + hum_change
    return w

def fetch_cost_from_db(input_date):
    query = """
    SELECT "Date","Time Block", "MCP (Rs/MWh) *"  -- Use double quotes for special column names
    FROM "AEML"."t_market_data"
    WHERE "Date" = %s
    """
    cost_df = pd.read_sql(query, con=engine, params=(input_date,))  
    return cost_df


def calculate_total_cost(input_date, start_time=None, end_time=None, temp_change=0, hum_change=0):
    try:
        # Ensure engine is defined
        if "engine" not in globals():
            raise ValueError("Database connection (engine) is not available.")

        # Step 1: Fetch Cost Data from DB
        query = """
        SELECT "Date", "Time Block", "MCP (Rs/MWh) *"
        FROM "AEML"."t_market_data"
        WHERE "Date" = %s
        """
        cost_df = pd.read_sql(query, con=engine, params=(input_date,))

        if cost_df.empty:
            print("No cost data found for the given date.")
            return None
        
        # Step 2: Forecast Demand
        forecasted_demand = direct_forecast_sktime_lgbm_JM(
            df_train=data_prep_train(input_date=input_date, lb=1, sp_id=4),
            df_test=data_prep_test(input_date=input_date, sp_id=4, start_time=start_time, end_time=end_time, temp_change=temp_change, hum_change=hum_change),
            wl=96,
            ll=0.7,
            model_id=7
        )

        # Ensure forecasted_demand is a valid DataFrame
        if not isinstance(forecasted_demand, pd.DataFrame):
            raise ValueError("Forecasted demand function did not return a valid DataFrame.")

        # Ensure required columns exist in forecasted_demand
        required_columns = {"datetime", "forecasted_demand"}
        if not required_columns.issubset(forecasted_demand.columns):
            raise ValueError(f"Missing required columns in forecasted demand: {required_columns - set(forecasted_demand.columns)}")

        # Step 3: Convert 'datetime' to 'Time Block' format
        def convert_to_time_block(dt):
            hour = dt.hour
            minute = dt.minute
            start_minute = (minute // 15) * 15
            end_minute = start_minute + 15
            if end_minute == 60:
                end_hour = hour + 1
                end_minute = 0
            else:
                end_hour = hour
            return f"{hour:02}:{start_minute:02} - {end_hour:02}:{end_minute:02}"

        forecasted_demand["Time Block"] = forecasted_demand["datetime"].apply(convert_to_time_block)

        # Step 4: Merge Forecasted Demand with Cost Data on "Time Block"
        merged_df = forecasted_demand.merge(cost_df, on="Time Block", how="inner")

        # Step 5: Compute Cost for Each Time Block
        merged_df["Total Cost (Rs)"] = merged_df["forecasted_demand"] * merged_df["MCP (Rs/MWh) *"]
        merged_df["Total Cost (Rs)"] = merged_df["Total Cost (Rs)"].astype(int)
        return merged_df  # Returns DataFrame with total cost per time block

    except Exception as e:
        print(f"Error: {e}")
        return None
 