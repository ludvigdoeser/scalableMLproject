import os
import modal
from create_feature_view import fix_data_from_feature_view

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    packages = ["hopsworks","pandas","numpy","tensorflow","pandas_market_calendars","joblib","scikit-learn","dataframe-image"]
    hopsworks_image = modal.Image.debian_slim().pip_install(packages)
    
    # schedule = modal.Period(days=1)
    @stub.function(image=hopsworks_image, secret=modal.Secret.from_name("scalableML"))
    def f():
        g()

def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    from sklearn.model_selection import train_test_split

    import pandas as pd
    import numpy as np
    from datetime import date, datetime, timedelta
    import joblib
    import inspect 
    import dataframe_image as dfi

    # Tensorflow
    import tensorflow
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding

    today = date.today()
    print("Today's date:", today)
    
    print('Login & Fetching feature view from hopsworks...')
    project = hopsworks.login(api_key_value="qufVZ6euYPKebH4n.zmsSbdrZmtbkv214l1hmxVukJy1s9tCRYHvFNDnE8oA8K4GXfLEzK9TKBHGQpGyr")
    fs = project.get_feature_store()

    # Get feature view and data 
    fv = fs.get_feature_view(
        name = 'stock_pred_modal',
        version = 1
    )
    
    df, _ = fv.training_data()
    
    # Get model from model registry:
    mr = project.get_model_registry()

    model = mr.get_model("stock_pred_model", version = 7)
    model_dir = model.download()

    loaded_model = tensorflow.saved_model.load(model_dir)
    serving_function = loaded_model.signatures["serving_default"]

    # Load data needed for inference 
 
    # For manual manipulation:
    today = datetime.strptime("2023-01-03","%Y-%m-%d") 
    
    df = fix_data_from_feature_view(df,(today - timedelta(days=30)).strftime("%Y-%m-%d"),today.strftime("%Y-%m-%d"))
    
    # Only use the last 7 business days
    df = df[-7:]

    # Drop date and swap order of columns... because of how training was set up
    df = df.drop(columns='date')
    column_order = ['exp_mean_7_days','close']
    df = df.reindex(columns=column_order) # Use reindex to change the order of columns
    print(df)
    
    c = tensorflow.constant(df.values.reshape(-1, df.shape[0], df.shape[1]), tensorflow.float32)
    y_pred = serving_function(c)['dense_1'].numpy()
    print('Prediction = ',y_pred[0][0])
    
    # Need the inverse transformation:
    td_transformation_functions = fv._transformation_functions 
    td_transformation_function = td_transformation_functions['close']
    sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
    param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
    
    # Perform the transformation 
    df_pred = pd.DataFrame(y_pred,columns=['pred_close'])
    pred = df_pred['pred_close'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"])
   
    truth = df['close'].to_numpy()[-1]
    print(truth)
    df_temp = pd.DataFrame([truth],columns=['truth'])
    yesterday_truth = df_temp['truth'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]) 
    
    data = {
        'date': [(today + timedelta(days=1)).strftime("%Y-%m-%d")],
        'prediction_close': [pred[0]],
        'true_close': [np.nan]
    }
    
    monitor_df = pd.DataFrame(data)
    
    print("Creating/getting feature group: stock_predictions")
    monitor_fg = fs.get_or_create_feature_group(name="stock_predictions",
                                         version=1,
                                         description="Tesla stock predictions from stock data and news sentiment data",
                                         primary_key=["date"],
                                         online_enabled=True)
    """
    monitor_fg.insert(monitor_df, 
                      write_options={"wait_for_job" : False})
    
    print('Load it')
    monitor_fg = fs.get_feature_group(name="stock_predictions",
                                             version=1,
                                             )
    """
    
    fg_query = monitor_fg.select_all()
    history_pred_df = fg_query.read()
    print('history_pred_df: ')
    print(history_pred_df)
    
    # Add our prediction to the history
    history_pred_df = pd.concat([history_pred_df, monitor_df])
    print('history_pred_df: ')
    print(history_pred_df)
    
    # Add true label on yesterdays NaN...
    history_pred_df.iloc[-2, history_pred_df.columns.get_loc('true_close')] = yesterday_truth[0]
    print('history_pred_df: ')
    print(history_pred_df)
    
    print('Insert this dataframe to the feature group')
    monitor_fg.insert(history_pred_df, 
                      write_options={"wait_for_job" : False},
                      overwrite=True)
    
    """
    # Save the latest 7 days to be presented at the UI:
    df_recent = history_pred_df.tail(7)
    
    # Store things to be presented in UI:
    dataset_api = project.get_dataset_api()    
    
    dfi.export(df_recent, './df_recent_tsla_predictions.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_tsla_predictions.png", "Resources/images", overwrite=True)
    """
    
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
    
    
    
    