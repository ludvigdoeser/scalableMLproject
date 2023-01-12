import os
import modal
from create_feature_view import fix_data_from_feature_view
from feature_preprocessing import next_business_day, today_is_a_business_day

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    packages = ["hopsworks","pandas","numpy","tensorflow","pandas_market_calendars","joblib","scikit-learn","dataframe-image","matplotlib","yfinance"]
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

    import matplotlib.pyplot as plt 
    import pandas as pd
    import numpy as np
    from datetime import date, datetime, timedelta
    import joblib
    import inspect 
    import dataframe_image as dfi
    import yfinance as yf

    # Tensorflow
    import tensorflow
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding

    today = date.today()
    print("Today's date:", today)
    
    if not today_is_a_business_day(today):
        return #do nothing these days
    else:
        print('Login & Fetching feature view from hopsworks...')
        project = hopsworks.login()
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

        # ---------------------------------------------------------
        # For manual manipulation:
        today = datetime.strptime("2023-01-12","%Y-%m-%d")  # have not run 12 yet!
        # ---------------------------------------------------------
        
        df = fix_data_from_feature_view(df,(today - timedelta(days=30)).strftime("%Y-%m-%d"),today.strftime("%Y-%m-%d"))

        # Only use the last 7 business days
        df = df[-7:]

        print('Feature view:')
        print(df)
        
        # Drop date and swap order of columns... because of how training was set up
        df = df.drop(columns='date')
        column_order = ['exp_mean_7_days','close']
        df = df.reindex(columns=column_order) # Use reindex to change the order of columns

        c = tensorflow.constant(df.values.reshape(-1, df.shape[0], df.shape[1]), tensorflow.float32)
        y_pred = serving_function(c)['dense_1'].numpy()
        print('Prediction for tomorrow = ',y_pred[0][0])

        # Need the inverse transformation:
        td_transformation_functions = fv._transformation_functions 
        td_transformation_function = td_transformation_functions['close']
        sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
        param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])

        # Perform the transformation 
        df_pred = pd.DataFrame(y_pred,columns=['pred_close'])
        pred = df_pred['pred_close'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"])

        truth_today = df['close'].to_numpy()[-1]
        print('Today truth: ',truth_today) #remember you're predicting tomorrow...
        df_temp = pd.DataFrame([truth_today],columns=['truth'])
        truth_today = df_temp['truth'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]) 

        truth_yesterday = df['close'].to_numpy()[-2]
        print('Yesterday truth: ',truth_yesterday) #to correct the daily return from "yesterday" (=last open business day) 
        df_temp = pd.DataFrame([truth_yesterday],columns=['truth'])
        truth_yesterday = df_temp['truth'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"]) 

        # Find tomorrow
        tomorrow = next_business_day(today)

        data = {
            'date': [tomorrow],
            'predicted_end_of_day_price': [pred[0]],
            'predicted_daily_return': [(pred[0]-truth_today[0])/truth_today[0]],
            'true_end_of_day_price': [np.nan],
            'true_daily_return': [np.nan]
        }

        monitor_df = pd.DataFrame(data)
        print('monitor_df:')
        print(monitor_df)

        print("Creating/getting feature group: stock_predictions")
        monitor_fg = fs.get_feature_group(name="stock_predictions",version=1)

        """
        monitor_fg = fs.get_or_create_feature_group(name="stock_predictions",
                                             version=1,
                                             description="Tesla stock predictions from stock data and news sentiment data",
                                             primary_key=["date"],
                                             online_enabled=True)
        # Run these lines the first time:
        monitor_fg.insert(monitor_df, 
                          write_options={"wait_for_job" : False})
        print('Load it')
        monitor_fg = fs.get_feature_group(name="stock_predictions",version=1)
        """
        
        fg_query = monitor_fg.select_all()
        history_pred_df = fg_query.read()

        history_pred_df = history_pred_df.sort_values("date")
        history_pred_df = history_pred_df.reset_index(drop=True)
        print('history_pred_df: ')
        print(history_pred_df)

        if str(tomorrow) == str(history_pred_df['date'].iloc[-1]):
            print('Do not insert anything... because tomorrow has already been predicted')
        else:
            print('Adding the last row to the feature group:')

            # Add our prediction to the history
            print('Add our prediction to the history')
            history_pred_df = pd.concat([history_pred_df, monitor_df])
            history_pred_df = history_pred_df.sort_values("date")
            history_pred_df = history_pred_df.reset_index(drop=True)
            print('history_pred_df: ')
            print(history_pred_df)

            # Add true label on yesterdays NaN...
            print('Add true label on yesterdays NaN...')
            history_pred_df.iloc[-2, history_pred_df.columns.get_loc('true_end_of_day_price')] = truth_today[0]
            history_pred_df.iloc[-2, history_pred_df.columns.get_loc('true_daily_return')] = (truth_today[0]-truth_yesterday[0])/truth_yesterday[0]
            print('history_pred_df: ')
            print(history_pred_df)

            """
            print('Insert this dataframe to the feature group')
            monitor_fg.insert(history_pred_df.tail(2), 
                              write_options={"wait_for_job" : False})
            """

            print('Save png of the recent predictions')
            # Save the latest 7 days to be presented at the UI:
            df_recent = history_pred_df.tail(7)
            df_recent = df_recent.rename(columns={'date': 'Date',
                                                  'predicted_end_of_day_price': 'Pred [$]',
                                                  'predicted_daily_return': 'Pred [%]',
                                                  'true_end_of_day_price': 'True [$]',
                                                  'true_daily_return': 'True [%]'
                                                 })

            # Store things to be presented in UI:
            dataset_api = project.get_dataset_api()  

            dfi.export(df_recent, './df_recent_tsla_predictions.png', table_conversion = 'matplotlib')
            dataset_api.upload("./df_recent_tsla_predictions.png", "Resources/images", overwrite=True)
            
            # Get the current date and the date one month ago
            today = datetime.now()
            print('today = ',today)
            a_week_ago = today - timedelta(days=10)

            # Get the TSLA stock data from yfinance
            tsla = yf.Ticker("TSLA") #VEFAB.ST

            # get historical market data
            data = tsla.history(start=a_week_ago, end=today)
            data = data.drop(columns=['Dividends','Stock Splits'])
            print('data = ',data)
            data['Close'].plot(c='lime',lw=1)
            print('close = ',data['Close'])
            
            fig = plt.gcf()
            fig.set_size_inches(10, 4)
            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams.update({'font.size': 10})
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.ylabel('End-of-Day Price [$]')
            
            dates_temp = df_recent['Date'].to_numpy()            
            pred_temp = df_recent['Pred [$]'].to_numpy()
            
            plt.scatter(dates_temp[:-1],pred_temp[:-1],s=20,c='darkorange')
            ax = plt.gca()

            plt.scatter(dates_temp[-1:],pred_temp[-1:],s=20,c='k')
            ax.axhline(pred_temp[-1:],c='k',alpha=0.3,ls='--')
            
            for pred,date_pred in zip(pred_temp,dates_temp):
                prev_date = (datetime.strptime(date_pred,"%Y-%m-%d")-timedelta(days=1)).strftime("%Y-%m-%d")
                close_prev_date = data.loc[data.index == prev_date, 'Close'].values[0]
                plt.plot([pd.Period(prev_date,'B'),pd.Period(date_pred,'B')],[close_prev_date,pred],c='darkorange')

            # Fill the area under the line with green
            #ax.fill_between(data.index, data['Close'], where=(data['Close']>0), color='lime', alpha=0.1)

            ax.legend(['End-of-Day Price','Past predictions','Prediction for tomorrow'],
                      loc='upper center', bbox_to_anchor=(0.5, 1.08),
                      fancybox=True, shadow=True, ncol=5)
                      
            xlim_min = (datetime.strptime(dates_temp[0],"%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            xlim_max = (datetime.strptime(dates_temp[-1],"%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            plt.xlim([xlim_min,xlim_max])
            plt.ylim([np.min(data['Close'])-10,np.amax(data['Close'])+10])
            plt.xlabel('Date')
            
            plt.savefig("./stock_price_w_pred.png")
            dataset_api.upload("./stock_price_w_pred.png", "Resources/images", overwrite=True)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
    