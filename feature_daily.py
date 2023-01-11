import os
import modal
from feature_preprocessing import getNews, pre_process_news, exponential_moving_average

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    packages = ["hopsworks","pandas","numpy","tensorflow","pandas_market_calendars","joblib","scikit-learn","yfinance"]
    hopsworks_image = modal.Image.debian_slim().pip_install(packages)
    
    # schedule this function to run every day at 21:05 (stock market closes 21:00 greenwich time)
    @stub.function(image=hopsworks_image, schedule=modal.Cron("5 21 * * *"), secret=modal.Secret.from_name("scalableML"))
    def f():
        g()
        
def g():
    import hopsworks
    import pandas as pd
    import numpy as np
    from datetime import date, datetime, timedelta
    import joblib
    import inspect 
    import yfinance as yf 
    
    print('Login & Fetching feature view from hopsworks...')
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    today = date.today()
    print("Today's date:", today)
    
    # ----------------------------------------------------------------------------------------
    # NEWS:
    # ----------------------------------------------------------------------------------------

    # Set the API endpoint and your API key
    endpoint = "https://eodhistoricaldata.com/api/news"
    api_key = "63b2efc1180c66.27144598"

    # Set the ticker symbol
    ticker = "TSLA.US" #TSLA
    
    #Create a Pandas dataframe from the response
    news_df = getNews(api_key,endpoint,ticker,today,today,num=100)
    print('Number of news articles today: ',len(news_df.index))
    print('Here are some of them:')
    print(news_df)
    
    # Fetch existing feature group
    news_sentiment_fg = fs.get_feature_group(name="news_sentiment",version=1)

    fg_query = news_sentiment_fg.select_all()
    df_old_news = fg_query.read() 
    df_old_news = df_old_news.sort_values("date")
    df_old_news = df_old_news.reset_index(drop=True)
    print('Existing news feature group:')
    print(df_old_news)
    
    #df_old_news = df_old_news.drop(df_old_news.index[-1]) #only use if made a mistake..

    # Get the exponential average for today
    
    # How many days to look back for exponential average (to be computed further down)
    window = 7
    
    # If no news:
    if len(news_df.index)==0:
        # Add polarity=0 (neutral) if no news that day
        news_df = pd.DataFrame({'date':[today.strftime("%Y-%m-%d") ],
                                'polarity':[0], 
                                'exp_mean_7_days':[np.nan]})
    else:
        
        # Feature engineering step:
        news_df = pre_process_news(news_df) 
        news_df = news_df.reset_index() # move "date"-index into its own column
        news_df['date'] = news_df['date'].apply(lambda x: x.strftime("%Y-%m-%d")) # Update how date is written:
        news_df[f'exp_mean_{window}_days'] = [np.nan] # temporarily set it to nan to be able to concat in next step

    df = pd.concat([df_old_news, news_df])
    df = exponential_moving_average(df, window=window) # now update the exponential_moving_average 
    df = df.reset_index(drop=True)
    print('Updated df for news:')
    print(df)
    
    print(today)
    print(df_old_news['date'].iloc[-1])
    if str(today) == str(df_old_news['date'].iloc[-1]):
        print('Do not insert anything... because today already exists in the df')
    else:
        print('Adding the last row to the feature group:')
        news_sentiment_fg.insert(df.tail(1), write_options={"wait_for_job" : False})
    
    # ----------------------------------------------------------------------------------------
    # Stock price:
    # ----------------------------------------------------------------------------------------
    
    # Get the current date and the date one month ago
    start_date= (today).strftime("%Y-%m-%d") 
    end_date = (today+timedelta(days=1)).strftime("%Y-%m-%d") 
    print('start_date, end_date = {}, {}'.format(start_date, end_date))

    try:
        # Get the TSLA stock data from yfinance
        tsla = yf.Ticker("TSLA") #VEFAB.ST

        # get historical market data
        data = tsla.history(start=start_date, end=end_date)
        tesla_df = data.drop(columns=['Dividends','Stock Splits'])

        # drop some columns
        tesla_df.index = tesla_df.index.strftime('%Y-%m-%d')

        # move "date"-index into its own column
        tesla_df = tesla_df.reset_index()

    except KeyError:
        
        # Add nan-values to the non-business days
        tesla_df = pd.DataFrame({'Date':[end_date],
                                 'open':[np.nan],
                                 'high':[np.nan],
                                 'low':[np.nan],
                                 'close':[np.nan],
                                 'volume':[np.nan]})

    # Rename column 'Date' to 'date'
    tesla_df['Date'] = start_date
    tesla_df.columns = tesla_df.columns.str.lower()
    
    # Fetch existing feature group
    tesla_fg = fs.get_feature_group(name="tsla_stock",version=1)
    
    fg_query = tesla_fg.select_all()
    df_old_stock = fg_query.read() 
    df_old_stock = df_old_stock.sort_values("date")
    df_old_stock = df_old_stock.reset_index(drop=True)
    print('Existing TSLA stock feature group:')
    print(df_old_stock)
    
    tesla_df = pd.concat([df_old_stock, tesla_df])
    tesla_df = tesla_df.reset_index(drop=True)
    print('Updated df for TSLA stock:')
    print(tesla_df)
    
    print('Adding the last row to the feature group:')
    tesla_fg.insert(tesla_df.tail(1), write_options={"wait_for_job" : False})
     
    # ----------------------------------------------------------------------------------------
        
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
    