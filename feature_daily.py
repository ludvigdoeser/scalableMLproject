import os
import modal
from feature_preprocessing import getNews, pre_process_news

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    packages = ["hopsworks","pandas","numpy","tensorflow","pandas_market_calendars","joblib","scikit-learn","yfinance"]
    hopsworks_image = modal.Image.debian_slim().pip_install(packages)
    
    # schedule = modal.Period(days=1)
    @stub.function(image=hopsworks_image, secret=modal.Secret.from_name("scalableML"))
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
    
    # NEWS:
    
    today = date.today()
    print("Today's date:", today)
    
    # Set the API endpoint and your API key
    endpoint = "https://eodhistoricaldata.com/api/news"
    api_key = "63b2efc1180c66.27144598"

    # Set the ticker symbol
    ticker = "TSLA.US" #TSLA
    
    #Create a Pandas dataframe from the response
    news = getNews(api_key,endpoint,ticker,today,today,num=100)
    df = pre_process_news(news) 
    print('df = ',df)
    
    # Add new row to feature group
    """
    news_sentiment_fg = fs.get_feature_group(name="news_sentiment",version=1)
    
    fg_query = news_sentiment_fg.select_all()
    df_old_news = fg_query.read() 
    print('df_old_news = ',df_old_news)
    
    #df = exponential_moving_average(df_processed, window=7)
       
    news_sentiment_fg.insert(df_last_row, write_options={"wait_for_job" : False})
    """
    
    # Stock price:
    
    # Get the current date and the date one month ago
    start_date= (today-timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (today).strftime("%Y-%m-%d")
    print('start_date = {}'.format(start_date))
    print('end_date = {}'.format(end_date))

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
                                'Open':[np.nan],
                                 'High':[np.nan],
                                 'Low':[np.nan],
                                 'Close':[np.nan],
                                 'Volume':[np.nan]})

    # Rename column 'Date' to 'date'
    tesla_df = tesla_df.rename(columns={'Date': 'date'})
    print('Final size of dataframe',np.shape(tesla_df))
    print(tesla_df)
    print('\n')
     
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
    