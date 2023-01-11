import requests
import pandas as pd
import json
import pandas_market_calendars as mcal
import datetime
import numpy as np

# Stock market:
def extract_business_day(start_date,end_date):
    """
    Given a start_date and end_date.
    
    `Returns`:
    
    isBusinessDay: list of str (with all dates being business days)
    is_open: boolean list
        e.g is_open = [1,0,...,1] means that start_date = open, day after start_date = closed, and end_date = open
    """
    
    # Save for later
    end_date_save = end_date
    
    # Get the NYSE calendar
    cal = mcal.get_calendar('NYSE')

    # Get the NYSE calendar's open and close times for the specified period
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    
    # Only need a list of dates when it's open (not open and close times)
    isBusinessDay = np.array(schedule.market_open.dt.strftime('%Y-%m-%d')) 
    
    # Go over all days: 
    delta = datetime.timedelta(days=1)
    start_date = datetime.datetime.strptime(start_date,"%Y-%m-%d") #datetime.date(2015, 7, 16)
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d") #datetime.date(2023, 1, 4)
    
    # Extract days from the timedelta object
    num_days = (end_date - start_date).days + 1
    
    # Create boolean array for days being open (1) and closed (0) 
    is_open = np.zeros(num_days)
    
    # iterate over range of dates
    current_BusinessDay = isBusinessDay[0]
    count_dates = 0
    next_BusinessDay = 0
    
    while (start_date <= end_date):
    
        if start_date.strftime('%Y-%m-%d') == current_BusinessDay:
            is_open[count_dates] = True

            if current_BusinessDay == end_date_save:
                break
            else:
                next_BusinessDay += 1
                current_BusinessDay = isBusinessDay[next_BusinessDay]
        else:
            is_open[count_dates] = False

        count_dates += 1   
        start_date += delta
        
    return isBusinessDay, is_open

# News sentiment:
def getNews(api_key,endpoint,ticker,from_date,to_date,num=1000):
    # Set the parameters for the request
    params = {
        "api_token": api_key,
        "s": ticker,
        "from": from_date, 
        "to": to_date,
        "limit": num,
    }
    
    # Make the request to the API
    response = requests.get(endpoint, params=params)
    
    # Print the response from the API
    #print(response.json())

    #Return a Pandas dataframe from the response
    return pd.DataFrame(response.json())

def expand_sentiment(df,extract_string='polarity'):
    """
    # Debugging:
    sentiment_dict = []
    for i in range(0,num):
        try:
            sentiment_dict.append(json.loads(df['sentiment'][i].replace("'", '"')))
        except AttributeError:
            print(i)
    """
    num = len(df['sentiment'].index) 
    
    try: # if dictionary
        sentiment_dict = [df['sentiment'][i] for i in range(0,num)]
        polarity = [s[extract_string] for s in sentiment_dict]
    except: # if string
        print('Have to convert dict to string')
        sentiment_dict = [json.loads(df['sentiment'][i].replace("'", '"')) for i in range(0,num)]
        polarity = [s[extract_string] for s in sentiment_dict]

    return polarity

def remove_duplicates(df,p=False):
    # Keep the first occurrence of each set of duplicates
    
    if p:
        print(len(df.index))
    df = df.drop_duplicates(keep='first')
    if p:
        print(len(df.index))
    return df

def exponential_moving_average(df, window):
    df[f'exp_mean_{window}_days'] = df.polarity.ewm(span = window).mean()
    return df

def pre_process_news(df,resample=True):
    
    # First of all: Remove NaNs
    num_with_nan = len(df['sentiment'].index)
    df = df[df['sentiment'].notna()]
    df = df.reset_index()
    print('Removed {} news articles for which sentiment score is missing'.format(num_with_nan-len(df['sentiment'].index)))
    
    # Expand polarity... From sentiment dictionary into it's own column
    df['polarity'] = expand_sentiment(df)
    #df['neg'] = expand_sentiment(df,extract_string='neg')
    #df['neu'] = expand_sentiment(df,extract_string='neu')
    #df['pos'] = expand_sentiment(df,extract_string='pos')
    
    # Remove duplicates (only index will differ... have to drop some columns for duplicate-function to work properly)
    df = df.drop(columns=['index','symbols','tags'])
    if type(df['sentiment'][0]) is dict:
        df["sentiment"] = df["sentiment"].astype(str)
    df = remove_duplicates(df)
        
    # Set the index to the datetime column
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Get rid off some columns and set date to index
    df = df.drop(columns=['title','content','link','sentiment'])
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    if resample:
        num = len(df.index) 
        
        # Resample by taking mean over news segments:
        df = df.resample('1d').mean()
        print('Resampled from {} rows to {} rows'.format(num,len(df.index)))
    
    # Fill all NaN-values with zeros
    df = df.fillna(0)

    return df

def stock_2_fg():
    pass

def news_2_fg():
    pass 
    
    
    