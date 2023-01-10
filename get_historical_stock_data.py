import datetime
import yfinance as yf
import pandas as pd
from feature_preprocessing import extract_business_day
import numpy as np

def create_tsla_history():

    start_date ='2015-07-16'
    end_date = '2023-01-05'

    start_date = datetime.datetime.strptime(start_date,"%Y-%m-%d") #datetime.date(2015, 7, 16)
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d") #datetime.date(2023, 1, 4)

    # Get the TSLA stock data from yfinance
    tsla = yf.Ticker("TSLA") #VEFAB.ST
    # info = tsla.info

    # get historical market data
    data = tsla.history(start=start_date, end=end_date)

    # drop some columns
    tesla_df = data.drop(columns=['Dividends','Stock Splits'])
    tesla_df.index = tesla_df.index.strftime('%Y-%m-%d')
    
    print('Number of business days included in data set: ',np.shape(tesla_df))

    # Create an array of all dates in the specified period
    all_dates = np.array([start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days)])
    all_dates = [d.strftime('%Y-%m-%d') for d in all_dates]

    # Use setdiff1d() to find the non-business days
    isBusinessDay, _ = extract_business_day(start_date='2015-07-16',end_date='2023-01-04')
    non_business_days = np.setdiff1d(all_dates, isBusinessDay)

    # Add nan-values to the non-business days
    print('Add {} non business days with NaN-values'.format(len(non_business_days)))
    for d in non_business_days:
        tesla_df.loc[d,:] = [np.nan,np.nan,np.nan,np.nan,np.nan]

    # sort index (dates)
    tesla_df = tesla_df.sort_index()
 
    # move "date"-index into its own column
    tesla_df = tesla_df.reset_index()
    
    # Rename column 'Date' to 'date'
    tesla_df = tesla_df.rename(columns={'Date': 'date'})
    print('Final size of dataframe',np.shape(tesla_df))
    
    # Write the merged dataframe to a CSV file
    start_date ='2015-07-16'
    end_date = '2023-01-05'
    save_path = "data/stock/tesla_{}-{}.csv".format(start_date,end_date)
    
    print('Save at :',save_path)
    tesla_df.to_csv(save_path, index=False)
    
    return tesla_df