from calendar import monthrange
from feature_preprocessing import *
import glob
import pandas as pd

# Set the API endpoint and your API key
endpoint = "https://eodhistoricaldata.com/api/news"
api_key = "63b2efc1180c66.27144598"

# Set the ticker symbol
ticker = "TSLA.US" #TSLA

def getNews_historical(api_key,endpoint,ticker,year,month,num=1000):
  
    for start,end in zip([1,15],[16,monthrange(year, month)[1]]):
    
        from_date = '{}-{:02d}-{:02d}'.format(year,month,start)
        to_date = '{}-{:02d}-{:02d}'.format(year,month,end)
        
        print('Grabbing News data between {}-{}'.format(from_date,to_date))    
        news = getNews(api_key,endpoint,ticker,from_date,to_date)
        
        print('Number of articles: ',len(news.index))
        news.head(n=num)

        # Store the dataframe as a CSV file
        news.to_csv("data/news/tesla_from_{}_to_{}.csv".format(from_date,to_date))

# Grab old data
for year in range(2013,2024):
    for month in range(1,13):
        getNews_historical(api_key,ticker,year,month)
        if year == 2023 and month == 1:
            break

# List the CSV files to merge
csv_files = glob.glob('data/news/tesla_*.csv')

# Read the CSV files into a list of dataframes
dataframes = [pd.read_csv(f) for f in csv_files]

# Merge the dataframes into a single dataframe
df = pd.concat(dataframes)

df = df.sort_values("date")

df.reset_index()

# Write the merged dataframe to a CSV file
df.to_csv("data/news/tesla_2013-2023.csv", index=False)