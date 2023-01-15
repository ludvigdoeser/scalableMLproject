import os
import modal
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group for historical stock (TSLA) data
tesla_df = pd.read_csv("https://raw.githubusercontent.com/ludvigdoeser/scalableMLproject/ludvig/data/stock/tesla_2015-07-16-2023-01-05.csv")
 
tesla_fg = fs.get_or_create_feature_group(
    name="tsla_stock",
    description="Tesla stock dataset from yfinance",
    version=1,
    primary_key=["date"],
    online_enabled=True,
    )

tesla_fg.insert(tesla_df, write_options={"wait_for_job" : False})

# Create feature group for historical news data
news_df = pd.read_csv('https://raw.githubusercontent.com/ludvigdoeser/scalableMLproject/ludvig/data/news/training_data_exp_mean_7_days.csv')

news_sentiment_fg = fs.get_or_create_feature_group(
    name='news_sentiment',
    description='News sentiment from EOD Historical Data',
    version=1,
    primary_key=['date'],
    online_enabled=True,
)

news_sentiment_fg.insert(news_df)