import os
import modal
import hopsworks
import pandas as pd
from feature_preprocessing import extract_business_day
import numpy as np

def fix_data_from_feature_view(df,start_date,end_date):
    df = df.sort_values("date")
    df = df.reset_index()
    df = df.drop(columns=["index"])

    # Create a boolean mask for rows that fall within the date range
    mask = (pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))
    len_df = np.shape(df)
    df = df[mask] # Use the boolean mask to filter the DataFrame
    print('From shape {} to {} after cropping to given date range: {} to {}'.format(len_df,np.shape(df),start_date,end_date))

    # Get rid off all non-business days
    isBusinessDay, is_open = extract_business_day(start_date,end_date)
    is_open = [not i for i in is_open] # Invert the mask to be able to drop all non-buisiness days

    filtered_df = df.drop(df[is_open].index) # Use the mask to filter the rows of the DataFrame
    print('From shape {} to {} after removing non-business days'.format(np.shape(df),np.shape(filtered_df)))
    print(filtered_df)
    
    return filtered_df

if __name__ == "__main__":
    
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get stock TSLA feature group
    tesla_fg = fs.get_feature_group(
        name='tsla_stock',
        version=1
    )

    # Get news feature group
    news_sentiment_fg = fs.get_feature_group(
        name='news_sentiment',
        version=1
    )

    # Create feature view 

    # Query Preparation
    fg_query = tesla_fg.select_except(["open","low","high","volume"]).join(news_sentiment_fg.select_except(['polarity']))

    df = fg_query.read()

    # Remove non-business days etc:
    filtered_df = fix_data_from_feature_view(df,start_date='2015-07-16',end_date = '2023-01-04')

    # Columns to apply transformation function on:
    columns_to_transform = filtered_df.columns
    columns_to_transform = columns_to_transform.tolist()
    columns_to_transform.remove("date")

    # Map features to transformation functions.
    transformation_functions = {col: fs.get_transformation_function(name="min_max_scaler") for col in columns_to_transform}

    feature_view = fs.create_feature_view(
        name='stock_pred_modal',
        version=1,
        description="Join TSLA stock prices with news sentiment data",
        transformation_functions=transformation_functions,
        query=fg_query
    )