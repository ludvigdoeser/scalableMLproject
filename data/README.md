# Historical data

The raw data for our stock price prediction project consists mainly of the following two parts:

* the historical TSLA stock prices, and
* the News Sentiment data about Tesla, Inc.,

tracing from 2015-07-16 to 2023-01-04.

## news
* `tesla_2013-2023.csv` contains all news data from EOD between 2013-01-01 and 2023-01-04.
* `training_data_exp_mean_7_days.csv` contains the processed data to be used for training between 2015-07-16 to 2023-01-04.

## stock
* `tesla_2015-07-16-2023-01-05.csv` contains all historical stock data for tesla, including NaNs for non-business days (to enable merging with the news sentiment data)
