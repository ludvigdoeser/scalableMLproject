# scalableMLproject
Project in Advanced Scalable Machine Learning Course at KTH

* Predict: stock market
* Data sources: 
- News sentiment in English: https://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms
- News headlines: https://newsapi.org/docs/endpoints/top-headlines
- Nasdaq stock market: https://data.nasdaq.com/tools/python 
* Group number: 26

![Schematic of Project Idea](Schematic.png)

## Literature

* https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571 
* https://stocksnips.net/learn/news-based-stock-sentiment/ 
* https://towardsdatascience.com/sentiment-analysis-on-news-headlines-classic-supervised-learning-vs-deep-learning-approach-831ac698e276 

## Tutorials

* https://github.com/logicalclocks/hopsworks-tutorials/tree/master/advanced_tutorials
* In particular, this one: https://github.com/logicalclocks/hopsworks-tutorials/tree/master/advanced_tutorials/bitcoin 
* https://github.com/featurestoreorg/serverless-ml-course

## Notes

* Need to create API keys/secrets at modal. Create all API keys in same secret
* Backfill feature pipeline gets historical data
* Wait_for_job=False recommendation from Jim when it comes to uploading to feature group?
* Add some great_expectations or something that looks at the data and disregards it if it is not what you expected
