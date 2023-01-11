# scalableMLproject
Project in Advanced Scalable Machine Learning Course at KTH

# Requirements for running the code

Download the repo, create a new conda environment and then run:

```
pip install -r requirements.txt
```

For the interested reader, one can create a pip3 compatible `requirements.txt` file using:

```
pip3 freeze > requirements.txt  # Python3
```

# Deploy a model at modal

modal deploy --name get_tesla_news_and_stock feature_daily.py

(scalableML) ludo@Ludvigs-MBP scalableMLproject % modal deploy --name get_tesla_news_and_stock feature_daily.py
✓ Initialized. View app at https://modal.com/apps/ap-meNRCmGr0cgl2zNoDoZXbZ
✓ Created objects.
├── 🔨 Created f.
├── 🔨 Mounted feature_daily.py at /root
├── 🔨 Mounted /Users/ludo/Documents/PhD/Courses:Cluster/ScalableMachineLearning/2022/scalableMLproject/feature_daily.py at /root/.
└── 🔨 Mounted /Users/ludo/Documents/PhD/Courses:Cluster/ScalableMachineLearning/2022/scalableMLproject/feature_preprocessing.py at /root/.
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/ludvigdoeser/get_tesla_news_and_stock
