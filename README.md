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

For the batch_inference: https://modal.com/apps/ludvigdoeser/pred_tesla_stock_tomorrow 

```python
def create_model(LSTM_filters=64,
                 dropout=0.1,
                 recurrent_dropout=0.1,
                 dense_dropout=0.5,
                 activation='relu',
                 depth=1,
                 input_channels=1):

    model = Sequential()
    
    if depth>1:
        for i in range(1,depth):
            # Recurrent layer
            model.add(LSTM(LSTM_filters, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout,input_shape=(timelag, input_channels)))

    # Recurrent layer
    model.add(LSTM(LSTM_filters, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout, input_shape=(timelag, input_channels)))
    
    # Fully connected layer
    if activation=='relu':
        model.add(Dense(LSTM_filters, activation='relu'))
    elif activation=='leaky_relu':
        model.add(Dense(LSTM_filters))
        model.add(keras.layers.LeakyReLU(alpha=0.1))

    # Dropout for regularization
    model.add(Dropout(dense_dropout))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model

model = create_model(activation='relu',input_channels=2,depth=2)
```

