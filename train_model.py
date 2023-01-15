import os
import modal
from create_feature_view import fix_data_from_feature_view

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    packages = ["hopsworks","pandas","numpy","tensorflow","pandas_market_calendars","joblib","scikit-learn"]
    hopsworks_image = modal.Image.debian_slim().pip_install(packages)
    @stub.function(image=hopsworks_image, secret=modal.Secret.from_name("scalableML"))
    def f():
        g()

def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    from sklearn.model_selection import train_test_split

    import pandas as pd
    import numpy as np
    from datetime import date
    import joblib
    import inspect 
   
    # Tensorflow
    import tensorflow
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding

    today = date.today()
    print("Today's date:", today)

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

    ## ------------------------------------------------------------------------------------------------------------------------

    # ------------------------ Create training data -----------------------------: 
    
    print('Fetching feature view from hopsworks...')
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get feature view 
    fv = fs.get_feature_view(
        name = 'stock_pred_modal',
        version = 1
    )

    # Get dataframe of training data from feature view
    df, _ = fv.training_data()

    # Remove non-business days etc:
    filtered_df = fix_data_from_feature_view(df,start_date='2015-07-16',end_date='2023-01-04')

    # Data for training (already scaled):
    print('Prepare training data...')
    data = filtered_df[['exp_mean_7_days','close']].to_numpy()

    X,y = [],[]
    timelag = 7 #days

    for i in range(timelag,(len(data)-1)):
        X.append(data[i-timelag:i])
        y.append(data[i+1,1])
        
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(np.array(X),np.array(y), test_size=0.1, random_state=42)

    # ------------------------ Create model and train it -----------------------------: 
    
    ### Hyperparam keras tuner resulted in:

    # filters: 128
    # dropout: 0.01
    # recurrent_dropout: 0.1
    # dense_dropout: 0.01
    # activation: leaky_relu
    # depth: 1
    # Score: 0.001255395240150392

    print('Create model...')
    model = create_model(LSTM_filters=128,
                     dropout=0.01,
                     recurrent_dropout=0.1,
                     dense_dropout=0.01,
                     activation='leaky_relu',
                     depth=1,
                     input_channels=2)

    # Early stopping
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
    
    # The contents of the 'stock_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="stock_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    
    # Save losses
    csv_logger = keras.callbacks.CSVLogger(model_dir + '/log_{}.csv'.format(today), separator=",", append=False)
    
    # Train the model
    print('Train model...')
    history = model.fit(X_train, y_train, 
                        epochs=500, 
                        batch_size=256, 
                        validation_split=0.1,
                        callbacks=[es, csv_logger],
                        verbose=1)
    
    print('Done...')
    print('History val loss: ',history.history['val_loss'])
    
    # ------------------------ Predictions -----------------------------: 
    
    y_pred = model.predict(X_test)
    
    # Need the inverse transformation:
    td_transformation_functions = fv._transformation_functions 
    td_transformation_function = td_transformation_functions['close']
    sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
    param_dict = dict([(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
    
    # Perform the transformation 
    df_temp = pd.DataFrame(y_pred,columns=['close'])
    df_pred = df_temp['close'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"])
    
    df_temp = pd.DataFrame(y_test,columns=['close'])
    df_test = df_temp['close'].map(lambda x: x*(param_dict["max_value"]-param_dict["min_value"])+param_dict["min_value"])
    
    # Compute rsme:
    rsme = np.sqrt(np.mean(df_pred.to_numpy()-df_test.to_numpy())**2 )
   
    # -----------------------------------------------------------------
    
    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()
    
    # Save our model to 'model_dir', whose contents will be uploaded to the model registry
    #export_path = model_dir+"/stock_pred_model"
    print('Exporting trained model to: {}'.format(model_dir))
    tensorflow.saved_model.save(model, model_dir)
    
    print('The model directory now contains: ',os.listdir(model_dir))
    
    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)
     
    # Create an entry in the model registry that includes the model's name, desc, metrics    
    stock_pred_model = mr.tensorflow.create_model(
        name="stock_pred_model",
        metrics={"rmse" : rsme},
        model_schema=model_schema,
        description="Stock Market TSLA Predictor from News Sentiment",
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    stock_pred_model.save(model_dir)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()