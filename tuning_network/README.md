# Hyperparameter Optimization

The result of the `keras-tuner` applied on our model (see readme in root directory). The following optimization scheme with BayesianOptimization was run:

```python 
def tune_model(hp):
    
    model = create_model(LSTM_filters=hp.Choice('filters', [32,64,128]),
                         dropout=hp.Choice('dropout', [0.01,0.1,0.3]),
                         recurrent_dropout=hp.Choice('recurrent_dropout', [0.01,0.1,0.3]),
                         dense_dropout=hp.Choice('dense_dropout', [0.01,0.1,0.3]),
                         activation=hp.Choice('activation', ['relu','leaky_relu']),
                         depth=hp.Choice('depth', [1,2]),
                         input_channels=2)
    
    return model 

"""
tuner_search = RandomSearch(tune_model,
                            objective='val_loss',
                            max_trials=5,
                            directory='.',
                            project_name="hyperparam_optimization")
"""

tuner_search = BayesianOptimization(
    hypermodel=tune_model,
    objective='val_loss',
    max_trials=20,
    num_initial_points=4,
    alpha=0.0001,
    beta=2.6,
    seed=42,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    directory='tuning_network',
    project_name="Hyperparam_BayesianOptimization",
    overwrite=True,
)

# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)

cp = [es]

tuner_search.search(x_train,y_train,epochs=50,validation_split=0.1,callbacks=cp)
```
