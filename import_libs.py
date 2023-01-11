# Classic
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import glob
import json
import inspect 

# Dates related 
from calendar import monthrange
import pandas_market_calendars as mcal
import datetime
from datetime import date, timedelta

# Finance 
import yfinance as yf

# Tensorflow
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding

from kerastuner import RandomSearch
from kerastuner import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters