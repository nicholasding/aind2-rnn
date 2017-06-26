import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import math

# fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1))) # layer 1 - LSTM layer with 5 hidden units
    model.add(Dense(1)) # layer 2 - dense layer with one unit

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


### list all unique characters in the text and remove any non-english ones
def clean_text(text):
    chars = {k:1 for k in text}

    print('Unique Characters', chars.keys())

    # Valid English/proper punctuation characters
    valid_chars = ['i', 's', ' ', 'e', 'y', 'h', 'c', 'l', 'p', 'a', 'n', 'd', 'r', 'o', 'm', 't', 'w', 'f', 'x', '.', 'k', 'v', ',', 'u', 'b', 'g', '-', "'", 'j', 'q', 'z', ';', '"', '!', '?']

    # remove as many non-english characters and character sequences as you can 
    chars = list(text)
    for idx, c in enumerate(chars):
        if c not in valid_chars:
            chars[idx] = ' '

    text = ''.join(chars)
    
    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    return text


### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(math.ceil((len(text) - window_size) / step_size)):
        inputs.append(text[i * step_size : i * step_size + window_size])
        outputs.append(text[i * step_size + window_size])
    
    return inputs, outputs
