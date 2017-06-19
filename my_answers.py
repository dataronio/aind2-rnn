import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [[series[i+j] for j in range(0, window_size)] for i in range(len(series)-window_size)]
    y = [[series[i]] for i in range(window_size, len(series))]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(6, input_shape=(window_size, 1)))
    model.add(Dense(1))  # default linear output ie. regression

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    chars = ''.join(set(text))
    print(chars)
    # remove as many non-english characters and character sequences as you can
    text = text.replace('\u00e9', 'e')
    text = text.replace('\u00e8', 'e')
    text = text.replace('\u00e2', 'a')
    text = text.replace('\u00e0', 'a')
    text = text.replace('$', ' ')
    text = text.replace('%', ' ')
    text = text.replace('/', ' ')
    text = text.replace('*', ' ')
    text = text.replace('-', ' ')
    text = text.replace('&', ' ')
    text = text.replace('@', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('"', ' ')
    text = text.replace("'", ' ')

    text = text.replace('0', ' ')
    text = text.replace('1', ' ')
    text = text.replace('2', ' ')
    text = text.replace('3', ' ')
    text = text.replace('4', ' ')
    text = text.replace('5', ' ')
    text = text.replace('6', ' ')
    text = text.replace('7', ' ')
    text = text.replace('8', ' ')
    text = text.replace('9', ' ')


### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i+window_size] for i in range(0, len(text)-window_size, step_size)]
    outputs = [text[i] for i in range(window_size, len(text), step_size)]
    return inputs, outputs
