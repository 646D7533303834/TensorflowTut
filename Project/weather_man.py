import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from tensorflow.python.framework import dtypes

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - (2*time_steps)):
        if labels:
            if data.HOURLYPrecip[(i+time_steps):(i+(2*time_steps))].sum() > 0.0:
                rnn_df.append([1.0, 0.0])
            else:
                rnn_df.append([0.0, 1.0])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)

def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def prepare_data(data, time_steps, labels=False, val_size=0.15, test_size=0.15):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def load_csvdata(rawdata, time_steps, seperate=False):
    data = rawdata
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)
"""
def lstm_model(time_steps, rnn_layers, dense_layers=None):
    """"""
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param time_steps: the number of time steps the model will be looking at.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """"""

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['steps'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['steps'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return learn.ops.dnn(input_layers,
                                 layers['layers'],
                                 activation=layers.get('activation'),
                                 dropout=layers.get('dropout'))
        elif layers:
            return learn.ops.dnn(input_layers, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = learn.ops.split_squeeze(1, time_steps, X)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        return learn.models.linear_regression(output, y)

    return _lstm_model
"""
"""
lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(RNN_LAYERS),tf.contrib.rnn.BasicLSTMCell(RNN_LAYERS)])

regressor = tf.contrib.learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                                                 n_classes = 0,
                                                 verbose = 1,
                                                 steps=TRAINING_STEPS,
                                                 optimizer='Adagrad',
                                                 learning_rate=0.03,
                                                 batch_size=BATCH_SIZE)

validation_monitor = tf.contrib.learn.ValidationMonitor(X['val'], y['val'],
                                                        every_n_steps=PRINT_STEPS,
                                                        early_stopping_rounds = 1000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)

predicted = regressor.predict(X['test'])
mse = tf.metrics.mean_absolute_error(y['test'], predicted)
print ("Error: %f" % mse)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label ='test')
plt.legend(handles=[plot_predicted, plot_test])

print(df_inputs.head())
"""

def main():

    file_path = 'data/'
    file_name = '1114147.csv'
    df_raw_data = pd.read_csv(file_path + file_name, encoding="ISO-8859-1", low_memory=False)

    df_inputs = df_raw_data.iloc[:, 2:]
    df_inputs.DATE = pd.to_datetime(df_inputs.DATE)

    df_inputs['year'] = df_inputs.DATE.dt.year
    df_inputs['month'] = df_inputs.DATE.dt.month
    df_inputs['day'] = df_inputs.DATE.dt.day
    df_inputs['hour'] = df_inputs.DATE.dt.hour

    df_inputs.HOURLYWindGustSpeed = df_inputs.HOURLYWindGustSpeed.fillna(0)
    df_inputs.HOURLYPrecip = df_inputs.HOURLYPrecip.fillna(0)

    # print(df_inputs.HOURLYWindGustSpeed)

    df_inputs = df_inputs[[
        'year',
        'month',
        'day',
        'hour',
        'HOURLYWindDirection',
        'HOURLYWindGustSpeed',
        'HOURLYVISIBILITY',
        'HOURLYDRYBULBTEMPF',
        'HOURLYWETBULBTEMPF',
        'HOURLYDewPointTempF',
        'HOURLYRelativeHumidity',
        'HOURLYAltimeterSetting',
        'HOURLYSeaLevelPressure',
        'HOURLYPrecip'
    ]]

    df_inputs = df_inputs.dropna()

    # df_inputs = df_inputs.infer_objects()
    # df_inputs = df_inputs.to_numeric()
    df_inputs.HOURLYWindDirection.replace('VRB', -1, inplace=True)
    df_inputs.HOURLYVISIBILITY.replace(['V', 's'], '', regex=True, inplace=True)
    df_inputs.HOURLYDRYBULBTEMPF.replace(['V', 's'], '', regex=True, inplace=True)
    df_inputs.HOURLYDewPointTempF.replace(['V', 's'], '', regex=True, inplace=True)
    df_inputs.HOURLYAltimeterSetting.replace(['V', 's'], '', regex=True, inplace=True)
    df_inputs.HOURLYSeaLevelPressure.replace(['V', 's'], '', regex=True, inplace=True)
    df_inputs.HOURLYPrecip.replace(['T', 's'], [0.001, ''], regex=True, inplace=True)
    df_inputs = df_inputs.apply(pd.to_numeric)
    df_inputs = df_inputs.astype('float32')

    LOG_DIR = './test_logs'
    TIMESTEPS = 24
    RNN_LAYERS = [{'steps': TIMESTEPS}]
    DENSE_LAYERS = [10, 10]
    TRAINING_STEPS = 100000
    BATCH_SIZE = 100
    PRINT_STEPS = TRAINING_STEPS / 100

    X, y = load_csvdata(df_inputs,
                        TIMESTEPS,
                        seperate=False)

    #classifier = learn.DynamicRnnEstimator