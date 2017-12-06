import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from tensorflow.python.framework import dtypes

LOG_DIR = './test_logs'
TIMESTEPS = 24
FEATURE_COUNT = 14
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 60000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100
OUTPUT_SIZE = 2
LEARNING_RATE = 0.01

"""
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
"""


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

#lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(RNN_LAYERS),tf.contrib.rnn.BasicLSTMCell(RNN_LAYERS)])

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
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

    train_x, val_x, test_x = prepare_data(df_inputs, TIMESTEPS)
    train_y, val_y, test_y = prepare_data(df_inputs, TIMESTEPS, labels=True)

    print(train_y.shape)

    ##n_)hidden = len(train_x[0])
    ##n_output = len(train_y[0]

    x = tf.placeholder("float", [None, TIMESTEPS, FEATURE_COUNT])
    y = tf.placeholder("float", [None, OUTPUT_SIZE])

    # Define an lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(FEATURE_COUNT)

    x = tf.unstack(x, TIMESTEPS, 1)
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([FEATURE_COUNT, OUTPUT_SIZE]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([OUTPUT_SIZE]))
    }

    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
    #                                            sequence_length=seqlen)

    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    batch_size = tf.shape(outputs)[0]

    index = tf.range(0, batch_size) * TIMESTEPS + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, FEATURE_COUNT]), index)

    pred = tf.matmul(outputs, weights['out']) + biases['out']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(1, TRAINING_STEPS + 1):
            batch_x = train_x[step]
            batch_y = train_y[step]
            batch_seqlen = np.empty(BATCH_SIZE)
            batch_seqlen.fill(TIMESTEPS)
            print(batch_x.shape)



            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y})
            if step % PRINT_STEPS == 0 or step == 1:
                acc, loss = sess.run([accuracy, cost],
                                     feed_dict={x: batch_x,
                                                y: batch_y})
                print("Step " + str(step*batch_size) + ", Minibatch Loss = " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print("Optimization Finished!")

        test_data = test_x
        test_label = test_y
        test_seqlen = np.array(test_x.shape[0])
        test_seqlen.fill(TIMESTEPS)
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: test_data,
                                            y: test_label}))

    """
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }

    cell = tf.nn.rnn_cell.LSTMCell(num_units=OUTPUT_SIZE, state_is_tuple=True)
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=X_lengths,
        inputs=X
    )

    result = tf.contrib.learn.run_n()
    """


if __name__ == "__main__":
    main()
