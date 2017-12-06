import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import normalizer as norm

LOG_DIR = './test_logs'
# unrolled through 24 time steps
TIME_STEPS = 24
# number of inputs
FEATURE_COUNT = 12
RNN_LAYERS = [{'steps': TIME_STEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 60000
BATCH_SIZE = 100
# hidden LSTM units
NUM_UNITS = 100
PRINT_STEPS = TRAINING_STEPS / 100
OUTPUT_SIZE = 2
# learning rate for adam
LEARNING_RATE = 0.001
# classes
N_CLASSES = 400

# id_matrix = np.eye(N_CLASSES)


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
    for i in range(len(data) - time_steps - 1):
        if labels:
            data_ = data.iloc[i + time_steps + 1].as_matrix()
            # data_ = data.iloc[(i + 1):(i + time_steps + 1)].as_matrix()
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()

        # rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
        rnn_df.append(data_)
    return np.array(rnn_df)


def fcnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    fcnn_df = []
    for i in range(len(data) - time_steps - 1):
        if labels:
            data_ = data.iloc[i + time_steps + 1].as_matrix()
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
        fcnn_df.append(data_)
    
    return np.array(fcnn_df.values)


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


def main():
    file_path = 'data/'
    file_name = '1114147.csv'
    df_raw_data = pd.read_csv(file_path + file_name, encoding="ISO-8859-1", low_memory=False)

    df_inputs = df_raw_data.iloc[:, 2:]
    df_inputs.DATE = pd.to_datetime(df_inputs.DATE)

    df_inputs['year'] = df_inputs.DATE.dt.year
    df_inputs['month'] = df_inputs.DATE.dt.month
    df_inputs['day'] = df_inputs.DATE.dt.dayofyear
    df_inputs['hour'] = df_inputs.DATE.dt.hour

    df_inputs.HOURLYWindGustSpeed = df_inputs.HOURLYWindGustSpeed.fillna(0)
    df_inputs.HOURLYPrecip = df_inputs.HOURLYPrecip.fillna(0)

    # print(df_inputs.HOURLYWindGustSpeed)

    df_inputs = df_inputs[[
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

    df_inputs.hour, hour_min, hour_diff = norm.generic_normalize(df_inputs.hour)
    df_inputs.day, day_min, day_diff = norm.generic_normalize(df_inputs.day)
    df_inputs.HOURLYWindDirection, HOURLYWindDirection_min, HOURLYWindDirection_diff = \
        norm.generic_normalize(df_inputs.HOURLYWindDirection)
    df_inputs.HOURLYWindGustSpeed, HOURLYWindGustSpeed_min, HOURLYWindGustSpeed_diff = \
        norm.generic_normalize(df_inputs.HOURLYWindGustSpeed)
    df_inputs.HOURLYVISIBILITY, HOURLYVISIBILITY_min, HOURLYVISIBILITY_diff = \
        norm.generic_normalize(df_inputs.HOURLYVISIBILITY)
    df_inputs.HOURLYDRYBULBTEMPF, HOURLYDRYBULBTEMPF_min, HOURLYDRYBULBTEMPF_diff = \
        norm.generic_normalize(df_inputs.HOURLYDRYBULBTEMPF)
    df_inputs.HOURLYWETBULBTEMPF, HOURLYWETBULBTEMPF_min, HOURLYWETBULBTEMPF_diff = \
        norm.generic_normalize(df_inputs.HOURLYWETBULBTEMPF)
    df_inputs.HOURLYDewPointTempF, HOURLYDewPointTempF_min, HOURLYDewPointTempF_diff = \
        norm.generic_normalize(df_inputs.HOURLYDewPointTempF)
    df_inputs.HOURLYRelativeHumidity, HOURLYRelativeHumidity_min, HOURLYRelativeHumidity_diff = \
        norm.generic_normalize(df_inputs.HOURLYRelativeHumidity)
    df_inputs.HOURLYAltimeterSetting, HOURLYAltimeterSetting_min, HOURLYAltimeterSetting_diff = \
        norm.generic_normalize(df_inputs.HOURLYAltimeterSetting)
    df_inputs.HOURLYSeaLevelPressure, HOURLYSeaLevelPressure_min, HOURLYSeaLevelPressure_diff = \
        norm.generic_normalize(df_inputs.HOURLYSeaLevelPressure)
    df_inputs.HOURLYPrecip, HOURLYPrecip_min, HOURLYPrecip_diff = \
        norm.generic_normalize(df_inputs.HOURLYPrecip)

    print(df_inputs.shape)

    train_x, val_x, test_x = prepare_data(df_inputs, TIME_STEPS)
    train_y, val_y, test_y = prepare_data(df_inputs, TIME_STEPS, labels=True)

    print(train_x.shape)
    print(train_y.shape)
    print(len(train_x))
    print(len(train_y))

    # weights and biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([NUM_UNITS, FEATURE_COUNT]))
    out_bias = tf.Variable(tf.random_normal([FEATURE_COUNT]))

    # defining placeholders
    # input image placeholder
    x = tf.placeholder("float", [None, TIME_STEPS, FEATURE_COUNT])
    # input label placeholder
    y = tf.placeholder("float", [None, FEATURE_COUNT])

    # processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
    input = tf.unstack(x, TIME_STEPS, 1)

    lstm_layer=rnn.BasicLSTMCell(NUM_UNITS, forget_bias=1)
    #lstm_layer2=rnn.BasicLSTMCell(NUM_UNITS, forget_bias=1)
    #multi_cell=rnn.MultiRNNCell([lstm_layer1, lstm_layer2])
    outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    prediction = tf.matmul(outputs[-1], out_weights) + out_bias

    # loss_function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimization
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variables
    init = tf.global_variables_initializer()

    tensor_capacity = len(train_x)
    train_x_tensor = tf.convert_to_tensor(train_x)
    train_y_tensor = tf.convert_to_tensor(train_y)

    c = np.c_[train_x.reshape(len(train_x), -1), train_y.reshape(len(train_y), -1)]
    x2 = c[:, :train_x.size//len(train_x)].reshape(train_x.shape)
    y2 = c[:, :train_y.size//len(train_y)].reshape(train_y.shape)

    with tf.Session() as sess:
        sess.run(init)
        iter = 1
        while iter < 800:
            batch_x, batch_y = tf.train.shuffle_batch(
                [train_x_tensor, train_y_tensor],
                batch_size=BATCH_SIZE,
                capacity=len(train_x),
                min_after_dequeue=10000,
                enqueue_many=True
            )

            #np.random.shuffle(c)
            batch_x = x2[:BATCH_SIZE]
            batch_y = y2[:BATCH_SIZE]

            #batch_x = batch_x.reshape((BATCH_SIZE, TIME_STEPS, FEATURE_COUNT))

            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

            if iter % 10 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print("For iter ", iter)
                print("Accuracy ", acc)
                print("Loss ", los)
                print("__________________")

            iter = iter + 1


if __name__ == "__main__":
    main()
