import tensorflow as tf
import pandas as pd
import numpy as np
import project_lib as pl

FILE_PATH = 'data/'
FILE_NAME = '1114147.csv'
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


def main():

    df_raw = pl.get_weather_data(FILE_PATH, FILE_NAME)
    df_inputs = pl.format_weather_data(df_raw)

    train_y, val_y, test_y = pl.prepare_data(df_inputs['HOURLYPrecip'].values, TIME_STEPS, labels=True)
    train_x, val_x, test_x = pl.prepare_data(df_inputs, TIME_STEPS)

    print(train_x.shape)
    print(train_y.shape)

    """
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        iter = 1
        while iter < 800:

            # Add optimizer and dict to run method
            sess.run()
    """


if __name__ == "__main__":
    main()
