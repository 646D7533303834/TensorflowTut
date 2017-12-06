import numpy as np
import pandas as pd
import tensorflow as tf


def generic_normalize(x, define_max=0):
    norm_min = x.min()
    if define_max == 0:
        val_max = x.max()
    else:
        val_max = define_max
    diff = val_max - norm_min

    return ((x - norm_min) / diff) - 0.5, norm_min, diff


def generic_denormalize(x, denorm_min, diff):
    return ((x + 0.5) * diff) + denorm_min


def get_weather_data(file_path, file_name):
    return pd.read_csv(file_path + file_name, encoding="ISO-8859-1", low_memory=False)


def format_weather_data(data):
    df_formatted_data = data.iloc[:, 2:]
    df_formatted_data.DATE = pd.to_datetime(df_formatted_data.DATE)

    df_formatted_data['year'] = df_formatted_data.DATE.dt.year
    df_formatted_data['month'] = df_formatted_data.DATE.dt.month
    df_formatted_data['day'] = df_formatted_data.DATE.dt.dayofyear
    df_formatted_data['hour'] = df_formatted_data.DATE.dt.hour

    df_formatted_data.HOURLYWindGustSpeed = df_formatted_data.HOURLYWindGustSpeed.fillna(0)
    df_formatted_data.HOURLYPrecip = df_formatted_data.HOURLYPrecip.fillna(0)

    df_formatted_data = df_formatted_data[[
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

    df_formatted_data = df_formatted_data.dropna()

    df_formatted_data.HOURLYWindDirection.replace('VRB', -1, inplace=True)
    df_formatted_data.HOURLYVISIBILITY.replace(['V', 's'], '', regex=True, inplace=True)
    df_formatted_data.HOURLYDRYBULBTEMPF.replace(['V', 's'], '', regex=True, inplace=True)
    df_formatted_data.HOURLYDewPointTempF.replace(['V', 's'], '', regex=True, inplace=True)
    df_formatted_data.HOURLYAltimeterSetting.replace(['V', 's'], '', regex=True, inplace=True)
    df_formatted_data.HOURLYSeaLevelPressure.replace(['V', 's'], '', regex=True, inplace=True)
    df_formatted_data.HOURLYPrecip.replace(['T', 's'], [0.001, ''], regex=True, inplace=True)
    df_formatted_data = df_formatted_data.apply(pd.to_numeric)
    df_formatted_data = df_formatted_data.astype('float32')

    return df_formatted_data


def normalize_weather_data(data):
    """
    Normalizes a pandas dataframe formatted by format_weather_data()
    :param data:
    :return:
    """
    df_norm_data = data
    df_norm_data.hour, hour_min, hour_diff = generic_normalize(df_norm_data.hour)
    df_norm_data.day, day_min, day_diff = generic_normalize(df_norm_data.day)
    df_norm_data.HOURLYWindDirection, HOURLYWindDirection_min, HOURLYWindDirection_diff = \
        generic_normalize(df_norm_data.HOURLYWindDirection)
    df_norm_data.HOURLYWindGustSpeed, HOURLYWindGustSpeed_min, HOURLYWindGustSpeed_diff = \
        generic_normalize(df_norm_data.HOURLYWindGustSpeed)
    df_norm_data.HOURLYVISIBILITY, HOURLYVISIBILITY_min, HOURLYVISIBILITY_diff = \
        generic_normalize(df_norm_data.HOURLYVISIBILITY)
    df_norm_data.HOURLYDRYBULBTEMPF, HOURLYDRYBULBTEMPF_min, HOURLYDRYBULBTEMPF_diff = \
        generic_normalize(df_norm_data.HOURLYDRYBULBTEMPF)
    df_norm_data.HOURLYWETBULBTEMPF, HOURLYWETBULBTEMPF_min, HOURLYWETBULBTEMPF_diff = \
        generic_normalize(df_norm_data.HOURLYWETBULBTEMPF)
    df_norm_data.HOURLYDewPointTempF, HOURLYDewPointTempF_min, HOURLYDewPointTempF_diff = \
        generic_normalize(df_norm_data.HOURLYDewPointTempF)
    df_norm_data.HOURLYRelativeHumidity, HOURLYRelativeHumidity_min, HOURLYRelativeHumidity_diff = \
        generic_normalize(df_norm_data.HOURLYRelativeHumidity)
    df_norm_data.HOURLYAltimeterSetting, HOURLYAltimeterSetting_min, HOURLYAltimeterSetting_diff = \
        generic_normalize(df_norm_data.HOURLYAltimeterSetting)
    df_norm_data.HOURLYSeaLevelPressure, HOURLYSeaLevelPressure_min, HOURLYSeaLevelPressure_diff = \
        generic_normalize(df_norm_data.HOURLYSeaLevelPressure)
    df_norm_data.HOURLYPrecip, HOURLYPrecip_min, HOURLYPrecip_diff = \
        generic_normalize(df_norm_data.HOURLYPrecip)

    return df_norm_data


def fcnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
        f = 7.2332; str(f)[::-1].find('.')
    """
    """degrees = 0
    power = 0
    if labels:
        degrees = data.max() - data.min()
        power = str(degrees)[::-1].find('.')
        print('Power: ', power)
        degrees = int(degrees * (10 ** power))
        print('Degrees: ', degrees)
        idx = (data * (10 ** power)).astype(int)
        print('idx: ', idx)
        onehot_lbl = tf.one_hot(idx, degrees)
    """
    fcnn_df = []
    for i in range(len(data) - time_steps - 1):
        if labels:
            # idx = int(generic_denormalize(data.HOURLYPrecip[i+time_steps+1], p_min, p_diff)*100)
            # print(idx)
            # data_ = id_matrix[idx]
            data_ = data[i + time_steps]
            # data_ = data.iloc[(i + 1):(i + time_steps + 1)].as_matrix()
        else:
            norm_data = normalize_weather_data(data)
            data_ = norm_data.iloc[i: i + time_steps].as_matrix()

        # rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
        fcnn_df.append(data_)
    return np.array(fcnn_df)


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
    if labels:
        ntest = int(round(len(data) * (1 - test_size)))
        nval = int(round(len(data[:ntest]) * (1 - val_size)))
        df_train, df_val, df_test = data[:nval], data[nval:ntest], data[ntest:]
    else:
        df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (fcnn_data(df_train, time_steps, labels=labels),
            fcnn_data(df_val, time_steps, labels=labels),
            fcnn_data(df_test, time_steps, labels=labels))
