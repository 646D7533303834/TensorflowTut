import tensorflow as tf
import pandas as pd

file_path = 'data/'
file_name = '1114147.csv'
df_raw_data = pd.read_csv(file_path + file_name, encoding="ISO-8859-1", low_memory=False)

df_num_subset = df_raw_data.iloc[:, 2:]
df_num_subset.DATE = pd.to_datetime(df_num_subset.DATE)
# Trying to put the years more middle of the road here
df_num_subset['year_norm'] = (df_num_subset.DATE.dt.year - 2000) / 20
df_num_subset['month_norm'] = df_num_subset.DATE.dt.month / 12
df_num_subset['day_norm'] = df_num_subset.DATE.dt.day / 31
df_num_subset['time_norm'] = ((df_num_subset.DATE.dt.hour * 60) + df_num_subset.DATE.dt.minute) / 1440
print(df_num_subset['time_norm'])