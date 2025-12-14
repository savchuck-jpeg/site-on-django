import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_stock_data(directory):
    stock_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            ticker = filename[:-4]
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            data = process_data(data)
            stock_data.append((ticker, data))
    return stock_data


def process_data(data):
    data['begin'] = pd.to_datetime(data['begin'])
    sorted_data = data.sort_values(by='begin')
    return sorted_data


def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data[['open', 'close', 'high', 'low', 'value', 'volume']])
    normalized_data = pd.DataFrame(scaled_values, columns=['open', 'close', 'high', 'low', 'value', 'volume'])
    normalized_data['begin'] = data['begin'].values
    return normalized_data


def prepare_data(normalized_data, time_steps=1):
    feature_columns = ['open', 'high', 'low', 'value', 'volume']
    X, y = [], []
    for i in range(len(normalized_data) - time_steps):
        X.append(normalized_data[feature_columns].iloc[i:(i + time_steps), :].values)
        y.append(normalized_data['close'].iloc[i + time_steps])
    return np.array(X), np.array(y)


def load_and_process_data(directory):
    stock_data = read_stock_data(directory)
    processed_data = []
    for ticker, data in stock_data:
        normalized_data = normalize_data(data)
        processed_data.append((ticker, normalized_data))
    return processed_data
