import mplfinance as mpf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from data_preprocessing import load_and_process_data
from tensorflow.keras.callbacks import EarlyStopping

class MoexModel:
    def __init__(self, training_mode=False, directory='moex_data', load_test_data=False):
        self.training = training_mode
        self.directory = directory
        if load_test_data:
            self.data = load_and_process_data(self.directory)
        if self.training:
            self.model = self.create_model()
            for ticker, normalized_data in self.data:
                X_test, y_test = self.train_stock_model(
                    normalized_data,
                    window_size=28,
                    test_size=0.2,
                    epochs=250,
                    batch_size=64
                )


                y_pred, mse = self.evaluate_model(X_test, y_test)
                self.plot_predictions(y_test, y_pred, ticker)
            self.save_model()
        else:
            self.model = load_model(f'neural_moex_model.keras')

    def test_case(self):
        errors = 0
        values = 0
        for ticker, normalized_data in self.data:
            test_data = self.get_data_for_graphics(normalized_data, test_size=48)
            X_test, y_test = self.prepare_data_with_rolling_window(test_data, 28)

            errors_tmp, values_tmp = self.testing_data(X_test, y_test)
            errors += errors_tmp
            values += values_tmp

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(168, input_shape=(28, 6), return_sequences=True))
        self.model.add(LSTM(84, return_sequences=False))
        self.model.add(Dense(6))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def prepare_data_with_rolling_window(self, normalized_data, window_size=28):
        X, y = [], []
        features = ['open', 'high', 'low', 'value', 'volume', 'close']
        targets = ['open', 'high', 'low', 'value', 'volume', 'close']

        for i in range(len(normalized_data) - window_size - 1):
            X.append(normalized_data[features].iloc[i:i + window_size].values)
            y.append(normalized_data[targets].iloc[i + window_size + 1].values)
        X, y = shuffle(np.array(X), np.array(y), random_state=42)

        return np.array(X), np.array(y)

    def split_data(self, normalized_data, test_size):
        num_test_samples = int(len(normalized_data) * test_size)

        train_data = normalized_data[:-num_test_samples]
        test_data = normalized_data[-num_test_samples:]

        return train_data, test_data

    def get_data_for_graphics(self, normalized_data, test_size: int):
        return normalized_data[-test_size:]

    def train_model(self, X_train, y_train, epochs, batch_size):
        self.model = self.create_model()
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

    def train_stock_model(self, normalized_data, test_size, epochs, batch_size, window_size=28, ):
        train_data, test_data = self.split_data(normalized_data, test_size=0.2)

        X_train, y_train = self.prepare_data_with_rolling_window(train_data, window_size)
        self.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size)

        X_test, y_test = self.prepare_data_with_rolling_window(test_data, window_size)

        return X_test, y_test

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()
        mse = np.mean((y_test - y_pred) ** 2)

        return y_pred, mse

    def plot_predictions(self, y_test, y_pred, ticker):
        plt.figure(figsize=(14, 7))
        plt.plot(y_test[:10][0], label='Real Prices')
        plt.plot(y_pred[:10][0], label='Predicted Prices', alpha=0.7)
        plt.title(f'Predicted vs Real Prices for {ticker}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save(f'neural_moex_model.keras')

    def testing_data(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        closing_prices_real = y_test[:, 5]
        closing_prices_pred = y_pred[:, 5]

        dates = pd.date_range(start='2023-01-01', periods=len(closing_prices_real), freq='D')

        real_data = pd.DataFrame({
            'Open': y_test[:, 0],
            'High': y_test[:, 1],
            'Volume': y_test[:, 3],
            'Low': y_test[:, 2],
            'Close': y_test[:, 5],
        }, index=dates)

        pred_data = pd.DataFrame({
            'Open': y_pred[:, 0],
            'High': y_pred[:, 1],
            'Volume': y_pred[:, 3],
            'Low': y_pred[:, 2],
            'Close': y_pred[:, 5],
        }, index=dates)

        mpf.plot(real_data, type='candle', volume=True, style='charles', title='Real Prices', ylabel='Price',
                 show_nontrading=True)
        mpf.plot(pred_data, type='candle', volume=True, style='charles', title='Predicted Prices', ylabel='Price',
                 show_nontrading=True)

        mse = np.mean((closing_prices_real - closing_prices_pred) ** 2)

        errors = 0
        for i in range(1, len(closing_prices_real)):
            predicted_movement = 'up' if closing_prices_pred[i] > closing_prices_real[i - 1] else 'down'
            actual_movement = 'up' if closing_prices_real[i] > closing_prices_real[i - 1] else 'down'

            if predicted_movement != actual_movement:
                errors += 1

        return errors, len(closing_prices_real)

    def predict_next_values(self, input_data):

        features = ['open', 'high', 'low', 'value', 'volume', 'close']
        x = []
        
        for data in list(input_data):
            if all(feature in data for feature in features):
                x.append([
                    data['open'],
                    data['high'],
                    data['low'],
                    data['value'],
                    data['volume'],
                    data['close'],
                ])
            else:
                print(f"Отсутствует ключ в данных: {data}")

        input_data_array = np.array(x)

        input_data_array = input_data_array.reshape((1, input_data_array.shape[0], input_data_array.shape[1]))

        prediction = self.model.predict(input_data_array)
        prediction = prediction.flatten().tolist()
        output_data = {
            'open': round(prediction[0] + list(input_data)[-1]['open'], 2) if prediction else None,
            'high': round(prediction[1] + list(input_data)[-1]['high'], 2) if prediction else None,
            'low': round(prediction[2] + list(input_data)[-1]['low'], 2) if prediction else None,
            'value': round(prediction[3] + list(input_data)[-1]['value'], 2) if prediction else None,
            'volume': round(prediction[4] + list(input_data)[-1]['volume'], 2) if prediction else None,
            'close': round(prediction[5] + list(input_data)[-1]['close'], 2) if prediction else None
            
            }

        
        return output_data



def main():
    model = MoexModel(training_mode=False)
    model.test_case()

if __name__ == '__main__':
    main()