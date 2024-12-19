from matplotlib import pyplot as plt
from torch import dropout
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2

# Data Fetching and Preprocessing Class
class DataHandler:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data['Return'] = self.data['Close'].pct_change()  # Calculate daily returns
        self.data.dropna(inplace=True)  # Drop rows with NaN values in the beginning

    def add_technical_indicators(self):
        self.data['SMA'] = self.data['Close'].rolling(window=20).mean()  # 20-day SMA
        self.data['EMA'] = self.data['Close'].ewm(span=20, adjust=False).mean()  # 20-day EMA
        self.data['RSI'] = self._calculate_rsi(self.data['Close'])  # Calculate RSI
        self.data.dropna(inplace=True)  # Remove rows with missing SMA, EMA, or RSI

    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_data(self):
        return self.data

class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.X = None
        self.y = None

    def prepare_features_and_target(self):
        self.X = self.data[['SMA', 'EMA', 'RSI']]
        self.y = (self.data['Return'] > 0).astype(int)

    def split_and_scale(self, lookback=10):
        X_rolled, y_rolled = [], []
        for i in range(len(self.X) - lookback):
            X_rolled.append(self.X.iloc[i:i+lookback].values)
            y_rolled.append(self.y.iloc[i+lookback])

        X_rolled = np.array(X_rolled)
        y_rolled = np.array(y_rolled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_rolled, y_rolled, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[2]))
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[2]))

        X_train_scaled = X_train_scaled.reshape(X_train.shape[0], lookback, X_train.shape[2])
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], lookback, X_test.shape[2])

        return X_train_scaled, X_test_scaled, y_train, y_test

class StockModel:
    def __init__(self, lookback):
        self.lookback = lookback
        self.model = None

    def build_model(self, input_shape):
        # LSTM with Dropout and L2 regularization
        self.model = Sequential([
            LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1)  # Linear activation for regression-based prediction
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, batch_size=32, epochs=100):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_future(self, X_seed, days=30, data_history=None):
        predictions = []
        current_input = X_seed  # Start with the last rolling window (shape: (lookback, num_features))

        for _ in range(days):
            # Reshape to batch format for prediction (add batch dimension)
            current_input_batch = np.expand_dims(current_input, axis=0)  # Shape: (1, lookback, num_features)

            # Predict the next value
            next_prediction = self.model.predict(current_input_batch)[0, 0]  # Extract the scalar prediction
            predictions.append(next_prediction)

            # Simulate new feature values (e.g., repeat prediction for all features)
            next_real_data = np.array([next_prediction] * current_input.shape[1])  # Shape: (num_features,)
            next_real_data = next_real_data.reshape(1, -1)  # Shape: (1, num_features)

            # Update the rolling window
            current_input = np.vstack([current_input[1:], next_real_data])  # Slide the window forward

        return predictions




# Main Script
if __name__ == "__main__":
    # Data Fetching and Preprocessing
    handler = DataHandler("AAPL", "2024-01-03", "2024-12-03")
    handler.fetch_data()
    handler.add_technical_indicators()
    data = handler.get_data()
    data.to_excel("apple_data.xlsx")

    preprocessor = DataPreprocessor(data)
    preprocessor.prepare_features_and_target()

    # Data Preparation
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale()
    print(f"Shape of X_train after reshaping: {X_train.shape}")

    # Model training
    lookback = 10
    model = StockModel(lookback)
    model.build_model(input_shape=(lookback, X_train.shape[2]))
    model.train(X_train, y_train)

    # Predict future
    X_seed = X_test[-1]
    predictions = model.predict_future(X_seed, days=30, data_history=data['Close'])
    print(predictions)

    # Visualization
    plt.plot(predictions, label="Predicted Prices")
    plt.title("Future Predictions")
    plt.legend()
    plt.show()