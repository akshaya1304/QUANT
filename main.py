import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle

#i had run this code in python file inside and environment E2 in anaconda GUI. 
#you can access this file by typing <streamlit run main.py> inside administrator
#i had used streamlit to integrate backend and frontend interface
#and trained the model from scratch here using linear regression method and not using libraries like ski-learn ect:
#python model is saved in pickel, uploaded and opened in pandas, indicators and calculated manually using NumPy and ML trained using neural network here
#its a combination of weights and activation values which through summation in each line finally traines the machine to either BUY or SELL
#SMA (Simple Moving Average)= summation(Closing price on each day)/ no. of days
#RSI (Relative Strength Index)RSI is a momentum indicator that measures how overbought or oversold an asset is

import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.title("Forex Trading Signal Generator")

file = st.file_uploader("Upload your forex CSV file", type="csv")

if file is not None:
    df = pd.read_csv(file)
    st.write(df.head())

    # 2. calculate indicators
    def compute_sma(data, window):
        return data['close_eurusd'].rolling(window=window).mean()

    def compute_rsi(data, period=14):
        delta = data['close_eurusd'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['SMA_14'] = compute_sma(df, 14)
    df['RSI_14'] = compute_rsi(df, 14)
    df.dropna(inplace=True)

    st.line_chart(df[['close_eurusd', 'SMA_14']])
    st.line_chart(df[['RSI_14']])

    # 3. Machine Learning Model
    df['Future'] = df['close_eurusd'].shift(-1)
    df['Signal'] = np.where(df['Future'] > df['close_eurusd'], 1, 0)
    df.dropna(inplace=True)

    X = df[['SMA_14', 'RSI_14']].values
    y = df['Signal'].values
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(X, y, lr=0.01, epochs=1000):
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0

        for _ in range(epochs):
            z = np.dot(X, weights) + bias
            pred = sigmoid(z)
            dw = (1/m) * np.dot(X.T, (pred - y))
            db = (1/m) * np.sum(pred - y)
            weights -= lr * dw
            bias -= lr * db

        return weights, bias

    def predict(X, weights, bias):
        z = np.dot(X, weights) + bias
        return sigmoid(z)

    weights, bias = train(X_train, y_train)
    preds = predict(X_test, weights, bias)
    pred_labels = np.where(preds > 0.5, 1, 0)
    acc = np.mean(pred_labels == y_test)
    st.success(f"Model Accuracy: {acc:.2f}")

    with open("model.pkl", "wb") as f:
        pickle.dump((weights, bias), f)

    # 4. Signal Generator
    latest_data = X[-1].reshape(1, -1)
    signal = predict(latest_data, weights, bias)[0]

    if signal > 0.5:
        st.markdown("### 📈 Signal: BUY")
    else:
        st.markdown("### 📉 Signal: SELL")

    st.write(f"Signal Score: {signal:.4f}")
    st.caption("Signal based on latest RSI and SMA values.")












        
