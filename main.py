import streamlit as st
import yfinance as yf
from datetime import date
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


st.set_page_config(
    page_title="📈 Stock Price Prediction",
    page_icon="🧊",
    layout="wide"
)
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("Predict the future trends of any stock for the next 1 month using LSTM.")


stock_symbol = st.text_input("Enter the Stock Symbol (e.g., AAPL, MSFT, TSLA):", "AAPL")
predict_button = st.button("Predict Future's Trends")





def create_dataset(dataset,time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps),0]
        dataX.append(a)
        dataY.append(dataset[i+time_steps,0])
    return np.array(dataX), np.array(dataY)


if predict_button:
    try:
        st.write(f"Fetching data for {stock_symbol}...")

        current_date = date.today().strftime('%Y-%m-%d')
        data = yf.download(stock_symbol,start="2005-01-01",end=current_date)

        if data.empty:
            st.error("No data found for the given stock symbol. Please try another symbol.")
        else:
            st.success("Data fetched successfully!")

            st.subheader("Stock Data (Last 5 Rows)")
            st.write(data.tail())

            last_year_data = data.tail(365)

            df1 = data['Close'].values.reshape(-1,1)
            scaler = MinMaxScaler(feature_range=(0,1))
            df1_scaled = scaler.fit_transform(df1)

            training_size = int(len(df1_scaled) * 0.75)
            train_data,test_data = df1_scaled[:training_size], df1_scaled[training_size:]

            time_step = 60
            x_train, y_train = create_dataset(train_data,time_step)
            x_test, y_test = create_dataset(test_data,time_step)

            x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
            x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
                Dropout(0.2),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(loss='mean_squared_error',optimizer='adam')
            st.write("Training the model... This might take a few minutes.")
            model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50,batch_size=64,verbose=1)
            st.success("Model training completed!")


            x_input=test_data[-time_step:].reshape(1,-1)
            temp_input=list(x_input[0])


            lst_output = []
            n_future = 30
            for _ in range(n_future):

                if len(temp_input)>time_step:
                    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
                else:
                    x_input = np.array(temp_input).reshape(1, time_step, 1)

                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])


            future_prediction = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))

            plt.figure(figsize=(12, 6))
            plt.plot(last_year_data.index, scaler.inverse_transform(df1_scaled)[-365:],
                     label="Historical Data (Last 1 Year)")
            plt.plot(pd.date_range(last_year_data.index[-1], periods=n_future + 1, freq='D')[1:], future_prediction,
                     label="Future Predictions")
            plt.title(f'{stock_symbol} Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()

            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error Occurred:{e}")