import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import datetime

def page1():
    st.title(f'Closing Stock Price Analysis')

    min_date = datetime.date(2000, 1, 1)
    max_date = datetime.date(2050,12,30)
    start = st.date_input('Enter the Start Date (format: YYYY-MM-DD):', min_value=min_date, max_value=max_date)
    end = st.date_input('Enter the End Date (format: YYYY-MM-DD):', min_value=min_date, max_value=max_date)

    start_year = start.year
    end_year = end.year

    user_input = st.text_input('Enter Stock Ticker', 'AAPL')

    # Fetch data from Yahoo Finance using yfinance
    df = yf.download(user_input, start=start, end=end)

    st.subheader('Stock Attributes Description')

    st.markdown("""

        - **Date:** This is the date of the trading day.
        - **Open:** The price of the stock at the opening of the trading day.
        - **High:** The highest price of the stock during the trading day.
        - **Low:** The lowest price of the stock during the trading day.
        - **Close:** The price of the stock at the closing of the trading day.
        - **Adj Close:** The adjusted closing price.
        - **Volume:** The number of shares traded during the trading day.
        """)

    # Check if the DataFrame is empty (no data retrieved)
    if df.empty:
        st.write(f"No data available for {user_input} in the specified date range.")
    else:

        # Describing data
        st.subheader('Data Description:')
        st.dataframe(df.describe())

        # Visualizations
        st.subheader('Closing Price vs Time chart')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Close'])
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df['Close'].rolling(window=100).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ma100)
        ax.plot(df['Close'])
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        st.pyplot(fig)

        st.subheader('Closing Price vs Time chart with 100MA and 200MA')
        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ma100, 'r', label='MA100')
        ax.plot(ma200, 'g', label='MA200')
        ax.plot(df['Close'], 'b', label='Close Price')
        ax.legend()
        st.pyplot(fig)

        # Splitting data into training and testing
        data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        scaler=MinMaxScaler(feature_range=(0,1))

        # Load model
        model = load_model('stock_price.h5')

        # Testing part
        past_100_days = data_training
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(np.array(final_df).reshape(-1, 1))

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Final graph
        st.subheader('Predictions vs Original')
        fig2, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, 'b', label='Original Price')
        ax.plot(y_predicted, 'r', label='Predicted Price')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig2)

def page2():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf  # Import yfinance library
    from keras.models import load_model
    import streamlit as st
    from sklearn.preprocessing import MinMaxScaler

    st.title('Stock Price Prediction')
    st.write("Please enter the details below to predict the stock price:")

    # Function to preprocess the input date
    def preprocess_date(date_str):
        return pd.to_datetime(date_str)

    # Function to predict stock price for a given date
    def predict_stock_price(date, user_input):
        start = '2010-01-01'
        end = date
        df = yf.download(user_input, start=start, end=end)
        st.write("Fetched data:")
        st.dataframe(df)

        data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        scaler=MinMaxScaler(feature_range=(0,1))

        # Load model
        model = load_model('stock_price.h5')

        past_100_days = data_training
        final_df = past_100_days.append(data_testing, ignore_index=True)
        input_data = scaler.fit_transform(np.array(final_df).reshape(-1, 1))

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_  # Invert the scaling
        predicted_price = y_predicted * (1 / scaler[0])  # Apply inverse scaling

        st.write("Predicted price:", predicted_price)
        return predicted_price[0][0]

    # User input for stock ticker
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')

    # User input for date
    user_date = st.date_input('Enter the Date for Stock Price Prediction')

    # Preprocess the input date
    preprocessed_date = preprocess_date(str(user_date))

    # Button to trigger prediction
    if st.button('Predict Stock Price'):
        st.write("Calling predict_stock_price function...")
        # Predict stock price for the user-defined date
        predicted_price = predict_stock_price(preprocessed_date, user_input)
        st.success(f'{user_input} Predicted Stock Price for {preprocessed_date.date()}: {predicted_price}')

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ['Analysis', 'Prediction'])

    if selection == 'Analysis':
        page1()
    elif selection == 'Prediction':
        page2()

if __name__ == '__main__':
    main()
