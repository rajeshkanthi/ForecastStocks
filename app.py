import streamlit as st
from datetime import date, datetime, timezone
import yfinance
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly

start = "2020-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("stock pred 2")

stocks = ("GOOG", "AAPL", "MSFT", "GME")
selected_stock = st.selectbox("select stock",stocks)

n_years = st.slider("Years of prediction: ", 1, 4)
period= n_years * 365

def load_data(stock):
    data = yfinance.download(stock, start, today)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader("stock data")
st.write(data.tail())

# plot stock data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Open"], name="stock_open"))
fig1.add_trace(go.Scatter(x=data["Date"],y=data["Close"], name="stock_close"))
fig1.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

df_train = data[["Date", "Close"]]
st.write(df_train.tail())

def remove_timezone(dt):
    return dt.replace(tzinfo=None)

df_train['Date'] = df_train['Date'].apply(remove_timezone)

df_train = df_train.rename(columns={"Date": "ds", "Close": "y"}) 

m=Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#plot forecast
st.subheader("Forecast data")
fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2)
