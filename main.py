import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from sidebar import SideBar 


SideBar.sidebar_show()
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App 💰')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','ABNB','NFLX')

selected_stock = st.selectbox('Select dataset for prediction 🙏🏻',stocks)




tickerData = yf.Ticker(selected_stock)
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)
string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

@st.cache 
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)   
    data.reset_index(inplace=True)
    return data
data_load_state = st.text('Loading date... 🌩')
data = load_data(selected_stock)
data_load_state.text('Loading data... done! 🌨')

st.subheader('Raw Data')
st.write(data.tail())
st.subheader('Raw Data Description')
st.write(data.describe())





def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()



df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

n_years = st.slider('Years of prediction 🤔 :', 1, 4)

period = n_years * 365 

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Prediction Data 🧐')
st.write(forecast.tail())



st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

