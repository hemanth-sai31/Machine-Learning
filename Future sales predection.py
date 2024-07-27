import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
def create_sample_data():
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=24, freq='M'),
        'Sales': [200, 220, 230, 250, 240, 255, 260, 270, 275, 290, 300, 310,
                  320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df, label='Sales')
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
def check_stationarity(df):
    result = adfuller(df['Sales'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))
def decompose_data(df):
    decomposition = seasonal_decompose(df['Sales'], model='additive')
    decomposition.plot()
    plt.show()
def build_arima_model(df):
    model = ARIMA(df['Sales'], order=(5, 1, 0))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit
def forecast_sales(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    future_dates = pd.date_range(start=df.index[-1], periods=steps + 1, freq='M')[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(model_fit.fittedvalues, label='Fitted Values')
    plt.plot(future_dates, forecast, label='Forecast', color='red')
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    return forecast
def main(df, forecast_steps):
    visualize_data(df)
    check_stationarity(df)
    decompose_data(df)
    model_fit = build_arima_model(df)
    forecast = forecast_sales(model_fit, steps=forecast_steps)
    print('Forecasted Sales:')
    print(forecast)
df = create_sample_data()
forecast_steps = 12
main(df, forecast_steps)
