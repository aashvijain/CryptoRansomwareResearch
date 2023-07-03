#####
#ARIMA model based prediction program to 
#predict the total marketcap of cryptocurrency
#for 2020 to 2030.
#It uses a data set from Statistica that has the weekly marketcap for all 
# cryptocurrency as the input data set.
####

#Import all the required python packages
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import chart_studio
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    forecast_arr = []
    actual_arr = []

    print("Calculating ARIMA Model Accuracy")

    for val1 in np.nditer(forecast):
        forecast_arr.append(val1)

    print("Forecast Len: ")
    print(len(forecast_arr))
    print(forecast_arr)

    for val2 in np.nditer(actual):
        actual_arr.append(val2)

    print("Actual Len: ")
    print(len(actual_arr))
    print(actual_arr)

    forecast_val = np.array(forecast_arr)
    actual_val = np.array(actual_arr)

    mape = np.mean(np.abs(forecast_val - actual_val)/np.abs(actual_val) * 100)  # MAPE
    me = np.mean(forecast_val - actual_val)             # ME
    mae = np.mean(np.abs(forecast_val - actual_val))    # MAE
    mpe = np.mean((forecast_val - actual_val)/actual_val * 100)   # MPE
    rmse = np.mean((forecast_val - actual_val)**2)**.5  # RMSE
    corr = np.corrcoef(forecast_val, actual_val)[0,1]   # corr
    mins = np.amin(np.hstack([forecast_val[:,None], 
                              actual_val[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast_val[:,None], 
                              actual_val[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax

    print('MAPE: ' + str(mape) + ',ME: ' + str(me) + ',MAE: ' + str(mae) + 
            ',MPE: ' + str(mpe) + ',RMSE: ' + str(rmse) + 
            ',CORR: ' + str(corr) + ',MINMAX: ' + str(minmax))


#Load dataset for crypto weekly marketcap
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('coin-dance-market-cap-weekly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#Set the columns and Index
#series.columns = ['Crypto MarketCap']
series.index = pd.to_datetime(series.index)
series.plot()
pyplot.xlabel("Year")
pyplot.ylabel("USD Billions")
#pyplot.title("Total Marketcap of Crypto Currency Per Year")
pyplot.ticklabel_format(style='plain', axis='y')
pyplot.show()

#Decompose the series to find the trend and make it stationary.
#Applying additive model with a 100 week period. 
#result = seasonal_decompose(series, model='multiplicative', extrapolate_trend=10, period=100)
#Uncomment these to see the decomposed series plots.
#result.plot()
#pyplot.show()

#Using auto_arima to find value of P, D and Q
stepwise_model = auto_arima(series, start_p=1, start_q=1,
    max_p=3, max_q=3, m=100,
    start_P=0, seasonal=True,
    d=1, D=1, trace=True,
    error_action='ignore',
    supress_warnings=True,
    stepwise=True)

print(stepwise_model.summary())

train = series.loc['2016-02-22':'2020-11-15']
print(train.tail())

stepwise_model.fit(train)

#test = series.loc['2020-11-25':]

#print(test.head())
#print(test.tail())
prediction_start_date = datetime(2020, 11, 22)
#prediction_start_date = datetime(2021, 1, 11)
prediction_end_date = datetime(2030, 11, 22)
prediction_index = pd.date_range(prediction_start_date, prediction_end_date, freq='W-MON') 

future_forecast = stepwise_model.predict(522)
print(future_forecast)


#future_forecast.index = pd.to_datetime(test.index)
future_forecast.index = prediction_index
print(future_forecast.index)
future_forecast = pd.DataFrame(future_forecast,index = future_forecast.index,columns=[''])
#future_forecast = pd.DataFrame(future_forecast,index = future_forecast.index,columns=['Prediction'])
#future_forecast.plot()
#pyplot.show()

#pd.concat([test,future_forecast],axis=1).plot()
#pyplot.show()

test = series.loc['2023-01-02':]
predicted_value = future_forecast['2023-01-02':'2023-06-19']
print(test)
print(predicted_value)
forecast_accuracy(predicted_value, test)

future_forecast2 = future_forecast
pd.concat([series,future_forecast2],axis=1).plot(legend=False)
pyplot.xlabel("Year")
pyplot.ylabel("USD Billions")
#pyplot.title("Prediction of Marketcap of Crypto Currency Per Year")
pyplot.show()

#Save the predicted values as a CSV file
future_forecast2.to_csv('crypto_marketcap_prediction.csv')

#print the model for the accuracy value
print(stepwise_model.arima_res_.data.endog)
