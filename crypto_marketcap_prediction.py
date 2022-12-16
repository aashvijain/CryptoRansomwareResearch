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


#Load dataset for crypto weekly marketcap
def parser(x):
    return datetime.strptime(x, '%b %d, %Y')
series = read_csv('crypto_weekly_marketcap.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#print(series.head())

#Set the columns and Index
series.columns = ['Crypto MarketCap']
series.index = pd.to_datetime(series.index)
series.plot()
pyplot.xlabel("Year")
pyplot.ylabel("USD Billions")
pyplot.title("Total Marketcap of Crypto Currency Per Year")
pyplot.show()

#Decompose the series to find the trend and make it stationary.
#Applying additive model with a 100 week period. 
result = seasonal_decompose(series, model='multiplicative', extrapolate_trend=10, period=100)

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

#print(stepwise_model.aic())

train = series.loc['2016-01-06':'2020-11-18']
print(train.tail())

stepwise_model.fit(train)

test = series.loc['2020-11-25':]
#print(test.head())
#print(test.tail())
prediction_start_date = datetime(2020, 11, 25)
prediction_end_date = datetime(2030, 11, 25)
prediction_index = pd.date_range(prediction_start_date, prediction_end_date, freq='W-WED') 

future_forecast = stepwise_model.predict(522)
print(future_forecast)

#future_forecast.index = pd.to_datetime(test.index)
future_forecast.index = prediction_index
print(future_forecast.index)
future_forecast = pd.DataFrame(future_forecast,index = future_forecast.index,columns=['Prediction'])
#future_forecast = pd.DataFrame(future_forecast,index = future_forecast.index,columns=['Prediction'])
#future_forecast.plot()
#pyplot.show()

pd.concat([test,future_forecast],axis=1).plot()
#pyplot.show()


future_forecast2 = future_forecast
pd.concat([series,future_forecast2],axis=1).plot()
pyplot.xlabel("Year")
pyplot.ylabel("USD Billions")
pyplot.title("Prediction of Marketcap of Crypto Currency Per Year")
pyplot.show()

#Save the predicted values as a CSV file
future_forecast2.to_csv('crypto_marketcap_prediction.csv')






