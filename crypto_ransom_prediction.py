#####
#ARIMA model based prediction program to 
#predict the ransomware payment made in cryptocurrency
#for 2020 to 2030.
#It uses a data set from Statistica & Chainanalysis that has the yearly  
#ransomware payment made using cryptocurrency as the input data set.
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

#Load dataset for ransomware payment in cryptocurrency 
def parser(x):
    return datetime.strptime(x, '%b %d, %Y')
series = read_csv('crypto_ransom_payment.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#print(series.head())

series.plot()

  
pyplot.xlabel("Year")
pyplot.ylabel("USD Millions")
pyplot.title("Total Ransomware Payment in Crypto Currency Per Year")
pyplot.show()

#Decompose the series to find the trend and make it stationary.
#Applying additive model with a 100 week period. 
result = seasonal_decompose(series, model='additive', extrapolate_trend='freq', period=3)

#Uncomment these to see the decomposed series plots.
#result.plot()
#pyplot.show()

#Using auto_arima to find value of P, D and Q
stepwise_model = auto_arima(series, start_p=1, start_q=1,
    max_p=3, max_q=3, m=3,
    start_P=1, seasonal=True,
    d=0, D=1, trace=False,
    error_action='ignore',
    supress_warnings=True,
    stepwise=False)

print(stepwise_model.aic())

#use data from 2013 to 2021 for training the model
#we are using all available samples for training as the 
#total number of samples are very few
train = series.loc['2013-01-01':'2021-01-01']
print(train.tail())

stepwise_model.fit(train)

test = series.loc['2020-01-01':]
#print(test.head())
#print(test.tail())
prediction_start_date = datetime(2020, 1, 1)
prediction_end_date = datetime(2030, 1, 1)
prediction_index = pd.date_range(prediction_start_date, prediction_end_date, freq='AS') 

#Predict for 2020-2030. 11 Years of predictions
future_forecast = stepwise_model.predict(11)
print(future_forecast)

future_forecast.index = prediction_index
#print(future_forecast.index)
future_forecast = pd.DataFrame(future_forecast,index = future_forecast.index,columns=['Prediction'])

#Show predicted values along with the input dataset values
#in a single graph
pd.concat([test,future_forecast],axis=1).plot()

future_forecast2 = future_forecast
pd.concat([series,future_forecast2],axis=1).plot()
pyplot.xlabel("Year")
pyplot.ylabel("USD Millions")
pyplot.title("Prediction of Ransomware Payment in Crypto Currency Per Year")

pyplot.show()

#Save the predicted values as a CSV file
future_forecast2.to_csv('cryto_ransom_prediction.csv')
