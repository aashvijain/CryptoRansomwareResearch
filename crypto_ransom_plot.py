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
