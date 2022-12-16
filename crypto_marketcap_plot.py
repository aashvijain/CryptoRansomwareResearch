from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import datetime


#Load dataset for crypto weekly marketcap
def parser(x):
    return datetime.strptime(x, '%b %d, %Y')
series = read_csv('crypto_weekly_marketcap.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#print(series.head())

#Set the columns and Index
series.columns = ['Crypto MarketCap']
series.index = pd.to_datetime(series.index)
series.plot()
pyplot.show()
