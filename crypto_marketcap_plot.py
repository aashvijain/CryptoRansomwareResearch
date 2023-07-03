from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import datetime

#Load dataset for crypto weekly marketcap
def parser(x):
    return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('coin-dance-market-cap-weekly.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#Set the columns and Index
series.columns = ['Crypto MarketCap']
series.index = pd.to_datetime(series.index)
series.plot()
pyplot.xlabel("Year")
pyplot.ylabel("USD Billions")
pyplot.title("Total Marketcap of Crypto Currency Per Year")
pyplot.ticklabel_format(style='plain', axis='y')
pyplot.show()
