import pandas as pd
from matplotlib import pyplot
import csv
import numpy as np

daily_average = {}
min_daily_average = {}
max_daily_average = {}

input_file = csv.DictReader(open("coin-dance-market-cap-weekly-2.csv"))
for row in input_file:
    year = pd.to_datetime(row['Date']).year
    print(year)
    print(row['Actual Marketcap'])
    if daily_average.get(year) is not None:
        daily_average[year] += int(float(row['Actual Marketcap'])) 
        min_daily_average[year] += int(float(row['Actual Marketcap'])*0.01)
        max_daily_average[year] += int(float(row['Actual Marketcap'])*0.05)
    else:
        daily_average[year] = int(float(row['Actual Marketcap']))
        min_daily_average[year] = int(float(row['Actual Marketcap'])*0.01)
        max_daily_average[year] = int(float(row['Actual Marketcap'])*0.05)


for index,value in daily_average.items():
    value = daily_average[index]
    daily_average[index] = int(value/52)
print(daily_average)

keys = list(daily_average.keys())
values = list(daily_average.values())
Min = list(min_daily_average.values())
Actual = [24, 150, 39, 152, 692, 602, 800]
Max = list(max_daily_average.values())
#pyplot.bar(range(len(daily_average)), values, tick_label=keys)
#pyplot.show()

n=8
r = np.arange(n)
width = 0.25

pyplot.bar(r, Min, color = 'b',
        width = width, edgecolor = 'black',
        label='Min (0.1% Of Crypto Marketcap)')
pyplot.bar(r+width, Actual, color = 'r',
        width = width, edgecolor = 'black',
        label='Actual Payment')
pyplot.bar(r + width+width, Max, color = 'black',
        width = width, edgecolor = 'black',
        label='Max (0.5% of Crypto Marketcap)')
  
pyplot.xlabel("Year")
pyplot.ylabel("USD Millions")
pyplot.title("Min-Max Range for Ransomware Payment Using Crypto Currency Per Year")
  
# plt.grid(linestyle='--')
pyplot.xticks(r + width/2,['2016', '2017', '2018','2019','2020','2021', '2022', '2023'])
pyplot.legend()
  
pyplot.show()
