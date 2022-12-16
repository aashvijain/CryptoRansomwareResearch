import pandas as pd
from matplotlib import pyplot
import csv
import numpy as np

daily_average = {}
min_daily_average = {}
max_daily_average = {}
prediction_daily_average = {}

input_file = csv.DictReader(open("crypto_marketcap_prediction.csv"))
for row in input_file:
    year = pd.to_datetime(row['Date']).year
    print(year)
    print(row['Prediction'])
    if daily_average.get(year) is not None:
        daily_average[year] += int(float(row['Prediction'])) 
        min_daily_average[year] += int(float(row['Prediction'])*0.01)
        max_daily_average[year] += int(float(row['Prediction'])*0.05)
    else:
        daily_average[year] = int(float(row['Prediction']))
        min_daily_average[year] = int(float(row['Prediction'])*0.01)
        max_daily_average[year] = int(float(row['Prediction'])*0.05)


for index,value in daily_average.items():
    value = daily_average[index]
    daily_average[index] = int(value/52)
print(daily_average)

keys = list(daily_average.keys())
values = list(daily_average.values())
Min = list(min_daily_average.values())

#Taken from the output of crypto_ransom_prediction.py
Prediction = [661, 909, 1041, 912, 864, 766, 886, 1249, 1222, 1230, 1732]

Max = list(max_daily_average.values())
#pyplot.bar(range(len(daily_average)), values, tick_label=keys)
#pyplot.show()

n=11
r = np.arange(n)
width = 0.25

pyplot.bar(r, Min, color = 'b',
        width = width, edgecolor = 'black',
        label='Min (0.1% of Crypto Marketcap)')
pyplot.bar(r+width, Prediction, color = 'red',
        width = width, edgecolor = 'black',
        label='Predicted Ransomware Payment')
pyplot.bar(r + width+width, Max, color = 'black',
        width = width, edgecolor = 'black',
        label='Max (0.5% of Crypto Marketcap)')
  
pyplot.xlabel("Year")
pyplot.ylabel("USD Millions")
pyplot.title("Prediction Of Ransomware Payment Using Crypto Currency Per Year")
  
# plt.grid(linestyle='--')
pyplot.xticks(r + width/2,['2020', '2021', '2022', '2023','2024','2025','2026', '2027', '2028', '2029', '2030'])
pyplot.legend()
  
pyplot.show()
