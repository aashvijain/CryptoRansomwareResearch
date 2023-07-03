import pandas as pd
import numpy as np
from datetime import datetime
import csv

df = pd.read_csv('coin-dance-market-cap-historical.csv')
df = df.reset_index()

# open the file in the write mode
f = open('coin-dance-market-cap-weekly.csv', 'w')

# create the csv writer
writer = csv.writer(f)
output = {}

for index, row in df.iterrows():
    date_object = datetime.strptime(row['Date'], '%Y-%m-%d').date()
    if (date_object.weekday() == 0):
        print(row['Date'], row['Value'])
        date = row['Date']
        val = row['Value'] / 1000000000
        info = str(date + "," + str(val))
        print(info)
        writer.writerow([info])
    
# close the file
f.close()
