import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Create a bar graph for the Crypto marketcap
crypto_marketcap= pd.read_csv('crypto_marketcap_prediction_annual.csv')
df = pd.DataFrame(crypto_marketcap)

X = list(df.iloc[:, 0])
Y= list(df.iloc[:, 1])

# Plot the data using bar() method
plt.bar(X, Y, color='g')
plt.title("Crypto Marketcap in USD Billions")
plt.xlabel("Years")
plt.ylabel("Marketcap")

# Show the plot
plt.show()

# Create a bar graph for the Ransowmware payment using crypto currency
crypto_marketcap= pd.read_csv('crypto_ransom_prediction.csv')
df = pd.DataFrame(crypto_marketcap)

X = list(df.iloc[:, 0])
Y= list(df.iloc[:, 1])

# Plot the data using bar() method
plt.bar(X, Y, color='b')
plt.title("Ransowmware Payment Using Crypto Currency in USD Millions")
plt.xlabel("Years")
plt.ylabel("Payments")

# Show the plot
plt.show()


sns.set_context('talk', font_scale=0.8)

x = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
xi = list(range(len(x)))

df = pd.read_csv('crypto_correlation.csv')
overall_pearson_r = df.corr().iloc[0,1]
print(f"Pandas computed Pearson r: {overall_pearson_r:.3g}")

r, p = stats.pearsonr(df.dropna()['Crypto Marketcap In USD Billions'], df.dropna()['Crypto Ransomware Payment In USD Millions'])
print(f"Scipy computer Pearson r: {r:0.3g} and p-value: {p: .3g}")

f,ax = plt.subplots(figsize=(14,8))
plt.xticks(xi, x)
df.rolling(window=1, center=True).median().plot(ax=ax)
ax.set(xlabel='Years', ylabel='Value', title=f"Overall Pearson Correlation r = {overall_pearson_r:.2g} p = {p:.2g}");
plt.show()
