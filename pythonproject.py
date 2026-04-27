import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker

plt.style.use('seaborn-v0_8')

# ==============================
# 🔹 LOAD DATA
# ==============================
df = pd.read_csv(r'C:\Users\Vighneshrana\Desktop\python project\covid_data.csv')
#It removes extra spaces from column names.
df.columns = df.columns.str.strip()



#Instead of writing 'date' again and again, you store it in a variable
date_col = 'date'
country_col = 'country'
confirmed_col = 'total_cases'
deaths_col = 'total_deaths'


#Converts column 'date' → datetime format
df['Date'] = pd.to_datetime(df[date_col])

# ==============================
# 🔹 FILTER INDIA
# ==============================
india = df[df[country_col] == 'India'].copy()
india = india.sort_values('Date')
india = india.dropna(subset=[confirmed_col])

# ==============================
#  OBJECTIVE 1: TREND ANALYSIS
# ==============================
plt.figure(figsize=(10,5))
plt.plot(india['Date'], india[confirmed_col], label='Total Cases', linewidth=2)
plt.plot(india['Date'], india[deaths_col], label='Total Deaths', linewidth=2)

plt.title("Trend Analysis of COVID-19 Cases and Deaths in India Over Time")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ==============================
#  OBJECTIVE 2: COUNTRY COMPARISON
# ==============================
countries = ['India', 'United States', 'China', 'Pakistan']
comp = df[df[country_col].isin(countries)]

latest = comp.sort_values('Date').groupby(country_col).tail(1)

plt.figure(figsize=(8,5))
plt.bar(latest[country_col], latest[confirmed_col])

plt.title("Comparison of Total COVID-19 Cases Across Selected Countries")
plt.xlabel("Country")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.grid(axis='y')
plt.show()

# ==============================
#  OBJECTIVE 3: DAILY GROWTH
# ==============================
india['daily_cases'] = india[confirmed_col].diff().fillna(0)

plt.figure(figsize=(10,5))
plt.plot(india['Date'], india['daily_cases'])

plt.title("Daily COVID-19 Cases in India to Identify Growth Patterns and Waves")
plt.xlabel("Date")
plt.ylabel("Daily Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.grid()
plt.show()

# ==============================
#  OBJECTIVE 4: DEATH RATE
# ==============================
india['death_rate'] = india[deaths_col] / india[confirmed_col]

plt.figure(figsize=(10,5))
plt.plot(india['Date'], india['death_rate'])

plt.title("COVID-19 Death Rate Trend in India Over Time")
plt.xlabel("Date")
plt.ylabel("Death Rate")
plt.grid()
plt.show()

# ==============================
#  OBJECTIVE 6: BOX PLOT
# ==============================
plt.figure(figsize=(8,5))
plt.boxplot(india[confirmed_col])

plt.title("Distribution of COVID-19 Total Cases in India (Outliers & Spread Analysis)")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.grid()
plt.show()

# ==============================
#  OBJECTIVE 7: HISTOGRAM
# ==============================
plt.figure(figsize=(8,5))
plt.hist(india['daily_cases'], bins=50)

plt.title("Frequency Distribution of Daily COVID-19 Cases in India")
plt.xlabel("Daily Cases")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# ==============================
# 🔥 OBJECTIVE 8: REGRESSION ANALYSIS
# ==============================
india_lr = india.copy()
india_lr['Days'] = np.arange(len(india_lr))

X = india_lr[['Days']]
y = india_lr[confirmed_col]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(10,5))
plt.scatter(india_lr['Days'], y, label='Actual Data')
plt.plot(india_lr['Days'], y_pred, label='Regression Line')

plt.title("Linear Regression Model Showing COVID-19 Case Growth Trend in India")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.grid()
plt.show()

# ==============================
# 🔥 OBJECTIVE 9: MULTI COUNTRY TREND
# ==============================
plt.figure(figsize=(10,5))

for c in countries:
    temp = df[df[country_col] == c]
    temp = temp.sort_values('Date')
    plt.plot(temp['Date'], temp[confirmed_col], label=c)

plt.title("Comparative COVID-19 Growth Trends Across Countries")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.grid()
plt.show()

# ==============================
# 🔥 OBJECTIVE 5: ML PREDICTION
# ==============================
india_ml = india.copy()
india_ml['Days'] = np.arange(len(india_ml))

X = india_ml[['Days']]
y = india_ml[confirmed_col]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(india_ml), len(india_ml)+30).reshape(-1,1)
pred = model.predict(future_days)

plt.figure(figsize=(10,5))
plt.plot(india_ml['Days'], y, label='Actual')
plt.plot(future_days, pred, '--', label='Prediction')

plt.title("Prediction of Future COVID-19 Cases in India Using Linear Regression")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend()
plt.grid()
plt.show()
