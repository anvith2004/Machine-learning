import pandas as pd
import statistics
import matplotlib.pyplot as plt

df = pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')

price = df['Price']
mean = statistics.mean(price)
variance = statistics.variance(price)

print(f"Mean of Price: {mean}")
print(f"Variance of Price: {variance}")

df['Date'] = pd.to_datetime(df['Date'])

price_data = df['Price']
date_data = df['Date']

wed_df = df[date_data.dt.weekday == 2]
wed_price_data = wed_df['Price']

pop_mean = statistics.mean(price_data)

sample_mean_wed = statistics.mean(wed_price_data)

print(f"population Mean: {pop_mean}")
print(f" mean Wednesday: {sample_mean_wed}")

april_df = df[date_data.dt.month == 4]
april_price_data = april_df['Price']

# Calculate population mean
population_mean = statistics.mean(price_data)

# Calculate sample mean for April
sample_mean_april = statistics.mean(april_price_data)

print(f"Population Mean: {population_mean}")
print(f"Sample Mean for April: {sample_mean_april}")

df['Date'] = pd.to_datetime(df['Date'])

# Extract 'Chg%' column and 'Date' column
chg_data = df['Chg%']
date_data = df['Date']

# Calculate the probability of making a loss (negative Chg%)
probability_of_loss = (chg_data < 0).mean()

print(f"Probability of Making a Loss: {probability_of_loss:.2%}")

# Filter data for Wednesdays
wednesday_df = df[date_data.dt.day_name() == 'Wednesday']
wednesday_chg_data = wednesday_df['Chg%']

# Calculate the probability of making a profit on Wednesday
probability_of_profit_on_wednesday = (wednesday_chg_data > 0).mean()

print(f"Probability of Making a Profit on Wednesday: {probability_of_profit_on_wednesday:.2%}")

# Calculate the total probability of being a Wednesday
probability_of_being_wednesday = (date_data.dt.day_name() == 'Wednesday').mean()

# Calculate the conditional probability of making a profit, given that today is Wednesday
conditional_probability_of_profit_given_wednesday = (probability_of_profit_on_wednesday / probability_of_being_wednesday)

print(f"Conditional Probability of Making a Profit, Given Today is Wednesday: {conditional_probability_of_profit_given_wednesday:.2%}")

# Add day of the week to DataFrame
df['Day_of_Week'] = date_data.dt.day_name()

# Scatter plot of Chg% against Day of the Week
plt.figure(figsize=(10, 6))
plt.scatter(df['Day_of_Week'], df['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter Plot of Chg% Data Against Day of the Week')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
