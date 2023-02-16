import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

pd.options.display.float_format = "{:,.2f}".format

# Create locators for ticks on the time axis
register_matplotlib_converters()

df_yearly = pd.read_csv('annual_deaths_by_clinic.csv')
# parse_dates avoids DateTime conversion later
df_monthly = pd.read_csv('monthly_deaths.csv', parse_dates=['date'])

"""Check out these two DataFrames"""
# print(df_yearly.shape)
# print(df_yearly.head())

# print(df_monthly.shape)
# print(df_monthly.head())

"""Check for Nan Values and Duplicates"""
# df_yearly.info()
# df_monthly.info()
# print(df_yearly.isna().values.any())
# print(df_monthly.isna().values.any())

# print(f"Any yearly duplicates? {df_yearly.duplicated().values.any()}")
# print(f"Any monthly duplicates? {df_monthly.duplicated().values.any()}")

"""Descriptive Statistics"""
# print(df_yearly.describe())
# print(df_monthly.describe())

"""Using the annual data, calculate the percentage of women giving birth who died throughout the 1840s at the hospital."""
prob = df_yearly.deaths.sum() / df_yearly.births.sum() * 100
# print(f'Chances of dying in the 1840s in Vienna: {prob:.3}%') # keep 3 decimals counting from first non-zero number

"""Create a Matplotlib chart with twin y-axes."""
# plt.figure(figsize=(14, 8), dpi=200)
# plt.title('Total Number of Monthly Births and Deaths', fontsize=18)
#
# ax1 = plt.gca()
# ax2 = ax1.twinx()
#
# ax1.grid(color='grey', linestyle='--')
#
# ax1.plot(df_monthly.date,
#          df_monthly.births,
#          color='skyblue',
#          linewidth=3)
#
# ax2.plot(df_monthly.date,
#          df_monthly.deaths,
#          color='crimson',
#          linewidth=2,
#          linestyle='--')
#
# plt.show()

"""Format the x-axis using locators for the years and months"""
# # Create locators for ticks on the time axis
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# years_fmt = mdates.DateFormatter('%Y')
#
# plt.figure(figsize=(14, 8), dpi=200)
# plt.title('Total Number of Monthly Births and Deaths', fontsize=18)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14, rotation=45)
#
# ax1 = plt.gca()
# ax2 = ax1.twinx()
#
# ax1.set_ylabel('Births', color='skyblue', fontsize=18)
# ax2.set_ylabel('Deaths', color='crimson', fontsize=18)
#
# # Use Locators
# ax1.set_xlim([df_monthly.date.min(), df_monthly.date.max()])
# ax1.xaxis.set_major_locator(years)
# ax1.xaxis.set_major_formatter(years_fmt)
# ax1.xaxis.set_minor_locator(months)
#
# ax1.grid(color='grey', linestyle='--')
#
# ax1.plot(df_monthly.date,
#          df_monthly.births,
#          color='skyblue',
#          linewidth=3)
#
# ax2.plot(df_monthly.date,
#          df_monthly.deaths,
#          color='crimson',
#          linewidth=2,
#          linestyle='--')
#
# plt.show()

"""Use plotly to create line charts of the births and deaths of the two different clinics at the Vienna General Hospital."""
# line_birth = px.line(df_yearly,
#                x='year',
#                y='births',
#                color='clinic',
#                title='Total Yearly Births by Clinic')
#
# line_birth.show()

# line_death = px.line(df_yearly,
#                x='year',
#                y='deaths',
#                color='clinic',
#                title='Total Yearly Births by Clinic')
#
# line_death.show()

"""Work out the percentage of deaths for each row in the df_yearly DataFrame by adding a column called "pct_deaths"."""
df_yearly["pct_deaths"] = df_yearly["deaths"] / df_yearly["births"]
# print(df_yearly.head())

"""Calculate the average maternal death rate for clinic 1 and clinic 2"""
# Solution 1
avg_dr = df_yearly.groupby("clinic").mean()

clinic1_avg_dr = avg_dr.iloc[0, 3]
# print("%.5f" % clinic1_avg_death)
# print("{0:.5f}".format(clinic1_avg_death))
# print(f"Average death rate in clinic 1 is {clinic1_avg_dr:.3}.")

clinic2_avg_dr = avg_dr.iloc[1, 3]
# print("%.5f" % clinic2_avg_death)
# print("{0:.5f}".format(clinic2_avg_death))
# print(f"Average death rate in clinic 2 is {clinic2_avg_dr:.3}.")

# Solution 2
clinic_1 = df_yearly[df_yearly.clinic == 'clinic 1']
avg_clinic1 = clinic_1.deaths.sum() / clinic_1.births.sum() * 100
# print(f'Average death rate in clinic 1 is {avg_clinic1:.3}%.')

clinic_2 = df_yearly[df_yearly.clinic == 'clinic 2']
avg_clinic2 = clinic_2.deaths.sum() / clinic_2.births.sum() * 100
# print(f'Average death rate in clinic 2 is {avg_clinic2:.3}%.')

"""Plotting the Proportion of Yearly Deaths by Clinic"""
# line = px.line(df_yearly,
#                x='year',
#                y='pct_deaths',
#                color='clinic',
#                title='Proportion of Yearly Deaths by Clinic')
#
# line.show()

"""Add a column called "pct_deaths" to df_monthly that has the percentage of deaths per birth for each row."""
# Date when handwashing was made mandatory
handwashing_start = pd.to_datetime('1847-06-01')

df_monthly["pct_deaths"] = df_monthly["deaths"] / df_monthly["births"]
# print(df_monthly.head())

"""Create two subsets from the df_monthly data: before and after Dr Semmelweis ordered washing hand."""
before_washing = df_monthly[df_monthly["date"] < handwashing_start]
bw_rate = before_washing["deaths"].sum() / before_washing["births"].sum() * 100
# print(f"Average death rate before 1847 was {bw_rate:.4}%.")

after_washing = df_monthly[df_monthly["date"] >= handwashing_start]
aw_rate = after_washing["deaths"].sum() / after_washing["births"].sum() * 100
# print(f"Average death rate AFTER 1847 was {aw_rate:.3}%.")

"""Create a DataFrame that has the 6 month rolling average death rate prior to mandatory handwashing."""
# Avoid the date column becoming index or being dropped during the calculation
roll_monthly = before_washing.rolling(window=6, on="date").mean()
# print(roll_monthly)

roll_df = before_washing.set_index("date").rolling(window=6).mean()
# print(roll_df)

"""Tuple with one element"""
single_tuple_error = (0)
# print(single_tuple_error)
# print(type(single_tuple_error))

single_tuple = (0,)
# print(single_tuple)
# print(type(single_tuple))

# tuple is an object delimited by comma, not an object enclosed in parentheses.
tup = 0, 1, 2
# print(tup)
# print(type(tup))

"""Empty tuple"""
empty_tuple = ()
# print(empty_tuple)
# print(type(empty_tuple))

empty_tuple_alt = tuple()
# print(empty_tuple_alt)
# print(type(empty_tuple_alt))

"""Tuple conversion"""
# when parameter is not passed
tuple1 = tuple()
# print("empty tuple:", tuple1)

# when an iterable(e.g., list) is passed
list1 = [1, 2, 3, 4]
tuple2 = tuple(list1)
# print("list to tuple:", tuple2)

# when an iterable(e.g., dictionary) is passed
dict = {1: 'one', 2: 'two'}
tuple3 = tuple(dict)
# print("dict to tuple:", tuple3)

# when an iterable(e.g., string) is passed
string = "geeksforgeeks"
tuple4 = tuple(string)
# print("str to tuple:", tuple4)

"""Tuple unpacking"""
l, = [1]
# print(l)
t, = (1,)
# print(t)
# print(tuple([1]))
# print(tuple((1,)))
x, y = [1, 2]
# print(x, y)
z, = [3]
# print(z)

"""Modify the Matplotlib chart from before to plot the monthly death rates"""
# # Create locators for ticks on the time axis
# years = mdates.YearLocator()
# months = mdates.MonthLocator()
# years_fmt = mdates.DateFormatter('%Y')
#
# plt.figure(figsize=(14, 8), dpi=200)
# plt.title('Percentage of Monthly Deaths over Time', fontsize=18)
# plt.yticks(fontsize=14)
# plt.xticks(fontsize=14, rotation=45)
#
# plt.ylabel('Percentage of Deaths', color='crimson', fontsize=18)
#
# ax = plt.gca()
# ax.xaxis.set_major_locator(years)
# ax.xaxis.set_major_formatter(years_fmt)
# ax.xaxis.set_minor_locator(months)
# ax.set_xlim([df_monthly.date.min(), df_monthly.date.max()])
#
# plt.grid(color='grey', linestyle='--')
#
# ma_line, = plt.plot(roll_df.index,
#                     roll_df.pct_deaths,
#                     color='crimson',
#                     linewidth=3,
#                     linestyle='--',
#                     label='6m Moving Average')
# bw_line, = plt.plot(before_washing.date,
#                     before_washing.pct_deaths,
#                     color='black',
#                     linewidth=1,
#                     linestyle='--',
#                     label='Before Handwashing')
# aw_line, = plt.plot(after_washing.date,
#                     after_washing.pct_deaths,
#                     color='skyblue',
#                     linewidth=3,
#                     marker='o',
#                     label='After Handwashing')
#
# plt.legend(handles=[ma_line, bw_line, aw_line],
#            fontsize=18)
#
# plt.show()

"""What was the average percentage of monthly deaths before/after handwashing?"""
avg_prob_before = before_washing.pct_deaths.mean() * 100
avg_prob_after = after_washing.pct_deaths.mean() * 100
mean_diff = avg_prob_before - avg_prob_after
times = avg_prob_before / avg_prob_after
# print(f'Chance of death during childbirth BEFORE handwashing: {avg_prob_before:.3}%.')
# print(f'Chance of death during childbirth AFTER handwashing: {avg_prob_after:.3}%.')
# print(f'Handwashing reduced the monthly proportion of deaths by {mean_diff:.3}%!')
# print(f'This is a {times:.2}x improvement!')

"""NumPy's .where() function example"""
# Where True, yield x, otherwise yield y. Example for multidimensional arrays
test = np.where([[True, False], [True, True]],  # True: [[1, 0], [1, 1]] False: [[0, 1], [0, 0]]
                [[1, 2], [3, 4]],  # condition: [[1, 0], [1, 1]] output: [[1, 0], [3, 4]]
                [[9, 8], [7, 6]])  # condition: [[0, 1], [0, 0]] output: [[0, 8], [0, 0]]

# print(test)

"""Use NumPy's .where() function to add a column to df_monthly that shows if a particular date was before or after the start of handwashing."""
df_monthly["washing_hands"] = np.where(df_monthly["date"] < handwashing_start, "No", "Yes")
# print(df_monthly.head())

"""Use plotly to create box plot of the data before and after handwashing."""
# box = px.box(df_monthly,
#              x='washing_hands',
#              y='pct_deaths',
#              color='washing_hands',
#              title='How Have the Stats Changed with Handwashing?')
#
# box.update_layout(xaxis_title='Washing Hands?',
#                   yaxis_title='Percentage of Monthly Deaths', )
#
# box.show()

"""Create a plotly histogram to show the monthly percentage of deaths."""
# hist = px.histogram(df_monthly,
#                     x='pct_deaths',
#                     color='washing_hands',
#                     nbins=30,
#                     opacity=0.6,
#                     barmode='overlay',
#                     histnorm='percent',
#                     marginal='box')
#
# hist.update_layout(xaxis_title='Proportion of Monthly Deaths',
#                    yaxis_title='Count', )
#
# hist.show()

"""Use Seaborn's .kdeplot() to create two kernel density estimates of the pct_deaths, one for before handwashing and one for after."""
# plt.figure(dpi=200)
# sns.kdeplot(before_washing.pct_deaths,
#             fill=True,
#             clip=(0,1))
# sns.kdeplot(after_washing.pct_deaths,
#             fill=True,
#             clip=(0,1))
# plt.title('Est. Distribution of Monthly Death Rate Before and After Handwashing')
# plt.xlim(0, 0.40)
# plt.show()

"""Use a t-test to determine if the differences in the means are statistically significant or purely due to chance."""
t_stat, p_value = stats.ttest_ind(before_washing.pct_deaths, after_washing.pct_deaths)
# print(f'p-palue is {p_value:.10f}')
# print(f't-statstic is {t_stat:.4}')
