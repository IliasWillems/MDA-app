import dash
import matplotlib.pyplot as plt
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
from urllib.request import urlopen
import json
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

fipsCountyState = pd.read_csv("Data/fipsCountyState.csv")
water_waste_cases_by_county = pd.read_csv("Data/water_waste_cases_by_county.csv")
water_by_county_import = pd.read_csv("Data/wastewater_by_county.csv")

cases_by_county_import = pd.merge(water_waste_cases_by_county, fipsCountyState, how='inner')

water_by_county = water_by_county_import
water_by_county = water_by_county.drop([0])  # drop 2020-01-01
water_by_county['sampling_week'] = pd.to_datetime(water_by_county.sampling_week)
water_USA = water_by_county[['sampling_week', 'effective_concentration_rolling_average']]
water_USA = water_USA.groupby(by=['sampling_week'], as_index=False).mean().sort_values(by='sampling_week')

# Turn weekly data to daily data using past 3 days and next 3 days
impute_date_list = []

for week in water_USA.sampling_week:
    impute_date = []
    for i in range(-3, 4):
        impute_date.append(week + pd.Timedelta('%dD' % i))
    impute_date_list.append(impute_date)

water_USA['date'] = impute_date_list
water_USA = water_USA.explode('date')

# drop column and reset index
water_USA = water_USA.drop('sampling_week', 1)
water_USA = water_USA.reset_index(drop=True)
water_USA['date'] = water_USA['date'].astype('str')

us_counties = cases_by_county_import
us_counties = us_counties.loc[us_counties['fips'].notnull(), :]
us_counties['fips'] = us_counties['fips'].astype('int')
cases_USA = us_counties[['date', 'cases']]
cases_USA = cases_USA.groupby(by=['date'], as_index=False).sum().sort_values(by='date')
cases_USA['increase_cases'] = cases_USA.cases.diff()
cases_USA.iloc[0, 2] = 0
cases_USA = cases_USA[['date', 'increase_cases']]
cases_USA['increase_cases'] = cases_USA['increase_cases'] / 1000

# merge two data frames
cases_and_water_USA = cases_USA.merge(water_USA, on='date')

# After tuning (see notebooks)
optimize_lag = 11
lag = 11

water_USA2 = water_USA.copy(deep=True)
water_USA2['date'] = water_USA2['date'].astype('datetime64')
water_USA2['date_adj'] = water_USA2['date'] + pd.Timedelta('%dD' % lag)
water_USA2 = water_USA2[['date_adj', 'effective_concentration_rolling_average']]
water_USA2.rename(columns={'date_adj': 'date'}, inplace=True)
water_USA2['date'] = water_USA2['date'].astype('str')

# merge two data frames
cases_and_water_USA2 = cases_USA.merge(water_USA2, on='date')

# Fit a linear model
After_may = cases_and_water_USA2[80:].copy(deep=True)
X = After_may.effective_concentration_rolling_average.to_numpy().reshape(-1, 1)
y = After_may.increase_cases.to_numpy().reshape(-1, 1)
reg11 = LinearRegression().fit(X, y)

# Plot the predictions versus actual data
water_USA3 = water_USA.copy(deep=True)
water_USA3['date'] = water_USA3['date'].astype('datetime64')
water_USA3['date_adj'] = water_USA3['date'] + pd.Timedelta('%dD' %lag)
water_USA3 = water_USA3[['date_adj','effective_concentration_rolling_average']]
water_USA3.rename(columns={'date_adj':'date'}, inplace=True)
water_USA3['date'] = water_USA3['date'].astype('str')

# merge two data frames
cases_and_water_USA3 = cases_USA.merge(water_USA3, on='date')
predict_cases = reg11.predict(
    cases_and_water_USA3.effective_concentration_rolling_average.to_numpy().reshape(-1, 1)).flatten()
true_cases = cases_and_water_USA3.increase_cases.to_numpy()
date = cases_and_water_USA3.date.to_numpy()

cases_and_water_USA3 = pd.DataFrame({'date': date, 'Predicted Cases': predict_cases, "True Cases": true_cases})

# time series plot
fig = make_subplots()
fig.add_trace(
    go.Scatter(x=cases_and_water_USA3['date'], y=cases_and_water_USA3['Predicted Cases'],
               name='Predicted Increased Cases', marker_color='blue')

)
fig.add_trace(
    go.Scatter(x=cases_and_water_USA3['date'], y=cases_and_water_USA3['True Cases'],
               mode='lines', name='True Increased Cases')
)

fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Cases (k)')
