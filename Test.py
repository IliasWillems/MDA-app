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
from datetime import timedelta

cases_by_county_import = pd.read_csv("Data/cases_by_county_reduced.csv")
water_by_county_import = pd.read_csv("Data/wastewater_by_county.csv")
state_fips = pd.read_csv("Data/state_fips.csv")

state_nbr = 9
state_abr = state_fips.loc[state_fips['fips'] == state_nbr, 'Postal Code'].values[0]
state = state_fips.loc[state_fips['fips'] == state_nbr, 'Name'].values[0]

# if state_abr not in water_by_county_import.state.unique().tolist():
#     return ("Please select another state (in abbreviation)")

water_by_county = water_by_county_import.drop([0])  # drop 2020-01-01
water_by_county.rename(
    columns={'sampling_week': 'date', 'effective_concentration_rolling_average': 'concentration'}, inplace=True)
water_USA = water_by_county[['date', 'concentration']].groupby(by=['date'], as_index=False).mean().sort_values(
    by='date')

water_one_state = water_by_county[water_by_county['state'] == state_abr].reset_index(drop=True)
water_one_state = water_one_state[['date', 'concentration']].groupby(by=['date'],
                                                                     as_index=False).mean().sort_values(by='date')

# imputation using average concentration
date_USA = water_USA['date'].tolist()
date_one_state = water_one_state['date'].tolist()
date_missing = list(set(date_USA) - set(date_one_state))
water_USA.query('date in @date_missing')
water_one_state = pd.concat([water_one_state, water_USA.query('date in @date_missing')])

# turn weekly to daily data
water_one_state['date'] = pd.to_datetime(water_one_state.date, format='%Y/%m/%d')
water_one_state = water_one_state.set_index('date').resample('D').ffill().reset_index()
water_one_state['date'] = (pd.to_datetime(water_one_state['date']) - timedelta(3))
water_one_state['date'] = water_one_state['date'].astype('str')

# cases
cases_one_state = cases_by_county_import.loc[cases_by_county_import['fips'] == state_nbr].reset_index(drop=True)
cases_one_state = cases_one_state[['date', 'cases']].groupby(by=['date'], as_index=False).sum().sort_values(
    by='date')

cases_one_state['increased_cases'] = cases_one_state.cases.diff()
cases_one_state['increased_cases'] = cases_one_state['increased_cases'].fillna(0)
cases_one_state = cases_one_state.mask(cases_one_state['increased_cases'] < 0, 0)

cases_one_state = cases_one_state[['date', 'increased_cases']]
cases_one_state['increased_cases'] = cases_one_state['increased_cases'].astype(int)

# merge two data frames
water_cases_one_state = water_one_state.merge(cases_one_state, on='date', how='inner')

# time series plot
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=water_cases_one_state['date'], y=water_cases_one_state['concentration'],
               name='Concentration Rolling Average', marker_color='blue'), secondary_y=False

)
fig.add_trace(
    go.Scatter(x=water_cases_one_state['date'], y=water_cases_one_state['increased_cases'],
               name='Increased Cases'), secondary_y=True
)

dt_all = pd.date_range(start=water_cases_one_state.date[0],
                       end=water_cases_one_state.date[len(water_cases_one_state.date) - 1],
                       freq='D')
dt_all_py = [d.to_pydatetime() for d in dt_all]
dt_obs_py = [d.to_pydatetime() for d in pd.to_datetime(water_cases_one_state['date'])]
dt_breaks = [d for d in dt_all_py if d not in dt_obs_py]

if len(dt_breaks) > 100:
    fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

fig.update_layout(title_text='%s' % state, title_x=0.3)
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Copies / mL of sewage', secondary_y=False)
fig.update_yaxes(title_text='Cases', secondary_y=True)

fig.show()
