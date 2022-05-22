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

# merge two data frames
cases_and_water_USA = pd.read_csv("Data/cases_and_water_USA.csv")

# time series plot
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=cases_and_water_USA['date'], y=cases_and_water_USA['effective_concentration_rolling_average'],
               name='Concentration Rolling Average'), secondary_y=False

)
fig.add_trace(
    go.Scatter(x=cases_and_water_USA['date'], y=cases_and_water_USA['increase_cases'],
               mode='lines', name='Increased Cases'), secondary_y=True
)

fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Copies / mL of sewage', secondary_y=False)
fig.update_yaxes(title_text='Cases (k)', secondary_y=True)


fig.show()
