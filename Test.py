import PIL.Image
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

week_merge = pd.read_csv("Data/week_merge.csv", dtype={'fips': str})
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(week_merge, geojson=counties, locations='fips',
                    color='casespercapita',
                    color_continuous_scale="Viridis",
                    range_color=(0, 0.5),
                    scope="usa",
                    labels={'casespercapita': '%new cases <br> (on county level)'},
                    animation_frame="week"
                    )

