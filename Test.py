import PIL.Image
import dash
import matplotlib.pyplot as plt
from dash import html
from dash import dcc
import dash_gif_component as gif
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

# Make the app
app = dash.Dash(__name__,
                title='MDA Project',
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# add this for heroku
server = app.server


# Add dcc.Location in the layout
app.layout = dbc.Container(
    gif.GifPlayer(
        gif='assets/media/Covid19_Spread_fulldata_lowQuality.gif',
        still='assets/media/PlaceholderImage3.png'
    )
)

if __name__ == '__main__':
    app.run_server(debug=True)
