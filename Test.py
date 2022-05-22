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

agg_week_state = pd.read_csv("Data/agg_week_state.csv", dtype={'fips': str})

state = agg_week_state.loc[agg_week_state['state'] == 6]

test_start_week = 90
timesteps = 5

# Define the pipeline. Note that this pipeline does not include the information about the lag
pipe = Pipeline([('regressor', SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05))])

pd.options.mode.chained_assignment = None
yhat = savgol_filter(state['r'], 21, 3)
state.loc[:, 'yhat'] = yhat
pd.options.mode.chained_assignment = 'warn'

train = state.copy()[state.week < test_start_week][['yhat']]
test = state.copy()[state.week >= test_start_week][['yhat']]
train_data = train.values
test_data = test.values
train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[:, :, 0]
test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0, len(test_data)-timesteps+1)])[:, :, 0]

x_train, y_train = train_data_timesteps[:, :timesteps-1], train_data_timesteps[:, timesteps-1]
x_test, y_test = test_data_timesteps[:, :timesteps-1], test_data_timesteps[:, timesteps-1]

# Which values to check?
gammas_to_check = 5**np.arange(-2.1, 1.9, 1)
Cs_to_check = 10**np.arange(6)

params = {'regressor__gamma': gammas_to_check,
          'regressor__C': Cs_to_check}

# Search over parameter space using a gridsearch
gridsearch = GridSearchCV(pipe, params, verbose=1).fit(x_train, y_train)

# Fit pipe with optimal hyperparameters
pipe = Pipeline([('regressor', SVR(kernel='rbf', gamma=gridsearch.best_params_['regressor__gamma'],
                                   C=gridsearch.best_params_['regressor__C'], epsilon=0.05))])
pipe.fit(x_train, y_train)

# Make predictions
y_test_pred = pipe.predict(x_test).reshape(-1, 1)

# Predict k weeks into the future
to_predict = 30

predicted_values = np.empty((0, 1))
predictors = np.empty((0, 4))
predictors = np.vstack([predictors, np.array(x_test[-1, :])])

for i in range(to_predict):
    predicted_values = np.array([pipe.predict(predictors)])
    new_predictors = predictors[i][range(1, 4)]
    new_predictors = np.concatenate([new_predictors, [predicted_values[0][i]]], axis=0)
    predictors = np.vstack([predictors, new_predictors])

# Plot the predicted versus actual values using tuned values
train = state.copy()[state.week <= test_start_week][['yhat']]
test = state.copy()[state.week >= test_start_week][['yhat']]

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(test_start_week + 1), y=train['yhat'], name="Training data"))
fig.add_trace(go.Scatter(x=np.arange(test_start_week, 117), y=test['yhat'], name="Test data"))
fig.add_trace(go.Scatter(x=np.arange(test_start_week + timesteps, 117 + to_predict),
                         y=np.concatenate([y_test_pred.flatten(), predicted_values[0][range(1, to_predict)]]),
                         name="Predicted data", connectgaps=True))
