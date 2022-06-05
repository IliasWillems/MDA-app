import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_gif_component as gif
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
from datetime import timedelta
import PIL

# Make the app
app = dash.Dash(__name__,
                title='MDA Project',
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# add this for heroku
server = app.server

# For Heroku:
#   Username: willemsilias2000@gmail.com
#   Password: n)eLLZQE9$Sk"hE

########################################################################################################################
#                                            Load all data in advance                                                  #
########################################################################################################################
states = pd.read_csv("Data/state_fips.csv")
week_merge = pd.read_csv("Data/week_merge.csv", dtype={'fips': str})
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
agg_week_state = pd.read_csv("Data/agg_week_state.csv", dtype={'fips': str})
measures = pd.read_csv("Data/measures.csv")
Kmeans_clusters = pd.read_csv('Data/Kmeans_clustering.csv', dtype={'cluster': 'string', 'fips': 'string'})
democrat_rebuplican_vote = pd.read_csv('Data/Democrat_Republican_votes.csv')

cases_by_county_import = pd.read_csv("Data/cases_by_county_reduced.csv")
water_by_county_import = pd.read_csv("Data/wastewater_by_county.csv")

cases_and_water_USA = pd.read_csv("Data/cases_and_water_USA.csv")
cases_and_water_USA3 = pd.read_csv("Data/cases_and_water_USA3.csv")

prediction_results = pd.read_csv("Data/prediction_results.csv", dtype={'fips': str})


########################################################################################################################
#                                         Define all functions in advance                                              #
########################################################################################################################


@app.callback(
    Output(component_id='slider-container', component_property='style'),
    [Input(component_id='visualization-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'slider':
        return {'display': 'block', 'padding': 10}
    if visibility_state in ['animate', 'clusters']:
        return {'display': 'none', 'padding': 10}


@app.callback(
    [Output(component_id='visualization-clusters-info', component_property='style'),
     Output(component_id='visualization-cluster-correlation-figure', component_property='style')],
    [Input(component_id='visualization-dropdown', component_property='value')]
)
def show_hide_visualization_cluster_info(visibility_state):
    if visibility_state == 'clusters':
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output(component_id="id_visualization-display-component", component_property="children"),
    [Input(component_id="visualization-slider", component_property="value"),
     Input(component_id="visualization-dropdown", component_property="value")]
)
def update_figure_vis(week, to_display):
    if to_display == "slider":
        fig = px.choropleth(week_merge[week_merge['week'] == week], geojson=counties, locations='fips',
                            color='casespercapita',
                            color_continuous_scale="Viridis",
                            range_color=(0, 0.5),
                            scope="usa",
                            labels={'casespercapita': '%new cases <br> (on county level)'}
                            )
        fig.update_layout(title_text="Covid-19 cases for week " + str(week) + ". " + "(" +
                                     dt.datetime.strptime(week_merge.loc[week, 'startOfWeek'], "%Y-%m-%d").strftime(
                                         "%d/%b/%Y")
                                     + " to " +
                                     dt.datetime.strptime(week_merge.loc[week, 'endOfWeek'], "%Y-%m-%d").strftime(
                                         "%d/%b/%Y") + ")",
                          margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                          width=800)

        return dcc.Graph(figure=fig)

    elif to_display == "animate":
        fig = gif.GifPlayer(
            gif='assets/media/Covid19_Spread_fulldata_lowQuality.gif',
            still='assets/media/PlaceholderImage3.png'
        )

        return fig

    else:
        fig = px.choropleth(Kmeans_clusters, geojson=counties, locations='fips', color='cluster',
                            color_discrete_map={'0':'#636EFA','1':'#EF553B'},
                            scope='usa', labels={'cluster': 'cluster'}
                            )
        fig.update_layout(title_text="Clusters of similar Covid evolution",
                          margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                          width=800)

    return dcc.Graph(figure=fig)


@app.callback(
    Output(component_id='id_infection-rates-figure', component_property='figure'),
    [Input(component_id='infrates-states', component_property='value'),
     Input(component_id='infrates-measures', component_property='value')]
)
def update_figure_inf(state_nbr_int, measure):
    state = agg_week_state.loc[agg_week_state['state'] == state_nbr_int]
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=state['week'], y=state['r'], name="Infection rates"),
        row=1, col=1)

    # If it is possible to make a smoothed curve, do the following
    if state['r'].shape[0] > 21:
        # Add the smoothed curve to the plot
        yhat = savgol_filter(state['r'], 21, 3)
        fig.add_trace(go.Scatter(x=state['week'], y=yhat, name="Smoothed data"),
                      row=1, col=1)

    # Display selected measure (preset to Vaccination)
    state_measures = measures.loc[measures['fips'] == state_nbr_int, ['week', measure]]
    state_measures = state_measures.loc[state_measures[measure].shift() != state_measures[measure]]
    state_measures.reset_index(inplace=True, drop=True)
    state_measures = state_measures.loc[1:, ]
    state_measures.reset_index(inplace=True, drop=True)

    for change in range(state_measures.shape[0]):
        week = state_measures.loc[change, 'week']
        c = 'red' if state_measures.loc[change, measure] == 1 else 'green'
        fig.add_vline(x=int(week), line_color=c)

    return fig


@app.callback(
    Output(component_id='id_infection-rates-figure-svm', component_property='figure'),
    [Input(component_id='infrates-states', component_property='value')]
)
def update_figure_inf_svm(state_nbr_int):
    selected_state = state_nbr_int
    state = agg_week_state.loc[agg_week_state['state'] == selected_state]

    pd.options.mode.chained_assignment = None
    state['yhat'] = savgol_filter(state['r'], 21, 3)
    pd.options.mode.chained_assignment = 'warn'

    if selected_state not in pd.unique(measures['fips']):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=state['week'], y=state['yhat'], connectgaps=True))
        fig.update_layout(title_text="No vaccination data available for this state")
        return fig

    # Preprocess the vaccination data
    state_measures = measures.loc[measures['fips'] == selected_state, ['week', 'Vaccination']]
    state_measures = state_measures.loc[state_measures['Vaccination'].shift() != state_measures['Vaccination']]
    state_measures.reset_index(inplace=True, drop=True)
    state_measures = state_measures.loc[1:, ]
    state_measures.reset_index(inplace=True, drop=True)

    # Define the pipeline. Note that this pipeline does not include the information about the lag
    pipe = Pipeline([('regressor', SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05))])

    # Get the week from which point onwards vaccinations became available
    test_start_week = state_measures.loc[state_measures['Vaccination'] == 1].iloc[0, 0]

    timesteps = 15

    train = state.copy()[state.week < test_start_week][['week', 'yhat']]
    test = state.copy()[state.week >= test_start_week][['week', 'yhat']]
    train_data = train.values
    train_data_timesteps = np.array(
        [[j for j in train_data[i:i + timesteps]] for i in range(0, len(train_data) - timesteps + 1)])[:, :, 1]

    x_train, y_train = train_data_timesteps[:, :timesteps - 1], train_data_timesteps[:, timesteps - 1]

    # Which values to check?
    gammas_to_check = 5 ** np.arange(-2.1, 1.9, 1)
    Cs_to_check = 10 ** np.arange(6)

    params = {'regressor__gamma': gammas_to_check,
              'regressor__C': Cs_to_check}

    # Search over parameter space using a gridsearch
    gridsearch = GridSearchCV(pipe, params, verbose=0).fit(x_train, y_train)

    # Predict the remaining weeks
    to_predict = 116 - test_start_week

    predicted_values = np.empty((0, 1))
    predictors = np.empty((0, timesteps - 1))
    predictors = np.vstack([predictors, np.array(x_train[-1, :])])

    for i in range(to_predict):
        predicted_values = np.array([gridsearch.predict(predictors)])
        new_predictors = predictors[i][range(1, timesteps - 1)]
        new_predictors = np.concatenate([new_predictors, [predicted_values[0][i]]], axis=0)
        predictors = np.vstack([predictors, new_predictors])

    # Plot the predicted versus actual values using tuned values

    # Make it so that the lines connect
    predicted_values = np.insert(predicted_values, 0, train.iloc[-1, 1])

    new = pd.DataFrame({'week': [train.iloc[-1, 0]], 'yhat': [train.iloc[-1, 1]]})
    test = pd.concat([new, test])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['week'], y=train['yhat'], name="Training data", connectgaps=True))
    fig.add_trace(go.Scatter(x=test['week'], y=test['yhat'], name="Test data", connectgaps=True))
    fig.add_trace(go.Scatter(x=np.arange(test_start_week - 1, 116),
                             y=predicted_values[range(0, to_predict)],
                             name="Predicted data", connectgaps=True))

    fig.update_layout(
        xaxis_title="Week number",
        yaxis_title="Infection rate"
    )

    return fig


@app.callback(
    Output(component_id="id_state-selected-figure", component_property='figure'),
    [Input(component_id='infrates-states', component_property='value')]
)
def update_state_selected(target_state):
    states['target'] = states.apply(lambda row: 1 if row['fips'] == target_state else 0, axis=1)

    fig = px.choropleth(states,
                        locations='Postal Code',
                        color='target',
                        color_continuous_scale='spectral_r',
                        hover_name='Name',
                        locationmode='USA-states',
                        scope='usa')

    fig.update_layout(coloraxis_showscale=False,
                      margin={"r": 0, "t": 0, "l": 0, "b": 0, "autoexpand": True},
                      width=400)

    return fig


@app.callback(
    Output(component_id="id_waste-water-per-state-figure", component_property="figure"),
    [Input(component_id='id_waste-water-per-state-fips-input', component_property="value")]
)
def update_figure_waste_water_per_state(state_nbr):
    state_abr = states.loc[states['fips'] == state_nbr, 'Postal Code'].values[0]
    state = states.loc[states['fips'] == state_nbr, 'Name'].values[0]

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

    return fig


def update_figure_waste_water_prediction():
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

    return fig


@app.callback(
    Output(component_id='id_community-detection-figure-cases', component_property='figure'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_figure_community_detection_cases(period):
    df = pd.read_csv("Data/CommunityDetection/df_LB_cases_%s.csv" % period, dtype={'state fips': str, 'cluster': str})
    df = df.loc[df['state'].shift() != df['state']]

    fig = px.choropleth(df,
                        locations='state abbreviation',
                        color='cluster',
                        color_continuous_scale='spectral_r',
                        hover_name='state',
                        locationmode='USA-states',
                        scope='usa')

    periods = ['21/01/2020 - 20/06/2020', '21/06/2020 - 20/12/2020', '21/12/2020 - 20/06/2021',
               '21/06/2021 - 20/12/2021', '21/12/2021 - 11/04/2022']

    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                      title_text="Community detection on cases for the period <br> " + periods[period - 1])

    return fig


@app.callback(
    Output(component_id='id_community-detection-figure-deaths', component_property='figure'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_figure_community_detection_deaths(period):
    df = pd.read_csv("Data/CommunityDetection/df_LB_deaths_%s.csv" % period, dtype={'state fips': str, 'cluster': str})
    df = df.loc[df['state'].shift() != df['state']]

    fig = px.choropleth(df,
                        locations='state abbreviation',
                        color='cluster',
                        color_continuous_scale='spectral_r',
                        hover_name='state',
                        locationmode='USA-states',
                        scope='usa')

    periods = ['21/01/2020 - 20/06/2020', '21/06/2020 - 20/12/2020', '21/12/2020 - 20/06/2021',
               '21/06/2021 - 20/12/2021', '21/12/2021 - 11/04/2022']

    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                      title_text="Community detection on deaths for the period <br> " + periods[period - 1])

    return fig


@app.callback(
    Output(component_id='id_community-detection-general-info', component_property='children'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_general_info_community_detection(period):
    info = [
        "During the first period of the pandemic, Covid-19 was some new and frightening virus. Many states responded to"
        " this in the same way by issuing statewide lockdown orders. As can be seen on the figures below,"
        " the evolution of Covid cases and deaths is found to"
        " be similar throughout the United States. For the cases, there is only one state that does not belong"
        " to the large cluster, namely Northern Mariana Islands."
        " Concerning the deaths, Northern Mariane Islands, as well as Puerto Rico and Virgin Islands belong to"
        " separate one-state clusters. These are all islands that are separated from the USA mainland,"
        " making a different evolution plausible. ",
        "During the second period of the pandemic, the different states still seem to evolve similarly"
        " in terms of cases. Only one large cluster can be observed, excluding Hawaii, Northern Mariana"
        " Islands, Puerto Rico and Virgin Islands, which are islands separated from the USA mainland."
        " Concerning the evolution of the deaths however, two large clusters can be detected, as well as"
        " some separate one-state clusters. In the winter of 2020, a second wave of Covid 19 cases occurred"
        " with a peak that was even higher than before. Different state characteristics could have had an influence"
        " on the ability of a state to take care of severely ill people under these circumstances."
        " This makes a different evolution of Covid-19 deaths across states plausible.",
        "During the first six months of the year 2021, a general downward trend in cases can be observed in the USA."
        " All states are clustered together based on their evolution of cases, except for Northern Mariana Island and"
        " Virgin Islands, which are two islands separated from the USA mainland, making a slightly different evolution"
        " plausible. Moreover, the evolution of the deaths is similar for most of the states. This could be due to the"
        " decreasing number of cases, leading to a better ability to take care of severely ill people."
        " Therefore, differences in state characteristics could be less decisive in the evolution of deaths than"
        " is the case when there is a peak in Covid-19 infections."
        " It is also useful to notice that during this period, vaccination against Covid-19 were beginning to be rolled"
        " out to the public. Since this is just the beginning of the vaccination campaign, the average vaccination"
        " rate over this period"
        " is still low for all states. Besides, vaccination rate does not seem to influence the cluster"
        " assigned to a state yet.",
        "During this period, the vaccination campaign is in full swing. The broad population has had a chance"
        " to get fully vaccinated. Besides, the delta variant of Covid became dominant, leading to a new peak"
        " of Covid infections in the USA. However, this evolution does not seem to be the same in all states, since"
        " two separate large clusters of Covid-19 cases evolution are detected. Also for the evolution of the deaths,"
        " two clear clusters are found. Different state characteristics can influence these clusters.",
        "During the most recent period, Omicron has become the dominating variant. After a high peak in January, the"
        " number of cases shows in general a decreasing trend. It was found that this trend is similar for all states,"
        " since they are clustered together, except for the Northern Mariana Islands."
        " It is possible that the effect of vaccinations on the clustering solution has diminished because the"
        " vaccination rate is more similar for the different states. Moreover, the vaccines are less effective"
        " against infection with Omicron, compared to earlier Covid variants. The clustering solution of the deaths"
        " shows two large clusters and 8 one-state clusters."]
    return info[period - 1]


@app.callback(
    Output(component_id='id_community-detection-cases-info', component_property='children'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_text_community_detection_cases(period):
    info = [
        "For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
        " the cluster it belongs to.",

        # Period 2
        "For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
        " the cluster it belongs to.",

        # Period 3
        "For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
        " the cluster it belongs to.",

        # Period 4
        html.Div([
            "For this period, multiple clusters are detected. It is interesting to note that these communities are "
            "very similar to the communities that were found using K-means clustering based on cases per county "
            "in section 1. We can also try to predict cluster membership for each "
            "state. In order to produce interpretable results, as well as to be able to do variable selection, we use "
            "a logistic regression model. Starting with the full list of covariates displayed above, we end up"
            " selecting ",
            html.I("Uninsured (p = 0.027)"),
            " and ",
            html.I("poverty rate (p = 0.025)"),
            ". ",
            "It can be noticed that also a model selecting ",
            html.I("proportion vaccinated (p = 0.038)"),
            " and ",
            html.I("poverty rate (p = 0.014)"),
            " would be a useful model, only containing significant variables. "
            "Note that the significance of ",
            html.I("proportion vaccinated"),
            " would not come as a surprise, as in this time period the effectiveness of the vaccination "
            "should be optimal for the broad public. Comparing cluster 0 and cluster 1, the results indicate "
            "that the probability to belong to cluster 1 is higher for states with a lower vaccination rate"
            " and a higher poverty rate. "
            "However, in a model containing ",
            html.I("Uninsured"),
            ", ",
            html.I("poverty rate"),
            " and ",
            html.I("proportion vaccinated"),
            " both the variables ",
            html.I("Uninsured"),
            " and ",
            html.I("proportion vaccinated"),
            " are non-significant. This is probably due to some slight multicollinearity (the correlation between them"
            " is -0.4)."

        ]),
        # Period 5
        "For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
        " the cluster it belongs to."]

    return info[period - 1]


@app.callback(
    Output(component_id='id_community-detection-deaths-info', component_property='children'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_text_community_detection_deaths(period):
    info = [
        "For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
        " the cluster it belongs to.",

        # Period 2
        html.Div([
            "For this period, there are two large clusters detected. After a backwards variable selection it is "
            "determined that only the variable ",
            html.I("Uninsured"),
            " is significant in predicting the cluster for each state. The probability to belong to cluster 1 is higher"
            " for states with a larger proportion of uninsured people ",
            html.I("(p = 0.008)"),
            ". Just like in the overall analysis, the poverty rate turns out to be borderline insignificant."
        ]),

        # Period 3
        "For this period there are a lot of clusters but only one of them contains more than one state. "
        "We do not consider these single-state clusters as clusters and hence there is only one 'real' cluster. "
        "Therefore, it does not make sense to apply logistic regression here.",

        # Period 4
        html.Div([
            "Also for this period there are 2 clusters detected. Now, ",
            html.I("poverty rate"),
            " is significant. The probability that a state belongs to cluster 1 is higher if the poverty rate"
            " is higher. ",
            html.I("(p = 0.010)"),
            ". Unlike before, the proportion of people who are uninsured is no longer significant."
        ]),

        # Period 5
        html.Div([
            "For the last period, again 2 large clusters are detected. ",
            html.I("Uninsured"),
            " is a significant variable. The probability to belong to cluster 1 is higher for states"
            " with a higher proportion of people that are uninsured. ",
            html.I("(p = 0.004)"),
            " while the ",
            html.I("poverty rate"),
            " is no longer significant. ",
            "Moreover, ",
            html.I("proportion vaccinated"),
            " is borderline insignificant. "
            "However, it can be noted that in a model containing only the variable ",
            html.I("proportion vaccinated"),
            ", this variable is highly significant (p=0.004). Some multicollinearity due to the variables ",
            html.I("proportion vaccinated"),
            " and ",
            html.I("Uninsured"),
            " probably causes the proportion of vaccinated people to be non-significant (their correlation is -0.40)."
        ]),
    ]

    return info[period - 1]


@app.callback(
    [Output(component_id="id_random-forest-figure-cases", component_property="figure"),
     Output(component_id="id_random-forest-figure-deaths", component_property="figure")],
    [Input(component_id="random-forest-periods", component_property="value")]
)
def update_figures_random_forests(period: int):
    im_cases_link = "Figures/avgcases_"
    im_deaths_link = "Figures/avgdeaths_"

    # Period = 0 corresponds to using all data
    if period == 0:
        im_cases_link += "all"
        im_deaths_link += "all"
        title_part = "over all periods"
    else:
        im_cases_link += ("period" + str(period))
        im_deaths_link += ("period" + str(period))
        title_part = "for period " + str(period)

    im_cases_link += "_high.png"
    im_deaths_link += "_high.png"

    im_cases = PIL.Image.open(im_cases_link)
    im_deaths = PIL.Image.open(im_deaths_link)

    fig_cases, fig_deaths = px.imshow(im_cases), px.imshow(im_deaths)

    fig_cases.update_layout(title_text="Analysis of cases " + title_part,
                            xaxis={'showticklabels': False, 'ticks': ""},
                            yaxis={'showticklabels': False, 'ticks': ""},
                            margin=dict(l=20, r=20, t=30, b=20),
                            autosize=False,
                            width=700,
                            height=700)

    fig_deaths.update_layout(title_text="Analysis of deaths " + title_part,
                             xaxis={'showticklabels': False, 'ticks': ""},
                             yaxis={'showticklabels': False, 'ticks': ""},
                             margin=dict(l=20, r=20, t=30, b=20),
                             autosize=False,
                             width=700,
                             height=700)
    fig_deaths.update_xaxes(nticks=0)

    return fig_cases, fig_deaths


def update_figure_random_forest_prediction_period_5():
    prediction_results['fips'] = \
        prediction_results.apply(lambda row: "0" + row['fips'] if len(row['fips']) == 4 else row['fips'], axis=1)

    fig = px.choropleth(prediction_results, geojson=counties, locations='fips',
                        color='category',
                        color_continuous_scale="Viridis",
                        scope="usa",
                        )
    fig.update_layout(title_text="Correctness of classification for each county",
                      margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                      width=800)

    return fig


########################################################################################################################
#                                           Covid spread visualization                                                 #
########################################################################################################################

# General text that is always displayed
Covid_spread_general_text = html.Div([
    "In this section, we display the newly reported cases per county and per week. The slider can be used to choose the "
    "week for which the data must be shown.",
    html.Br(),
    html.Br(),
    "In order to get more insight in how Covid-19 evolves throughout the United States, we can try to cluster counties "
    "together based on how they evolve in terms of cases per week. To this end, we applied a K-means and spectral clustering "
    "algorithm on the data set."
    " If you'd like to know more about the clusters found by these algorithms"
    " and how they were obtained, please select “clusters” in the dropdown box below (it may take a while to load the map).",
    html.Br(),
    html.Br()
])

# Extra information about the clusters
Covid_spread_clusters_text = html.Div([
    "In order to apply a clustering algorithm, we first need a measure that can assign a distance to each pair of "
    "counties. Since the aim is to capture the similarities or differences in the evolution of the number of"
    " Covid-19 cases between the"
    " counties, we can represent each county in a 116 dimensional space where each dimension represents the number of"
    " new Covid-19 cases in a specific week for that county. Then, we can define the distance between two counties to be"
    " the Euclidean distance between their representations in that high-dimensional space.",
    html.Br(),
    html.Br(),
    "However, it is well known that K-means clustering suffers from the ",
    html.I("curse of dimensionality"),
    ": it tends to perform worse in high dimensional spaces. Furthermore, when using clustering methods, it is always "
    "advisable to work on standardized data. Therefore, the 116 dimensional points were first scaled and then, before "
    "using the K-means algorithm, their dimensionality was reduced using principal component analysis (PCA). Although "
    "K-means is one of the most well-known clustering algorithms, other algorithms such as spectral clustering might "
    "outperform K-means due to a higher flexibility in the shape of the clusters."
    " The results of applying K-means as well as spectral clustering on the "
    "pre-processed data still left a lot to be desired. It turned out that some outlier counties were throwing off the "
    "clustering algorithms. The solution to this was to add an outlier detector after the scaling step.",
    html.Br(),
    html.Br(),
    "The process just described contains steps that require the choice of a hyperparameter. More specifically, one "
    "should choose the clustering algorithm and "
    "the number of clusters in the clustering algorithm. When using K-means clustering, the number of principal "
    "components to retain in the dimensionality reduction step has to be chosen as well. On the other hand, when using"
    " spectral clustering, different methods can be used to assign the clustering labels (i.e. K-means or discretize)"
    " and the number of nearest neighbors to create the adjacency matrix has to be chosen as well."
    " The choices of these hyperparameters are not a "
    "priori clear and hence a careful parameter tuning should be performed. Luckily, ",
    html.I("skLearn"),
    " allows to  construct a pipeline and can tune these parameters for us. The pipeline parameters"
    " contain two different scalers (StandardScaler and MinMaxScaler), two different clustering algorithms"
    " (spectral clustering and K-means), two different outlier detection techniques (IsolationForest and OneClassSvm),"
    " the number of retained principal components (for K-means) ranges from 2 until 9 and the number"
    " of clusters from 2 until 5. The number of nearest neighbors for spectral clustering can be 5, 10 or 15."
    " A custom score function based on the silhouette score was used to evaluate the performance of the pipelines. In "
    "short, the silhouette score compares for each point the mean distance to all the points in its assigned clusters "
    "with the mean distance to all the points in its neighbouring cluster. The neighbouring cluster is the cluster that "
    "is closest to the assigned cluster of the point under consideration.",
    html.Br(),
    html.Br(),
    "The optimal result was based on spectral clustering using 2 clusters, Isolation Forest to detect outliers,"
    " the StandardScaler, using 5 nearest neighbors and assigning labels based on discretization."
    " One of these clusters contained only 22 counties, while the other cluster contained the remaining 3078"
    " included US counties. The silhouette score was equal to 0.294. To investigate whether K-means clustering"
    " led to other clusters, a new pipeline was tuned, removing the option spectral clustering from the parameters. "
    "This resulted in the clusters that are displayed on the map. The MinMaxScaler, OneClassSVM and 8 principal"
    " components were chosen and the silhouette score was 0.1551, which is however somewhat lower"
    " than was the case for spectral clustering. Note that the K-means clustering "
    "algorithm did not have any geographical "
    "information about the counties. It was able to find these clusters solely based on how Covid-19 evolved throughout "
    "the US.",
    html.Br(),
    html.Br(),
    "Now, a logistic model is fit to predict the probability to belong to a K-means cluster for each county."
    " The variables that are included in the model are:",
    html.Ul(children=[
        html.Div(children=["1. Proportion of people voting Republican during the elections of 2020, referred to as ",
                           html.I("Vote Republican.")]),
        html.Div(children=["2. Proportion of people voting Democrat during the elections of 2020, referred to as ",
                           html.I("Vote Democrat.")]),
        html.Div(
            children=["3. Pagerank score of each county based on the commuting flows between counties, referred to as ",
                      html.I("Pagerank score.")]),
        html.Div(children=["4. Population density of the county, referred to as ",
                           html.I("pop_density.")]),
        html.Div(
            children=["5. The poverty rate of the county, referred to as ",
                      html.I("PovertyRate.")]),
        html.Div(
            children=["6. The median age of the county in the year 2019, referred to as ",
                      html.I("Median Age.")]),
        html.Div(children=["7. The life expectancy ",
                           html.I("Life expectancy.")]),
        html.Div(
            children=["8. The proportion of people that are uninsured in the year 2019, referred to as ",
                      html.I("Uninsured.")]),
        html.Div(children=["9. The number of airports, referred to as ",
                           html.I("Airports.")]),
        html.Div(
            children=["10. The median individual income in the year 2019, referred to as ",
                      html.I("Median individual income.")]),
        html.Div(
            children=["11. The average vaccination rate, referred to as ",
                      html.I("Vaccination.")])
    ]),
    html.Br(),
    "The results show that the probability to belong to cluster 1 instead of cluster 0 is higher if:",
    html.Ul(children=[
        html.Div(children=["1. ", html.I("Vote Democrat"), " is lower"]),
        html.Div(children=["2. ", html.I("Vote Republican"), " is lower"]),
        html.Div(children=["3. ", html.I("PovertyRate"), " is higher"]),
        html.Div(children=["4. ", html.I("Life expectancy"), " is lower"]),
        html.Div(children=["5. ", html.I("Uninsured"), " is higher"]),
        html.Div(children=["6. ", html.I("Airports"), " is higher"]),
        html.Div(children=["7. ", html.I("Vaccination"), " is lower"])
    ]),
    html.Br()

])

# Make correlation plot of voters
covid_spread_clusters_votes_correlation_figure = px.scatter(x=democrat_rebuplican_vote['Vote Republican'],
                                                            y=democrat_rebuplican_vote['Vote Democrat'])
covid_spread_clusters_votes_correlation_figure.update_xaxes(title_text='Vote Republican')
covid_spread_clusters_votes_correlation_figure.update_yaxes(title_text='Vote Democrat')

# Write some explanation
covid_spread_clusters_votes_correlation_text = html.Div(children=[
    "As a final remark, one could note that we likely have a multicollinearity issue in this analysis, as we have "
    "variables for the proportion of both republican and democrat voters. However, computing the correlation between "
    "the two variables we only obtain a value of r = -0.021. Let’s investigate why this is the case.",
    html.Br(),
    html.Br(),
    "Another way of determining whether two variables are correlated is by plotting them against each other. If the "
    "data cloud we obtain in that way represents a flat ellipse, then the data are correlated.  Plotting the two "
    "variables at hand against each other, we see several of these (very) flattened ellipses. Hence, it turns out that "
    "the two variables are indeed very correlated, but only when conditioning on the amount of people that went to vote "
    "(after all, voting is not obligatory in the United States). Therefore, when looking at all the data, the two "
    "variables are not correlated and furthermore, it is useful to include them both in the model as they then also "
    "give information about the amount of people that voted in a county."
])

# Create a dropdown for options 'animate' and 'slider'
visualization_dropdown = dcc.Dropdown(
    id='visualization-dropdown',
    options=[{"label": 'Slider', 'value': 'slider'},
             {"label": 'Animate', 'value': 'animate'},
             {"label": 'Clusters', 'value': 'clusters'}
             ],
    value='slider')

# Create a slider
slider = dcc.Slider(id='visualization-slider',
                    min=1,
                    max=max(week_merge['week']),
                    value=1,
                    marks={str(i): str(i) for i in [np.arange(1, max(week_merge['week']), 10)]})

# Create a variable that stores what to display
visualization_display = update_figure_vis(1, "slider")

########################################################################################################################
#                                                Infection Rates                                                       #
########################################################################################################################

# General text that is always displayed
infection_rate_general_text = html.Div([
    "In this section, we display the infection number for each week and each state."
    " To this end, the weekly percentage of infections"
    " on a state level is used. Assuming people are infectious for 7 days, the infection number for a certain week"
    " can be computed as"
    " the number of cases in the next week divided by the number of cases in the current week."
    " This number represents the average amount of people that"
    " each Covid patient infects in that week. Also a smoothed curve is added to the plot of the infection numbers. "
    "More specifically, the smoother used is a Savitsky-Golay filter. In a nutshell, it fits low-degree polynomials "
    "through subsets of successive points in the data based on least-squares [Gallagher, 2020].",
    html.Br(),
    html.Br(),
    "[Gallagher, 2020]: ",
    html.I("Savitzky-Golay Smoothing and Differentiation Filter"),
    ", Neal B. Gallagher, Eigenvector  research, retrieved from ",
    html.I("https://www.researchgate.net/publication/338518012_Savitzky-Golay_Smoothing_and_Differentiation_Filter"),
    ", on 28/05/2022",
    html.Br(),
    html.Br(),
    "Additionally, some Covid measurements are displayed in terms of closing schools and mask obligation.",
    html.Br(),
    "Legend for the colours:",
    html.Ul(children=[
        html.Div(children=["1. Vaccination"]),
        html.Ul(children=[
            html.Div(children=["1. Red = Vaccination available for all people at"
                               " risk (elderly, front-line workers, etc.) "]),
            html.Div(children=["2. Green = Vaccination no longer available for all these people"])
        ]),
        html.Div(children=["2. Masks"]),
        html.Ul(children=[
            html.Div(children=["1. Red = Masks obligatory in all public spaces where "
                               "social distancing is not possible "]),
            html.Div(children=["2. Green = Masks not obligatory in all public spaces where social"
                               " distancing is not possible"])
        ]),
        html.Div(children=["3. Schools"]),
        html.Ul(children=[
            html.Div(children=["1. Red = At least some types of schools need to close"]),
            html.Div(children=["2. Green = No such restrictions"])
        ]),
    ]),
    html.Br(),
    "The state and Covid measures that are displayed can be chosen in the dropdown box."
])

# Create dropdown for state
infrates_states_dropdown = dcc.Dropdown(
    id='infrates-states',
    options=[{"label": x, "value": y} for x, y in zip(states.Name, states.fips)],
    value=20)

# Create dropdown for measure
infrates_measures_dropdown = dcc.Dropdown(
    id='infrates-measures',
    options=[{"label": 'Masks', 'value': 'Masks'},
             {"label": 'Schools closed', 'value': 'Close_schools'},
             {"label": 'Vaccinations', 'value': 'Vaccination'}
             ],
    value='Vaccination')

# Make the plot with some initial values
fig_inf = update_figure_inf(20, 'Vaccination')

# Create a plot of the selected state
fig_state_selected = update_state_selected(20)

# General text for support vector machines (above plot)
text_inf_svm_1 = html.Div([
    "To investigate for each of these measures what their effects on the infection rates were, we could try to predict "
    "the infection rates if these measures had not been implemented. To do so, we first train a support vector machine "
    "on the time series of infection rates before the measure of interest went in effect. Next, we can predict the "
    "infection rates for the period in which the measure was enforced based on this model in order to get an idea of "
    "how things would have looked like without it.",
    html.Br(),
    html.Br(),
    "One caveat of this approach is that masks were made obligatory relatively early, leaving us with a very "
    "small/insufficient amount of data to train the model on. Likewise, schools were closed very soon, leading to the "
    "same problem. Therefore, we only analyse the effect of vaccinations. On top of that, support vector machines "
    "using a radial basis function kernel have two hyperparameters to be tuned. Tuning is done by means of a gridsearch over "
    "logarithmically equidistant points, as is customary when tuning SVMs. Note that we do not tune the lag of the "
    "model but set it to 15 in all cases. This is done in order to reduce computational complexity. The value 15 was "
    "chosen as manual inspection indicated it performed well in most cases."
])

# General text for support vector machines (below plot)
text_inf_svm_2 = html.Div([
    "From these results, it is hard to conclude anything. Clearly, the SVM did not have enough training data to capture "
    "the complexity of the curve. This was also an ambitious goal, given the multitude of factors influencing the "
    "infection rates that are not included in this model."
])

# Create a plot of the support vector machine prediction
fig_inf_svm = update_figure_inf_svm(20)

########################################################################################################################
#                                            Waste water analysis                                                      #
########################################################################################################################

# General introduction for this section
waste_water_introduction_text = html.Div([
    "When people are infected with COVID-19, even if they don’t have any symptoms, the virus can still spread with their "
    "feces as well as saliva, and eventually enter the waste water system. This allows waste water surveillance to serve "
    "as an early warning that COVID-19 is going to spread in the community. When the virus concentration in the "
    "waste water starts to rise, the health department can take early action to prevent further spread of COVID-19.",
    html.Br(),
    html.Br(),
    "In the dropdown box below, select the state for which you want to see the Covid-19 cases, overlaid with the waste "
    "water data."
])
# Explanation of the prediction results
waste_water_prediction_text = html.Div([
    "We use a linear model to predict the cases based on the covid concentration in the waste water for the whole USA. "
    "From the time series plot, an obvious time lag between daily increased cases and virus concentration rolling average"
    " can be observed. Instead of using the concentration rolling average as a variable to predict the time series"
    " of increased cases, we shifted the wastewater data N (N = 1, 2,...., 15) days backward and fitted N"
    " linear models with daily increased cases as the target. Lowest MSE occurred when N = 11, so we used "
    "this model to predict daily cases and plotted the results. Except for the period when a lot "
    "of people are infected, a simple linear model with shifted wastewater data as the only variable "
    "can roughly predict daily increased cases."
])

# Create Input box for fips number
waste_water_per_county_fips_input = dcc.Dropdown(
    id='id_waste-water-per-state-fips-input',
    options=[{"label": x, "value": y} for x, y in zip(states.loc[states.fips.isin(cases_by_county_import['fips'])].Name,
                                                      states.loc[
                                                          states.fips.isin(cases_by_county_import['fips'])].fips)],
    value=20)

# Create figure visualizing the waste water covid concentration and cases
# fig_waste_water_per_county = update_figure_waste_water_per_county("19153")
fig_waste_water_per_state = update_figure_waste_water_per_state(20)

# Create figure displaying predictions
fig_waste_water_USA_prediction = update_figure_waste_water_prediction()

########################################################################################################################
#                                              Community Detection                                                     #
########################################################################################################################

# Create dropdown to select period
community_detection_dropdown = dcc.Dropdown(
    id='community-detection-periods',
    options=[{"label": '21/01/2020 - 20/06/2020', 'value': 1},
             {"label": '21/06/2020 - 20/12/2020', 'value': 2},
             {"label": '21/12/2020 - 20/06/2021', 'value': 3},
             {"label": '21/06/2021 - 20/12/2021', 'value': 4},
             {"label": '21/12/2021 - 11/04/2022', 'value': 5},
             ],
    value=1)

# Create figure for cases
fig_community_detection_cases = update_figure_community_detection_cases(1)

# Create figure for deaths
fig_community_detection_deaths = update_figure_community_detection_deaths(1)

# General text that is always displayed, irrespective of period chosen
community_detection_methodology_and_general_results = html.Div([
    "In this section, we construct a graph using Neo4j based on the covid data set. More specifically, the nodes represent the "
    "different states and two states are connected if the evolution of Covid-19 in these states follows the same trend. "
    "To this end, we look at correlations between the weekly rolling averages of the number of cases and create an "
    "undirected connection between them if the correlation is larger than 0.7. In a next step, we perform a community "
    "detection algorithm based on label propagation and group states according to their communities. In a completely "
    "analogous way, communities of states are formed based on the evolution of Covid-19 related deaths in the different "
    "states. In the dropdown menu below, you can specify a specific 6 month-period for which the communities are "
    "computed. Note that this subdivision in periods can be insightful since the developments related to Covid-19 are "
    "quickly evolving.",
    html.Br(),
    html.Br(),
    "Next, we can investigate if we could predict these clusters using several variables, listed below:",
    html.Ul(children=[
        html.Div(children=["1. Proportion of people who are uninsured, referred to as ",
                           html.I("Uninsured.")]),
        html.Div(children=["2. Proportion of people living in poverty, referred to as ",
                           html.I("Poverty_rate.")]),
        html.Div(children=["3. Proportion of people above the age of 65, referred to as ",
                           html.I("prop_age.")]),
        html.Div(children=["4. Proportion of people who are vaccinated, referred to as ",
                           html.I("prop.")]),
        html.Div(
            children=["5. The degree centrality of the state based on commuting flows between states, referred to as ",
                      html.I("Degree_centrality_com.")])
    ]),
    html.Br(),
    "When we do not make a subdivision of time periods and just cluster states over the whole period under observation "
    "(21/01/2020 - 11/04/2022) based on the evolution of Covid related deaths in each state, we can also try to predict "
    "the resulting clusters. Using a logistic regression model, it can be determined that the only significant variable "
    "in this prediction is ",
    html.I("Uninsured, p = 0.004"),
    ". Furthermore, the model shows that the probability to belong to cluster 1 is higher for states where"
    " the proportion of uninsured people is lower."
    "The poverty rate turned out to be borderline insignificant. An analysis for the clusters based on the cases is "
    "not possible as the algorithm only found one cluster when considering the whole time period. Even when making the "
    "subdivision, only multiple clusters were found based on the cases for the fourth time period (excluding "
    "single-state clusters).",
    html.Br(),
    html.Br(),
    "Below we visualize the clusters for each period and apply a logistic regression to model cluster memberships based "
    "on the predictors listed above. Since the sample size for each of these predictions is small (each state corresponds "
    "to one observation), Firth's logistic regression was also tried to make the model. However, the conclusion were "
    "each time the same, so we will only discuss the more commonly used logistic regression models."
])

# General information related to Covid for that period
community_detection_general_info = update_general_info_community_detection(1)

# Analyses for each of the periods
community_detection_cases_info = update_text_community_detection_cases(1)
community_detection_deaths_info = update_text_community_detection_deaths(1)

########################################################################################################################
#                                              Random Forests                                                          #
########################################################################################################################

random_forest_general_info = html.Div([
    "In this final section, we combine all the knowledge we obtained throughout our investigation of the evolution of "
    "Covid-19 in the United States to predict for each county whether it is a Covid-19 hotspot. Here we define ",
    html.I("hotspot"),
    " in two ways:",
    html.Ul(children=[
        html.Div("1. A county is a hotspot if its average infection rate is above the overall average for all "
                 "counties."),
        html.Div("2. A county is a hotspot if its average covid-related death rate is above the overall average for "
                 "all counties.")
    ]),

    "As was already often done in the previous sections, we will analyse the usual 5 periods separately, as well as "
    "all periods together. A consequence of this is that whether or not a county is a hotspot depends on the period "
    "under observation. Therefore, predicting in general whether a county is a Covid-19 hotspot is a little ambiguous "
    "at first. To disambiguate, we will predict whether or not a county is a hotspot for the last period, namely period "
    "5. To this end, a model is trained based on the data for the first 4 periods.",
    html.Br(),
    html.Br(),
    html.H4("Exploratory analysis"),
    "Let us first investigate the characteristics of each period separately as kind of an exploratory analysis. As a "
    "first step, we select a prediction model (in our case, a random forest model). Next, for each period, we (re)train "
    "that model on the "
    "data for the selected period. Finally, to investigate the characteristics of that period, we would like to know "
    "how important each feature is in the trained model. To this end, a Shapley plot can be constructed.",
    html.Br(),
    html.Br(),
    "Shapley plots are basically effect size plots: for each model, it automatically selects which features are most "
    "important (these are listed in the plots). For those features, you can see the effect of high/low values on that "
    "feature (indicated by the colour) on the probability of being hotspot (indicated on the x-axis). So for example, "
    "if you take the plot for cases in period 1, you can see that the households_speak_limited_english feature is quite "
    "important, in that counties with a low percentage of households that only speak limited English have a lower "
    "probability of being a hotspot (since blue is on the negative side of the x-axis). So each point for a feature "
    "refers to a county and indicates the marginal contribution of that feature to the prediction for that county.",
    html.Br(),
    html.Br(),
    "In reference to Section 4, we can see some similarities. For example, to predict the communities based on deaths "
    "for period 4, ",
    html.I("poverty rate"),
    " turned out to be a significant variable. Likewise, analysing deaths on a county level for period 4, ",
    html.I("median individual income"),
    " turns out to be the most important predictor. Also for period 4, we see that voting behaviour is very important "
    "for the predictions. This was not observed in the previous section as voting behaviour was not included in the "
    "analysis. Besides, also ",
    html.I("life expectancy"),
    " is important, a variable that is likely to be closely related to ",
    html.I("Uninsured"),
    " and ",
    html.I("Poverty rate"),
    ".",
    html.Br(),
    html.Br()
])

random_forest_dropdown = dcc.Dropdown(
    id="random-forest-periods",
    options=[
        {"label": 'All data', 'value': 0},
        {"label": 'Period 1', 'value': 1},
        {"label": 'Period 2', 'value': 2},
        {"label": 'Period 3', 'value': 3},
        {"label": 'Period 4', 'value': 4},
        {"label": 'Period 5', 'value': 5},
    ],
    value=2
)

random_forest_figure_cases, random_forest_figure_deaths = update_figures_random_forests(2)

random_forest_prediction_text = html.Div([
    html.H4("Predictive analysis"),
    "Lastly, we construct a random forest model based on features collected over a large amount of varying data set, as "
    "well as extracted features like for example (but not limited to) centrality measures based on commuting flows. As "
    "explained before, we train this model on the first 4 periods and try to predict whether or not each county is a "
    "Covid-19 hotspot in period 5. To assess the quality of our predictor (classifier), we use the usual techniques "
    "based on the confusion matrix. More precisely, we compute that the area under the ROC is 0.645. This is a decent "
    "result, especially given how difficult the data is to predict.",
    html.Br(),
    html.Br(),
    "For each county, we also plot its prediction category, being True/False Negative/Positive, below. Although it "
    "could be argued that some regions of equal category can be detected, it is hard to say whether or not this is "
    "sufficient to conclude that the difficulty of predicting the class of a county depends on its geographical "
    "location."
])

random_forest_figure_prediction = update_figure_random_forest_prediction_period_5()

########################################################################################################################
#                                             Display everything                                                       #
########################################################################################################################

app.layout = dbc.Container(
    [
        html.Div(children=[html.H1(children='Modern Data Analytics project: Covid data'),
                           html.H2(children='Made by Team Sweden')],
                 style={'textAlign': 'center', 'color': 'black'}),
        html.Hr(),

        # Covid spead visualization
        html.Div(children=[html.H4(children='1. Visualization of the new cases per county on a weekly basis.')],
                 style={'textAlign': 'left', 'color': 'black'}),
        html.Div(Covid_spread_general_text),
        dbc.Row(
            [
                dbc.Col([dbc.Row(visualization_dropdown),
                         html.Div(id='slider-container', children=[slider], style={'display': 'block', 'padding': 10})],
                        md=3),
                dbc.Col(id="id_visualization-display-component", children=visualization_display, md=8)
            ],
            align="center",
        ),
        dbc.Row(
            [
                html.Div(id='visualization-clusters-info', children=[Covid_spread_clusters_text,
                                                                     covid_spread_clusters_votes_correlation_text],
                         style={'display': 'none'})
            ]
        ),
        dbc.Row(
            [
                dcc.Graph(id="visualization-cluster-correlation-figure",
                          figure=covid_spread_clusters_votes_correlation_figure,
                          style={'display': 'none'})
            ]
        ),
        html.Hr(),

        # Infection Rates
        html.Div(children=[html.H4(children='2. Infection rates')],
                 style={'textAlign': 'left', 'color': 'black'}),
        html.Div(infection_rate_general_text),
        dbc.Row(
            [
                dbc.Col([html.Div(children='Select a state'),
                         html.Div(children=[infrates_states_dropdown]),
                         html.Br(),
                         html.Div(children='Select a measure'),
                         html.Div(children=[infrates_measures_dropdown]),
                         dcc.Graph(id="id_state-selected-figure", figure=fig_state_selected)],
                        md=3),
                dbc.Col(dcc.Graph(id="id_infection-rates-figure", figure=fig_inf), md=8)
            ],
            align="center",
        ),
        html.Div(text_inf_svm_1),
        dbc.Row(
            [
                dcc.Graph(id='id_infection-rates-figure-svm', figure=fig_inf_svm)
            ]
        ),
        html.Div(text_inf_svm_2),
        html.Hr(),

        # Waste water analysis
        html.Div(children=[html.H4(children='3. Waste water analysis')],
                 style={'textAlign': 'left', 'color': 'black'}),
        html.Div(waste_water_introduction_text),
        dbc.Row(
            [
                dbc.Col([html.Div(children=["Select a state:",
                                            waste_water_per_county_fips_input])], md=3),
                dbc.Col(dcc.Graph(id="id_waste-water-per-state-figure", figure=fig_waste_water_per_state), md=8)
            ],
            align="center",
        ),
        html.Div(waste_water_prediction_text),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="", figure=fig_waste_water_USA_prediction), md=8)
            ],
            align="center",
            justify="center",
        ),
        html.Hr(),

        # Community detection
        html.Div(children=[html.H4(children='4. Community Detection per period')],
                 style={'textAlign': 'left', 'color': 'black'}),
        html.Div(community_detection_methodology_and_general_results),
        html.Br(),
        html.Div(children=['select a period', dbc.Col(community_detection_dropdown, md=3)]),
        html.Div(id='id_community-detection-general-info', children=[community_detection_general_info]),
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id="id_community-detection-figure-cases", figure=fig_community_detection_cases)],
                        md=5),
                dbc.Col([dcc.Graph(id="id_community-detection-figure-deaths", figure=fig_community_detection_deaths)],
                        md=5)
            ],
            align="center",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div(id='id_community-detection-cases-info', children=[community_detection_cases_info]),
                        md=5),
                dbc.Col(html.Div(id='id_community-detection-deaths-info', children=[community_detection_deaths_info]),
                        md=5)
            ],
            align="top",
            justify="center",
        ),
        html.Hr(),

        # Random forests
        html.Div(children=[html.H4(children="5. Random forest model per period")],
                 style={'textAlign': 'left', 'color': 'black'}),
        html.Div(random_forest_general_info),
        dbc.Row(
            [
                dbc.Col(html.Div(random_forest_dropdown), md=3),
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="id_random-forest-figure-cases", figure=random_forest_figure_cases), md=6),
                dbc.Col(dcc.Graph(id="id_random-forest-figure-deaths", figure=random_forest_figure_deaths), md=6)
            ],
            align="top",
            justify="center",
        ),
        html.Div(random_forest_prediction_text),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="id_random-forest-prediction", figure=random_forest_figure_prediction), md=5)
            ],
            align="top",
            justify="center",
        )
    ],
    fluid=True,
)

if __name__ == '__main__':
    app.run_server(debug=True)
