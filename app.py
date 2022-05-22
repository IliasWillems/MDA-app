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
cases_and_water_USA = pd.read_csv("Data/cases_and_water_USA.csv")

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
    Output(component_id="id_figure", component_property="figure"),
    [Input(component_id="visualization-slider", component_property="value"),
     Input(component_id="visualization-dropdown", component_property="value")]
)
def update_figure_vis(week, to_display):
    if to_display in ["slider", "animate"]:
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
    else:
        fig = px.choropleth(Kmeans_clusters, geojson=counties, locations='fips', color='cluster',
                            scope='usa', labels={'cluster': 'cases'}
                            )
        fig.update_layout(title_text="Clusters of similar Covid evolution",
                          margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                          width=800)

    return fig


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
    state = agg_week_state.loc[agg_week_state['state'] == state_nbr_int]

    test_start_week = 90
    timesteps = 5

    # Define the pipeline. Note that this pipeline does not include the information about the lag
    pipe = Pipeline([('regressor', SVR(kernel='rbf', gamma=0.5, C=10, epsilon=0.05))])

    pd.options.mode.chained_assignment = None
    state['yhat'] = savgol_filter(state['r'], 21, 3)
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
    gridsearch = GridSearchCV(pipe, params, verbose=0).fit(x_train, y_train)

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
    Output(component_id="id_waste-water-figure", component_property="figure"),
    [Input(component_id="waste-water-reset", component_property="n_clicks")]
)
def update_figure_waste_water(n_clicks):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=cases_and_water_USA['date'], y=cases_and_water_USA['effective_concentration_rolling_average'],
                   name='Concentration Rolling Average'), secondary_y=False

    )
    fig.add_trace(
        go.Scatter(x=cases_and_water_USA['date'], y=cases_and_water_USA['increase_cases'],
                   mode='lines', name='Increase Cases'), secondary_y=True
    )

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Copies / mL of sewage', secondary_y=False)
    fig.update_yaxes(title_text='Cases', secondary_y=True)

    return fig


@app.callback(
    Output(component_id='id_community-detection-figure-cases', component_property='figure'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_figure_community_detection_cases(period):
    df = pd.read_csv("Data/CommunityDetection/df_LB_cases_%s.csv" % period, dtype={'fips': str, 'cluster': str})
    df = df.loc[df['Name'].shift() != df['Name']]

    fig = px.choropleth(df,
                        locations='Postal Code',
                        color='cluster',
                        color_continuous_scale='spectral_r',
                        hover_name='Name',
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
    df = pd.read_csv("Data/CommunityDetection/df_LB_deaths_%s.csv" % period, dtype={'fips': str, 'cluster': str})
    df = df.loc[df['Name'].shift() != df['Name']]

    fig = px.choropleth(df,
                        locations='Postal Code',
                        color='cluster',
                        color_continuous_scale='spectral_r',
                        hover_name='Name',
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
            "For this period, multiple clusters are detected, so we can also try to predict cluster membership for each "
            "state. In order to produce interpretable results, as well as to be able to do variable selection, we use "
            "a logistic regression model. Starting with the full list of covariates displayed above, we end up"
            " selecting ",
            html.I("proportion vaccinated (p = 0.031)"),
            " and ",
            html.I("poverty rate (p = 0.016)"),
            ". ",
            "The significance of ",
            html.I("proportion vaccinated"),
            " does not come as a surprise, as in this time period the effectiveness of the vaccination "
            "should be optimal for the broad public. Comparing cluster 0 and cluster 1, the results indicate "
            "that the probability to belong to cluster 1 is higher for states with a lower vaccination rate"
            " and a higher poverty rate. "
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
            " for states with a larger proportion of uninsured people. ",
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
            "Moreover ",
            html.I("proportion vaccinated"),
            " is borderline insignificant."
        ]),
    ]

    return info[period - 1]


########################################################################################################################
#                                           Covid spread visualization                                                 #
########################################################################################################################
# ToDo: Implement animated visualization

# General text that is always displayed
Covid_spread_general_text = html.Div([
    "In this section, we display the newly reported cases per county and per week. The slider can be used to choose the "
    "week for which the data must be shown.",
    html.Br(),
    html.Br(),
    "In order to get more insight in how Covid-19 evolves throughout the United States, we can try to cluster counties "
    "together based on how they evolve in terms of cases per week. To this end, we applied a K-means clustering "
    "algorithm on the data set. If you’d like to know more about these clusters and how they were obtained, please "
    "select “Clusters” in the dropdown box below (it may take a while to load the map).",
    html.Br()
])

# Extra information about the clusters
Covid_spread_clusters_text = html.Div([
    "In order to apply the K-means algorithm, we first need a measure that can assign a distance to each pair of "
    "counties. Since the aim is to capture the similarities or differences in the evolution of the number of"
    " Covid-19 cases between the"
    " counties, we can represent each county in a 116 dimensional space where each dimension represents the number of"
    " new Covid-19 cases in a specific week for that county. Then, we can define the distance between two counties to be"
    " the Euclidean distance between their representations in that high-dimensional space.",
    html.Br(),
    html.Br(),
    "However, it is well known that K-means clustering suffers from the,",
    html.I("curse of dimensionality"),
    ": it tends to perform worse in high dimensional spaces. Furthermore, when using clustering methods it is always "
    "advisable to work on standardized data. Therefore, the 116 dimensional points where first scaled and then their "
    "dimensionality was reduced using principal component analysis (PCA). The results of applying K-means on these "
    "pre-processed data still left a lot to be desired. It turned out that some outlier counties were throwing off the "
    "clustering algorithm. The solution to this was to add an outlier detector between the scaling and PCA step.",
    html.Br(),
    html.Br(),
    "The process just described contains steps that require the choice of a hyperparameter. More specifically, one "
    "should choose the number of clusters in a K-means clustering algorithm, as well as the number of principal "
    "components to retain in the dimensionality reduction step. The choices of these hyperparameters are not a "
    "priori clear and hence a careful parameter tuning should be performed. Luckily, ",
    html.I("skLearn"),
    " allows to  construct a pipeline that can tune these parameters for us.",
    html.Br(),
    html.Br(),
    "The final result is displayed on the map. Note that the k-means clustering algorithm did not have any geographical "
    "information about the counties. It was able to find these clusters solely based on how Covid-19 evolved throughout "
    "the US.",
    html.Br(),
    html.Br(),
    "Now, a multinomial logistic model is fit to predict the probability to belong to a cluster for each county."
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
                      html.I("Median individual income.")])
    ]),
    html.Br(),
    "The results show that:",
    dbc.Row([
        dbc.Col(
            html.Ul(children=[
                html.Div(
                    children=["1. The probability that a county belongs to cluster 1, compared to its probability to"
                              " belong to cluster 0 is higher if: ",
                              html.Ul(children=[
                                  html.Div(children=["1. ", html.I("Vote Democrat"), " is lower"]),
                                  html.Div(children=["2. ", html.I("pagerank score"), " is higher"]),
                                  html.Div(children=["3. ", html.I("pop_density"), " is lower"]),
                                  html.Div(children=["4. ", html.I("Median age"), " is lower"]),
                                  html.Div(children=["5. ", html.I("Life expectancy"), " is lower"]),
                                  html.Div(children=["6. ", html.I("Uninsured"), " is higher"]),
                                  html.Div(children=["7. ", html.I("Airports"), " is higher"])
                              ])]),
            ]),
            md=5
        ),
        dbc.Col(
            html.Ul(children=[
                html.Div(
                    children=["3. The probability that a county belongs to cluster 3, compared to its probability to"
                              " belong to cluster 0 is higher if: ",
                              html.Ul(children=[
                                  html.Div(children=["1. ", html.I("Vote Republican"), " is lower"]),
                                  html.Div(children=["2. ", html.I("Vote Democrat"), " is lower"]),
                                  html.Div(children=["3. ", html.I("pagerank score"), " is higher"]),
                                  html.Div(children=["4. ", html.I("pop_density"), " is lower"]),
                                  html.Div(children=["5. ", html.I("PovertyRate"), " is higher"]),
                                  html.Div(children=["6. ", html.I("Median age"), " is lower"]),
                                  html.Div(children=["7. ", html.I("Life expectancy"), " is lower"]),
                                  html.Div(children=["8. ", html.I("Uninsured"), " is higher"]),
                                  html.Div(children=["9. ", html.I("Median individual income"), " is higher"])
                              ])]),
            ]),
            md=5,
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.Ul(children=[
                html.Div(
                    children=["2. The probability that a county belongs to cluster 2, compared to its probability to"
                              " belong to cluster 0 is higher if: ",
                              html.Ul(children=[
                                  html.Div(children=["1. ", html.I("Vote Democrat"), " is lower"]),
                                  html.Div(children=["2. ", html.I("pagerank score"), " is higher"]),
                                  html.Div(children=["3. ", html.I("pop_density"), " is lower"]),
                                  html.Div(children=["4. ", html.I("PovertyRate"), " is lower"]),
                                  html.Div(children=["5. ", html.I("Median age"), " is lower"]),
                                  html.Div(children=["6. ", html.I("Uninsured"), " is higher"]),
                              ])])
            ]),
            md=5
        ),
        dbc.Col(
            html.Ul(children=[
                html.Div(
                    children=["4. The probability that a county belongs to cluster 4, compared to its probability to"
                              " belong to cluster 0 is higher if: ",
                              html.Ul(children=[
                                  html.Div(children=["1. ", html.I("Vote Republican"), " is lower"]),
                                  html.Div(children=["2. ", html.I("Vote Democrat"), " is lower"]),
                                  html.Div(children=["3. ", html.I("pop_density"), " is lower"]),
                                  html.Div(children=["4. ", html.I("PovertyRate"), " is higher"]),
                                  html.Div(children=["5. ", html.I("Median age"), " is lower"]),
                                  html.Div(children=["6. ", html.I("Life expectancy"), " is lower"]),
                                  html.Div(children=["7. ", html.I("Uninsured"), " is higher"]),
                                  html.Div(children=["8. ", html.I("Airports"), " is higher"]),
                                  html.Div(children=["9. ", html.I("Median individual income"), " is higher"])
                              ])])
            ]),
            md=5,
        ),
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
    options=[{"label": 'slider', 'value': 'slider'},
             {"label": 'animate (Not implemented yet)', 'value': 'animate'},
             {"label": 'K-means clusters', 'value': 'clusters'}
             ],
    value='slider')

# Create a slider
slider = dcc.Slider(id='visualization-slider',
                    min=1,
                    max=max(week_merge['week']),
                    value=1,
                    marks={str(i): str(i) for i in [np.arange(1, max(week_merge['week']), 10)]})

# Create the figure
fig_visual = update_figure_vis(1, "slider")

########################################################################################################################
#                                                Infection Rates                                                       #
########################################################################################################################
# ToDo: Try to predict these curves with f.e. an SVM. Use Louvain communities as extra predictor.
# General text that is always displayed
infection_rate_general_text = html.Div([
    "In this section, we display the infection number for each week and each state."
    " To this end, the weekly percentage of infections"
    " on a state level is used. Assuming people are infectious for 7 days, the infection number for a certain week"
    " can be computed as"
    " the number of cases in the next week divided by the number of cases in the current week."
    " This number represents the average amount of people that"
    " each Covid patient infects in that week. Also a smoothed curve is added to the plot of the infection numbers.",
    html.Br(),
    html.Br(),
    " Additionally, some Covid measurements are displayed in terms of closing schools and mask obligation.",
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
    value=1)

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

# Create a plot of the support vector machine prediction
fig_inf_svm = update_figure_inf_svm(20)

########################################################################################################################
#                                            Waste water analysis                                                      #
########################################################################################################################

# Create figure reset button
waste_water_reset_button = html.Button('Reset figure', id='waste-water-reset', n_clicks=0)

fig_waste_water = update_figure_waste_water(0)

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
    "In this section, we construct a graph based on the covid data set. More specifically, the nodes represent the "
    "different states and two states are connected if the evolution of Covid-19 in these states follows the same trend. "
    "To this end, we look at correlations between the number of cases in both states on a weekly basis and create an "
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
    "on the predictors listed above. Since the sample size for each of these prediction is small (each state corresponds "
    "to one observation), Firth's logistic regression was also tried to make the model. However, the conclusion were "
    "each time the same, so we will only discuss the more commonly used logistic regression models."
])

# General information related to Covid for that period
community_detection_general_info = update_general_info_community_detection(1)

# Analyses for each of the periods
community_detection_cases_info = update_text_community_detection_cases(1)
community_detection_deaths_info = update_text_community_detection_deaths(1)

########################################################################################################################
#                                             Display everything                                                       #
########################################################################################################################
# ToDo: Write some explanations for each of the sections about what is displayed and what the user can do.

app.layout = dbc.Container(
    [
        html.Div(children=[html.H1(children='Modern Data Analytics project: Covid data'),
                           html.H2(children='Brought to you by the rubber duckies')],
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
                dbc.Col(dcc.Graph(id="id_figure", figure=fig_visual), md=8)
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
        dbc.Row(
            [
                dcc.Graph(id='id_infection-rates-figure-svm', figure=fig_inf_svm)
            ]
        ),
        html.Hr(),

        # Waste water analysis
        html.Div(children=[html.H4(children='3. Waste water analysis')],
                 style={'textAlign': 'left', 'color': 'black'}),
        dbc.Row(
            [
                dbc.Col([html.Div(children=[waste_water_reset_button])], md=3),
                dbc.Col(dcc.Graph(id="id_waste-water-figure", figure=fig_waste_water), md=8)
            ],
            align="center",
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
        )
    ],
    fluid=True,
)

if __name__ == '__main__':
    app.run_server(debug=True)
