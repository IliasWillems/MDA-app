import dash
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

# Make the app
app = dash.Dash(__name__,
                title='MDA Project',
                external_stylesheets=[dbc.themes.BOOTSTRAP])

########################################################################################################################
#                                            Load all data in advance                                                  #
########################################################################################################################
states = pd.read_csv("Data/state_fips.csv")
week_merge = pd.read_csv("Data/week_merge.csv", dtype={'fips': str})
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
agg_week_state = pd.read_csv("Data/agg_week_state.csv", dtype={'fips': str})
measures = pd.read_csv("Data/measures.csv")


########################################################################################################################
#                                         Define all functions in advance                                              #
########################################################################################################################


@app.callback(
    Output(component_id='slider-container', component_property='style'),
    [Input(component_id='visualization-dropdown', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == 'slider':
        return {'display': 'block', 'padding': 10}
    if visibility_state == 'animate':
        return {'display': 'none', 'padding': 10}


@app.callback(
    Output(component_id="id_figure", component_property="figure"),
    [Input(component_id="visualization-slider", component_property="value")]
)
def update_figure_vis(week):
    fig = px.choropleth(week_merge[week_merge['week'] == week], geojson=counties, locations='fips', color='cpp',
                        color_continuous_scale="Viridis",
                        range_color=(0, 0.5),
                        scope="usa",
                        labels={'cpp': '%new cases <br> (on county level)'}
                        )
    fig.update_layout(title_text="Covid-19 cases for week " + str(week) + ". " + "(" +
                                 dt.datetime.strptime(week_merge.loc[week, 'startOfWeek'], "%Y-%m-%d").strftime(
                                     "%d/%b/%Y")
                                 + " to " +
                                 dt.datetime.strptime(week_merge.loc[week, 'endOfWeek'], "%Y-%m-%d").strftime(
                                     "%d/%b/%Y") + ")",
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

    if state['r'].shape[0] > 21:
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
    cases_and_water_USA = pd.read_csv("Data/cases_and_water_USA.csv")
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
                      title_text="Community detection on cases for the period <br> " + periods[period-1])

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
                      title_text="Community detection on deaths for the period <br> " + periods[period-1])

    return fig


@app.callback(
    Output(component_id='id_community-detection-general-info', component_property='children'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_general_info_community_detection(period):
    info = ["During the first period of the pandemic, Covid-19 was some new and frightening virus. Many states responded to"
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
    return info[period-1]


@app.callback(
    Output(component_id='id_community-detection-cases-info', component_property='children'),
    [Input(component_id='community-detection-periods', component_property='value')]
)
def update_text_community_detection_cases(period):
    info = ["For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
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
                "should be optimal for the broad public."
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
    info = ["For this period, there is only one large cluster. Therefore, it would not make sense to predict for each state"
            " the cluster it belongs to.",

            # Period 2
            html.Div([
                "For this period, there are two large clusters detected. After a backwards variable selection it is "
                "determined that only the variable ",
                html.I("Uninsured"),
                " is significant in predicting the cluster for each state ",
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
                " is significant ",
                html.I("(p = 0.010)"),
                ". Unlike before, the proportion of people who are uninsured is no longer significant."
            ]),

            # Period 5
            html.Div([
                "For the last period, again 2 large clusters are detected. ",
                html.I("Uninsured"),
                " is a significant variable, ",
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
    "In this section, we display the newly reported cases per county and per week."
    " The slider can be used to choose the week for which the data must be shown.",
    html.Br()
])

# Create a dropdown for options 'animate' and 'slider'
visualization_dropdown = dcc.Dropdown(
    id='visualization-dropdown',
    options=[{"label": 'slider', 'value': 'slider'},
             {"label": 'animate (Not implemented yet)', 'value': 'animate'},
             ],
    value='slider')

# Create a slider
slider = dcc.Slider(id='visualization-slider',
                    min=1,
                    max=max(week_merge['week']),
                    value=1,
                    marks={str(i): str(i) for i in [np.arange(1, max(week_merge['week']), 10)]})

# Create the figure
fig_visual = update_figure_vis(1)

########################################################################################################################
#                                                Infection Rates                                                       #
########################################################################################################################
# ToDo: Try to predict these curves with f.e. an SVM. Use Louvain communities as extra predictor.
# General text that is always displayed
infection_rate_general_text = html.Div([
    "In this section, we display the infection number for each week and each state."
    " Therefore, the weekly percentage of infections"
    " on a state level is used. Assuming people are infectious for 7 days, the infection number for a certain week"
    " can be computed as"
    " the number of cases in the next week divided by the number of cases in this week."
    " This number represents the average amount of people that"
    " each Covid patient infects in that week. Also a smoothed curve is added to the plot of the infection numbers.",
    html.Br(),
    html.Br(),
    " Besides, some Covid measurements are displayes in terms of closing schools and mask obligation.",
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
                html.Div(children=["1. Orange = Masks obligatory in all public spaces where "
                                   "social distancing is not possible "]),
                html.Div(children=["2. Blue = Masks obligatory in all public spaces where social"
                                   " distancing is not possible"])
        ]),
        html.Div(children=["3. Schools"]),
            html.Ul(children=[
                html.Div(children=["1. Brown = At least some types of schools need to close"]),
                html.Div(children=["2. Yellow = No such restrictions"])
        ]),
    ]),
    html.Br(),
    "The state and Covid measures that are displayed can be chosen in the dropdown box."
])

# Create dropdown for state
infrates_states_dropdown = dcc.Dropdown(
    id='infrates-states',
    options=[{"label":x,"value":y} for x,y in zip(states.Name,states.fips)],
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
        html.Div(children=["5. The degree centrality of the state based on commuting flows between states, referred to as ",
                           html.I("Degree_centrality_com.")])
    ]),
    html.Br(),
    "When we do not make a subdivision of time periods and just cluster states over the whole period under observation "
    "(21/01/2020 - 11/04/2022) based on the evolution of Covid related deaths in each state, we can also try to predict "
    "the resulting clusters. Using a logistic regression model, it can be determined that the only significant variable "
    "in this prediction is ",
    html.I("Uninsured, p = 0.004"),
    ". The poverty rate turned out to be borderline insignificant. An analysis for the clusters based on the cases is "
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
