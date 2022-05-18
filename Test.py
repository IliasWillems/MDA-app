# Import the necessary packages
import pandas as pd
import plotly.express as px
from urllib.request import urlopen
import json

states = pd.read_csv("Data/state_fips.csv")

period = 2
df = pd.read_csv("Data/CommunityDetection/df_LB_deaths_%s.csv" % period, dtype={'fips': str, 'cluster': str})
df = df.loc[df['state'].shift() != df['state']]
df['fips'] = df.apply(lambda row: int(row['fips'][:2]), axis=1)
df = pd.merge(df, states, left_on='fips', right_on='fips')

fig = px.choropleth(df,
                    locations='Postal Code',
                    color='cluster',
                    color_continuous_scale='spectral_r',
                    hover_name='state',
                    locationmode='USA-states',
                    scope='usa')

periods = ['21/01/2020 - 20/06/2020', '21/06/2020 - 20/12/2020', '21/12/2020 - 20/06/2021',
           '21/06/2021 - 20/12/2021', '21/12/2021 - 11/04/2022']

fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0, "autoexpand": True},
                  title_text="Community detection on deaths for the period <br> " + periods[period-1])

fig.show()
