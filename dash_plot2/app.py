# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ada
#     language: python
#     name: python3
# ---

#opend the file
import numpy as np
import pandas as pd
import os
import base64
from IPython.display import display, HTML
import json
#For interactive plots
import plotly.graph_objects as go #pip install plotly
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output #pip install dash-bootstrap-components

franchise_revenue = pd.read_csv('data/franchise_revenue.csv')



# Initialize the Dash app
app = dash.Dash(__name__)


app.layout = html.Div([
    html.P("Select collection:"),
    dcc.Dropdown(
        id="dropdown",
        #xaxis=dict(showgrid=True, gridcolor='lightgray'),
        options=[{'label': name, 'value': name} for name in franchise_revenue['collection_name'].unique()],
        value='Star Wars Collection',
        #template="plotly_white",
        #font=dict(family='JetBrains Mono'),
        #paper_bgcolor='rgba(0,0,0,0)', # to make the background transparent
        #plot_bgcolor='rgba(0,0,0,0)',
        #autosize=True,
        clearable=False,
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))
def display_graph(collection_name):
    filtered_df = franchise_revenue[franchise_revenue['collection_name'] == collection_name].sort_values(by='release_year')
    filtered_df['release_year'] = filtered_df.loc[:,'release_year'].astype(int)
    # Melt the DataFrame to have budget and revenue in a single column
    melted_df = filtered_df.melt(id_vars=['release_year', 'Movie name'], value_vars=['real_budget','real_revenue'], 
                                 var_name='Value', value_name='Amount [$]')
    # Create an interactive plot using Plotly
    fig = px.bar(melted_df, x='release_year', y='Amount [$]', color='Value', barmode="group", color_discrete_map={'real_budget': 'skyblue', 'real_revenue': '#2471ff'})
    # Customize the layout
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=filtered_df['release_year'], ticktext=[f'{cat}\n{sub}' for cat,sub in zip(filtered_df['release_year'], filtered_df['Movie name'])]),
        xaxis_title=filtered_df["collection_name"].iloc[0],
        template="plotly_white",
        font=dict(family='JetBrains Mono'),
        paper_bgcolor='rgba(0,0,0,0)', # to make the background transparent
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True,
        hoverlabel=dict(
        font=dict(
            family="JetBrains Mono",  # Font family
            size=8,  # Font size
            color="#2471ff",  # Font color
        ),
        bgcolor="rgba(255, 255, 255, 0.7)",  # Background color of the hover label
        bordercolor="black",  # Border color for the hover label
        #borderwidth=1  # Border width
    )
    )

    return fig


# Run the Dash app on a different port
if __name__ == '__main__':
    app.run_server(debug=True, port=8099)

# %%


# %%
