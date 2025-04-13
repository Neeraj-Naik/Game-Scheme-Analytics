import dash
from dash import html

dash.register_page(__name__, path="/type1")

layout = html.Div([
    html.H2("Type 1 Games"),
    html.P("This page will show info related to Type 1 games."),
], style={"padding": "20px", "height": "100%", "width": "100%"})
