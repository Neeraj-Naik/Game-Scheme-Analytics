import dash
from dash import html

dash.register_page(__name__, path="/type2")

layout = html.Div([
    html.H2("Type 2 Games"),
    html.P("This page will show info related to Type 2 games."),
], style={"padding": "20px", "height": "100%", "width": "100%"})
