import dash
from dash import html

dash.register_page(__name__, path="/")

layout = html.Div([
    html.H1("Welcome to the Game Dashboard"),
    html.P("Use the navigation bar to view different types of games."),
], style={"padding": "20px", "height": "100%", "width": "100%"})
