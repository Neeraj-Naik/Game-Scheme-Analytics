import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Introduction", href="/")),
            dbc.NavItem(dbc.NavLink("Type 1 Games", href="/type1")),
            dbc.NavItem(dbc.NavLink("Type 2 Games", href="/type2")),
        ],
        brand=html.Span("Game Scheme Analytics", style={"fontSize": "25px", "fontWeight": "bold"}),
        color="primary",
        dark=True,
    ),
    html.Div(dash.page_container, style={
        "height": "100%",
        "width": "100%",
        "padding": "0",
        "margin": "0",
        "flex": "1",
        "overflow": "auto"
    })
], style={
    "display": "flex",
    "flexDirection": "column",
    "height": "100vh",
    "margin": "0",
    "padding": "0"
})

if __name__ == "__main__":
    app.run_server(debug=True)
