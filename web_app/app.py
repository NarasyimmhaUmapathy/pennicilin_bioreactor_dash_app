import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import os


# Create Dash app with multi-page routing enabled
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],  # Beautiful clean UI
    suppress_callback_exceptions=True,
)

app.title = "Penicillin Dashboard"



# ---------------------------
# Main App Layout
# ---------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        # NAVBAR
        dbc.NavbarSimple(
            brand="Penicillin Production Dashboard",
            color="primary",
            dark=True,
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Predictions", href="/predictions")),
                dbc.NavItem(dbc.NavLink("API Monitoring", href="/monitoring")),
                dbc.NavItem(dbc.NavLink("Data Drift Report", href="/data-drift")),
            ],
        ),

        # Main content always rendered here
        dash.page_container
    ],
)

server = app.server  # Used if deployed behind gunicorn or a flask server



if __name__ == "__main__":
    #port = int(os.environ.get("PORT", "8050"))  # 8050 locally, 8080 on Cloud Run

    app.run(host="0.0.0.0", port=8080, debug=True)
