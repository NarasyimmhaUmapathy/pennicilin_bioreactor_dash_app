import os
import glob
from pathlib import Path

from google.cloud import storage
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
from dash import html, dash_table, callback, Input, Output, State
import pandas as pd
from flask import send_from_directory

from utils.api_client import  get_predictions


# --------------------------------------------------------
#  PROJECT ROOT PATH
# --------------------------------------------------------
project_root = Path("/app")
reports_path = project_root / "reports"

dash.register_page(__name__, path="/data-drift", name="Data Drift Report")

server = dash.get_app().server

if "serve_report" not in server.view_functions:
    @server.route("/reports/<path:filename>")
    def serve_report(filename):
        return send_from_directory(reports_path, filename)

# --------------------------------------------------------
#  Utility: Get available reports
# --------------------------------------------------------
def list_reports():
    files = sorted(glob.glob(str(reports_path / "*.html")), reverse=True)

    if not files:
        return pd.DataFrame(columns=["Report", "Created"])

    data = []
    for f in files:
        data.append(
            {
                "Report": os.path.basename(f),
                "Created": pd.to_datetime(os.path.getmtime(f), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return pd.DataFrame(data)

# --------------------------------------------------------
#  Utility: Get latest report (without generating new)
# --------------------------------------------------------
def get_latest_report():
    files = sorted(glob.glob(str(reports_path / "*.html")), reverse=True)
    return files[0] if files else None


## generate http reports format from gcs URI format for web app rendering
def sign_gcs_uri(gcs_uri: str, minutes: int = 30) -> str:
    # gs://bucket/path/to/file.html -> signed https url
    path = gcs_uri.replace("gs://", "", 1)
    bucket_name, blob_name = path.split("/", 1)

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
    )

# --------------------------------------------------------
#  PAGE LAYOUT
# --------------------------------------------------------
layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "2rem",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1("📊 Data Drift Monitoring", style={"textAlign": "center", "marginBottom": "1rem"}),

        html.Div(
            style={"textAlign": "center", "marginBottom": "2rem"},
            children=[
                html.Button(
                    "🔄 View Most Recent Report",
                    id="load-report-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": "#0052cc",
                        "color": "white",
                        "padding": "10px 22px",
                        "border": "none",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "12px",
                        "boxShadow": "0 3px 6px rgba(0,0,0,0.2)",
                    },
                )
            ],
        ),

        # MAIN REPORT VIEWER
        html.Div(
            id="report-viewer",
            children=[
                html.Iframe(
                    id="report-frame",
                    style={
                        "width": "100%",
                        "height": "650px",
                        "border": "2px solid #ddd",
                        "borderRadius": "10px",
                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                    },
                )
            ],
        ),

        html.Hr(style={"margin": "1.5rem 0"}),

        # COLLAPSIBLE AVAILABLE REPORTS
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "📁 Show available reports",
                        id="toggle-reports-btn",
                        color="secondary",
                        outline=True,
                        className="mb-3",
                    ),
                    width="auto",
                ),
            ],
            justify="start",
        ),

        dbc.Collapse(
            id="reports-collapse",
            is_open=False,
            children=[
                dbc.Card(
                    className="shadow-sm",
                    children=[
                        dbc.CardHeader(html.B("Available Reports")),
                        dbc.CardBody(
                            dash_table.DataTable(
                                id="report-table",
                                columns=[
                                    {"name": "Report", "id": "Report"},
                                    {"name": "Created", "id": "Created"},
                                ],
                                data=list_reports().to_dict("records"),
                                style_table={"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
                                style_cell={"padding": "8px", "textAlign": "left", "fontSize": "12px"},
                                style_header={"fontWeight": "bold"},
                                page_size=10,
                                sort_action="native",
                                filter_action="native",
                            )
                        ),
                    ],
                )
            ],
        ),
    ],
)

# --------------------------------------------------------
#  CALLBACK — Load latest report into Iframe
# --------------------------------------------------------
@callback(
    Output("report-frame", "src"),
    Input("load-report-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_latest_report(n_clicks,batch_number):
    data = get_predictions(batch_number)
    latest_uri = data.get("drift_report_uri")
    if not latest_uri:
        return html.Div("No drift report URI returned yet.")

    signed = sign_gcs_uri(latest_uri, minutes=30)

    return html.Iframe(
        src=signed,
        style={"width": "100%", "height": "80vh", "border": "0", "borderRadius": "12px"},
    )
# --------------------------------------------------------
#  CALLBACK — Toggle collapsible reports table
# --------------------------------------------------------
@callback(
    Output("reports-collapse", "is_open"),
    Output("toggle-reports-btn", "children"),
    Input("toggle-reports-btn", "n_clicks"),
    State("reports-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_reports(n_clicks, is_open):
    new_open = not is_open
    label = "📁 Hide available reports" if new_open else "📁 Show available reports"
    return new_open, label
