import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

def feature_card(title, body, href, icon="🔎"):
    return dbc.Card(
        className="shadow-sm rounded-4 h-100",
        children=[
            dbc.CardBody(
                [
                    html.Div(icon, style={"fontSize": "1.4rem"}),
                    html.H5(title, className="mt-2 mb-2"),
                    html.Div(body, className="text-muted"),
                    dbc.Button(
                        "Open",
                        href=href,
                        color="primary",
                        outline=True,
                        className="mt-3",
                    ),
                ]
            )
        ],
    )

layout = dbc.Container(
    fluid=True,
    className="py-4",
    children=[
        # Hero
        dbc.Row(
            className="g-4",
            children=[
                dbc.Col(
                    width=12,
                    lg=8,
                    children=[
                        html.H1("🧪 Penicillin Production Dashboard", className="mb-2"),
                        html.P(
                            [
                                "Prediction of penicillin concentrations during production batch runs. ",
                                "Realtime monitoring of model and data drift metrics. ",
                                "Dockerized inference API, Grafana and Prometheus monitoring services deployed with Cloud Run on Google Cloud Platform",

                            ],
                            className="text-muted lead",
                        ),
                        dbc.Row(
                            className="g-2 mt-2",
                            children=[
                                dbc.Col(
                                    dbc.Button(
                                        "Go to Predictions",
                                        href="/predictions",
                                        color="primary",
                                        className="me-2",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Open Monitoring",
                                        href="/monitoring",
                                        color="secondary",
                                        outline=True,
                                    ),
                                    width="auto",
                                ),
                            ],
                        ),
                        html.Hr(className="my-4"),
                        dbc.Badge("ML Inference", color="primary", className="me-2"),
                        dbc.Badge("Prometheus Metrics", color="secondary", className="me-2"),
                        dbc.Badge("Grafana Dashboards", color="dark", className="me-2"),
                        dbc.Badge("Drift Monitoring", color="info", className="me-2"),
                        dbc.Badge("Explainability (optional)", color="warning"),
                    ],
                ),
                dbc.Col(
                    width=12,
                    lg=4,
                    children=dbc.Card(
                        className="shadow-sm rounded-4",
                        body=True,
                        children=[
                            html.H5("Quick Start", className="mb-2"),
                            html.Ol(
                                [
                                    html.Li("Go to Predictions → select a batch → Run Inference"),
                                    html.Li("Inspect curve + KPIs → identify stop threshold crossing"),
                                    html.Li("Open Monitoring → validate drift/latency/requests in Grafana"),
                                ],
                                className="text-muted mb-0",
                            ),
                        ],
                    ),
                ),
            ],
        ),

        # Main features
        dbc.Row(
            className="g-3 mt-3",
            children=[
                dbc.Col(
                    md=4,
                    children=feature_card(
                        title="Predictions",
                        body=(
                            "Run inference by batch number and view predicted concentration over time. "
                            "Includes smoothing, target bands, and KPI summaries. "
                            "Optional alert when a stop-threshold is predicted to be crossed soon."
                        ),
                        href="/predictions",
                        icon="📈",
                    ),
                ),
                dbc.Col(
                    md=4,
                    children=feature_card(
                        title="API Monitoring",
                        body=(
                            "Live dashboards for RMSE, drift score, and share of drifted features "
                            "(Grafana embed with time range + theme toggle)."
                        ),
                        href="/monitoring",
                        icon="📊",
                    ),
                ),
                dbc.Col(
                    md=4,
                    children=feature_card(
                        title="Data Drift Report",
                        body=(
                            "Open the latest drift report produced during inference "
                            "(dataset drift + per-feature drift scores)."
                        ),
                        href="/data-drift",
                        icon="🧭",
                    ),
                ),
            ],
        ),

        # How to navigate
        dbc.Row(
            children=[
                dbc.Col(
                    width=12,
                    children=dbc.Card(
                        className="shadow-sm rounded-4 mt-4",
                        body=True,
                        children=[
                            html.H4("How to navigate", className="mb-2"),
                            html.Div([html.B("Home: "), "Project overview and shortcuts."], className="text-muted mb-1"),
                            html.Div([html.B("Predictions: "), "Run inference per batch and explore curve + KPIs."], className="text-muted mb-1"),
                            html.Div([html.B("API Monitoring: "), "Grafana dashboards for live operational + model metrics."], className="text-muted mb-1"),
                            html.Div([html.B("Data Drift Report: "), "Latest drift report generated by the inference pipeline."], className="text-muted"),
                        ],
                    ),
                )
            ]
        ),

        # Summary + Next steps (bottom row, summary on the right)
        dbc.Row(
            className="g-3 mt-3",
            children=[
                dbc.Col(
                    md=6,
                    children=dbc.Card(
                        className="shadow-sm rounded-4 mt-4 h-100",
                        body=True,
                        children=[
                            html.H5("Next steps (optional)", className="mb-2"),
                            html.Ul(
                                [
                                    html.Li("Add alert rules for drift + latency thresholds."),
                                    html.Li("Link a Shapash explainability report per batch (optional)."),
                                    html.Li("Move training to Vertex AI pipelines (planned)."),
                                ],
                                className="text-muted mb-0",
                            ),
                        ],
                    ),
                ),
                dbc.Col(
                    md=6,
                    children=dbc.Card(
                        className="shadow-sm rounded-4 mt-4 h-100",
                        body=True,
                        children=[
                            html.H5("Project summary", className="mb-2"),
                            html.Div(
                                className="text-muted",
                                children=[
                                    html.P(
                                        "This app demonstrates an end-to-end, production-style MLOps workflow for monitoring a bioprocess batch run.",
                                        className="mb-2",
                                    ),
                                    html.Ul(
                                        [
                                            html.Li("A trained ML model predicts penicillin concentration over time for a selected batch."),
                                            html.Li("Operational metrics (latency/requests) and model quality metrics (RMSE, drift) are tracked."),
                                            html.Li("Drift reports are generated during inference and made available in the UI, and stored in Cloud Storage."),
                                            html.Li("Services are containerized and deployed on Google Cloud Run, with artifacts stored in Cloud Storage."),
                                            html.Li("The research paper which served as an inspiration for this project can be found here:"),
                                            html.Li("https://www.sciencedirect.com/science/article/pii/S0098135418305106"),

                                        ],
                                        className="mb-2",
                                    ),
                                    html.Div(
                                        "Goal: provide operators fast visibility into when to stop a batch run, while maintaining trust through monitoring.",
                                        className="small",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ),
            ],
        ),
    ],
)
