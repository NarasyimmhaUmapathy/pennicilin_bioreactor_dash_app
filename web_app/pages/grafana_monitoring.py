import os
import requests
from datetime import datetime
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/monitoring")

# Public dashboard URL (Grafana Cloud "public dashboards" link)
PUBLIC_DASHBOARD_URL = os.getenv(
    "GRAFANA_PUBLIC_URL",
    "https://narasyimmha.grafana.net/public-dashboards/c666b2b0aa544d20ae6ca0dddb28ad4b",
)

INFERENCE_URL = os.getenv(
    "INFERENCE_API_URL",
    "https://inference-api-242173489543.europe-west3.run.app",
)

# Keep these so your UI still has the same shape (titles for labels)
DASHBOARDS = {
    "rmse": {"title": "RMSE"},
    "drift_share": {"title": "Drift share"},
    "feature_drift": {"title": "Feature drift"},
}
DEFAULT_TAB = "rmse"

RANGE_OPTIONS = [
    {"label": "Last 5 minutes", "value": "now-5m"},
    {"label": "Last 15 minutes", "value": "now-15m"},
    {"label": "Last 30 minutes", "value": "now-30m"},
    {"label": "Last 1 hour", "value": "now-1h"},
]
DEFAULT_FROM = "now-1h"


def make_public_url() -> str:
    # Public dashboards often ignore extra query params; keep it simple & reliable.
    return PUBLIC_DASHBOARD_URL


layout = dbc.Container(
    fluid=True,
    className="py-4",
    children=[
        # Header + controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("📊 API Monitoring", className="mb-1"),
                        html.Div(
                            "Grafana monitoring dashboard (public link) + live KPI badges from the inference API.",
                            className="text-muted",
                        ),
                    ],
                    width=10,
                    lg=7,
                ),
                dbc.Col(
                    dbc.Card(
                        className="shadow-sm rounded-4",
                        body=True,
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Switch(
                                            id="grafana-theme-switch",
                                            label="Dark mode",
                                            value=True,
                                        ),
                                        width=12,
                                        lg="auto",
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="grafana-dashboard-dropdown",
                                            options=[
                                                {"label": v["title"], "value": k}
                                                for k, v in DASHBOARDS.items()
                                            ],
                                            value=DEFAULT_TAB,
                                            clearable=False,
                                            style={"minWidth": "240px"},
                                        ),
                                        width=12,
                                        lg=True,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="grafana-range-dropdown",
                                            options=RANGE_OPTIONS,
                                            value=DEFAULT_FROM,
                                            clearable=False,
                                            style={"minWidth": "200px"},
                                        ),
                                        width=12,
                                        lg=True,
                                    ),
                                    dbc.Col(
                                        dbc.Button(
                                            "Open Grafana Dashboard",
                                            id="open-grafana-btn",
                                            color="primary",
                                            className="w-100",
                                            external_link=True,
                                            target="_blank",
                                        ),
                                        width=12,
                                        lg="auto",
                                    ),
                                ],
                                className="g-2 align-items-center",
                            ),
                        ],
                    ),
                    width=12,
                    lg=5,
                ),
            ],
            className="g-3 mb-3",
        ),

        # Badges row
        dbc.Row(
            className="g-2 mb-3",
            children=[
                dcc.Interval(id="badge-interval", interval=10_000, n_intervals=0),
                dbc.Col(dbc.Badge(id="badge-rmse", color="secondary", pill=True, className="p-2"), width="auto"),
                dbc.Col(dbc.Badge(id="badge-drift-share", color="secondary", pill=True, className="p-2"), width="auto"),
                dbc.Col(dbc.Badge(id="badge-aeration", color="secondary", pill=True, className="p-2"), width="auto"),
                dbc.Col(dbc.Badge(id="badge-substrate", color="secondary", pill=True, className="p-2"), width="auto"),
                dbc.Col(html.Div(id="badge-updated", className="text-muted small"), width=True),
            ],
        ),

        # Replace iframe with a friendly card
        dbc.Card(
            className="shadow-sm rounded-4",
            children=[
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(html.B("Grafana dashboard")),
                            dbc.Col(html.Div(id="grafana-meta-label", className="text-muted small text-end")),
                        ]
                    )
                ),
                dbc.CardBody(
                    [
                        dbc.Alert(
                            [
                                html.Div("Grafana Cloud public dashboards can’t be embedded in iframes.", className="fw-bold"),
                                html.Div("Use the button above to open the dashboard in a new tab."),
                            ],
                            color="warning",
                            className="rounded-4",
                        ),
                        dbc.Button(
                            "Open Grafana Dashboard",
                            href=make_public_url(),
                            target="_blank",
                            color="primary",
                        ),
                    ]
                ),
            ],
        ),
    ],
)


@callback(
    Output("open-grafana-btn", "href"),
    Output("grafana-meta-label", "children"),
    Input("grafana-theme-switch", "value"),
    Input("grafana-range-dropdown", "value"),
    Input("grafana-dashboard-dropdown", "value"),
)
def update_grafana(dark_mode: bool, from_range: str, dashboard_key: str):
    # We keep these controls for UX consistency, but we always link to the same public URL.
    url = make_public_url()
    theme = "dark" if dark_mode else "light"
    title = DASHBOARDS.get(dashboard_key, {}).get("title", dashboard_key)

    label = f"{title} • Theme: {theme} • Range: {from_range.replace('now-', 'last ')}"
    return url, label


# ------------ badges callback ------------
@callback(
    Output("badge-rmse", "children"),
    Output("badge-rmse", "color"),
    Output("badge-drift-share", "children"),
    Output("badge-drift-share", "color"),
    Output("badge-aeration", "children"),
    Output("badge-aeration", "color"),
    Output("badge-substrate", "children"),
    Output("badge-substrate", "color"),
    Output("badge-updated", "children"),
    Input("badge-interval", "n_intervals"),
)
def refresh_badges(_):
    candidate_paths = ["/latest_monitoring_metrics", "/latest-metrics", "/status"]
    s = None

    for p in candidate_paths:
        try:
            r = requests.get(f"{INFERENCE_URL.rstrip('/')}{p}", timeout=4)
            if r.status_code == 200:
                s = r.json()
                break
        except Exception:
            continue

    if not s:
        return (
            "RMSE: —", "secondary",
            "Drift share: —", "secondary",
            "aeration_rate: —", "secondary",
            "substrate_flow_rate: —", "secondary",
            "Status: unavailable (run inference once)",
        )

    def safe_val(j, key):
        v = j.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    rmse = safe_val(s, "rmse")
    drift_share = safe_val(s, "share_of_drifted_columns")
    aer = safe_val(s, "aeration_drift_score")
    sub = safe_val(s, "substrate_flow_rate_drift_score")
    last_batch = s.get("batch_number")

    def metric_badge(val, fmt, good_lt=None, warn_lt=None):
        if val is None:
            return "—", "secondary"
        v = float(val)
        text = fmt.format(v)
        if (good_lt is not None) and (v <= good_lt):
            return text, "success"
        if (warn_lt is not None) and (v <= warn_lt):
            return text, "warning"
        return text, "danger"

    rmse_txt, rmse_col = metric_badge(rmse, "RMSE: {:.3f}", good_lt=0.25, warn_lt=0.5)
    drift_txt, drift_col = metric_badge(drift_share, "Share of drifted features: {:.2f}", good_lt=0.3, warn_lt=0.5)
    aer_txt, aer_col = metric_badge(aer, "aeration rate drift score: {:.3f}", good_lt=0.25, warn_lt=0.5)
    sub_txt, sub_col = metric_badge(sub, "substrate flow_rate drift score: {:.3f}", good_lt=0.25, warn_lt=0.5)

    updated = s.get("updated_at") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    suffix = f" • last batch: {last_batch}" if last_batch is not None else ""
    updated_txt = f"Updated: {updated}{suffix}"

    return rmse_txt, rmse_col, drift_txt, drift_col, aer_txt, aer_col, sub_txt, sub_col, updated_txt
