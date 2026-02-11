import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import root_mean_squared_error

from utils.api_client import  get_predictions
from utils.plotting import build_prediction_figure_finance

from loguru import logger


dash.register_page(__name__, path="/predictions")


# ---------- UI helpers ----------
def kpi_card(title, value, subtitle=None, color="light"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.Div(value, className="h4 mb-0"),
                html.Div(subtitle, className="text-muted small") if subtitle else None,
            ]
        ),
        className="shadow-sm rounded-4",
        color=color,
    )


def _empty_kpis():
    return [
        dbc.Col(kpi_card("Last value", "—", "latest predicted concentration"), md=3),
        dbc.Col(kpi_card("Min / Max", "—", "range"), md=3),
        dbc.Col(kpi_card("Mean", "—", "average"), md=3),
        dbc.Col(kpi_card("Points", "—", "time steps"), md=3),
    ]


# ---------- Figure builder (modern look) ----------
def build_prediction_figure(
    x_minutes,
    y_pred,
    *,
    show_markers: bool = True,
    smooth: bool = False,
    smooth_window: int = 5,
    target_low=None,
    target_high=None,
    batch_number=None,
    rmse=None,
):
    y = np.asarray(y_pred, dtype=float)
    x = np.asarray(x_minutes, dtype=float)

    fig = go.Figure()

    # Optional target band shading
    if target_low is not None and target_high is not None:
        try:
            lo = float(target_low)
            hi = float(target_high)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=float(x.min()),
                    x1=float(x.max()),
                    y0=lo,
                    y1=hi,
                    fillcolor="rgba(0,0,0,0.06)",
                    line={"width": 0},
                    layer="below",
                )
        except Exception:
            pass  # if user enters weird values, just skip the band

    # Main trace
    mode = "lines+markers" if show_markers else "lines"
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode=mode,
            name="Prediction",
            line={"width": 2},
            marker={"size": 5} if show_markers else None,
            hovertemplate="t=%{x:.0f} min<br>ŷ=%{y:.3f}<extra></extra>",
        )
    )

    # Optional rolling mean overlay (honest smoothing)
    if smooth and smooth_window and smooth_window > 1 and len(y) >= smooth_window:
        w = int(smooth_window)
        kernel = np.ones(w) / w
        y_smooth = np.convolve(y, kernel, mode="same")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_smooth,
                mode="lines",
                name=f"Rolling mean ({w})",
                line={"width": 3, "dash": "dot"},
                hovertemplate="t=%{x:.0f} min<br>ŷ_smooth=%{y:.3f}<extra></extra>",
            )
        )

    title_parts = ["Prediction curve"]
    if batch_number is not None:
        title_parts.append(f"batch={batch_number}")
    if rmse is not None:
        try:
            title_parts.append(f"RMSE={float(rmse):.4f}")
        except Exception:
            pass

    fig.update_layout(
        title=" • ".join(title_parts),
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=55, b=25),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title="Time (hours)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    )
    fig.update_yaxes(
        title="Predicted concentration",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    )

    return fig


# ---------- Layout ----------
layout = dbc.Container(
    fluid=True,
    className="py-4",
    children=[
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("🔬 Penicillin Concentration Prediction", className="mb-1"),
                        html.Div(
                            "Pennicilin concentration prediction based on incoming raw data on 12 minute intervals.",
                            "Predictions and model metrics stored in Cloud Storage Bucket after every inference run, for later analysis and reporting",
                            className="text-muted mb-4",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                # Controls
                dbc.Col(
                    [
                        dbc.Card(
                            className="shadow-sm rounded-4",
                            children=[
                                dbc.CardHeader(html.B("Controls")),
                                dbc.CardBody(
                                    [
                                        html.Label("Batch Number", className="small text-muted"),
                                        dcc.Dropdown(
                                            id="predictions-batch-dropdown",
                                            options=[{"label": f"Batch {i}", "value": i} for i in range(1, 81)],
                                            placeholder="Choose a batch...",
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Run Inference",
                                            id="predictions-run-btn",
                                            color="primary",
                                            className="w-100 mb-2",
                                            n_clicks=0,
                                        ),
                                        dbc.Button(
                                            "Download CSV",
                                            id="predictions-download-btn",
                                            color="secondary",
                                            outline=True,
                                            className="w-100 mb-3",
                                        ),
                                        dcc.Download(id="predictions-download"),
                                        html.Hr(),
                                        dbc.Checklist(
                                            options=[{"label": "Show markers", "value": "markers"}],
                                            value=["markers"],
                                            id="predictions-marker-toggle",
                                            switch=True,
                                            className="mb-2",
                                        ),
                                        dbc.Checklist(
                                            options=[{"label": "Smooth (rolling mean)", "value": "smooth"}],
                                            value=[],
                                            id="predictions-smooth-toggle",
                                            switch=True,
                                            className="mb-2",
                                        ),
                                        html.Label("Smoothing window (# points)", className="small text-muted"),
                                        dcc.Slider(
                                            id="predictions-smooth-window",
                                            min=1,
                                            max=25,
                                            step=1,
                                            value=5,
                                            marks={1: "1", 5: "5", 10: "10", 20: "20", 25: "25"},
                                            tooltip={"placement": "bottom"},
                                            className="mb-3",
                                        ),
                                        html.Label("Target band (optional)", className="small text-muted"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Input(
                                                        id="predictions-target-low",
                                                        type="number",
                                                        placeholder="Low",
                                                    ),
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Input(
                                                        id="predictions-target-high",
                                                        type="number",
                                                        placeholder="High",
                                                    ),
                                                    width=6,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                    ]
                                ),
                            ],
                        )
                    ],
                    width=12,
                    lg=3,
                ),
                # Chart + KPIs
                dbc.Col(
                    [
                        dbc.Row(
                            _empty_kpis(),
                            className="g-3 mb-3",
                            id="predictions-kpi-row",
                        ),
                        dbc.Alert(
                            id="predictions-run-status",
                            color="info",
                            is_open=False,
                            className="shadow-sm rounded-4",
                        ),
                        dbc.Card(
                            className="shadow-sm rounded-4",
                            children=[
                                dbc.CardHeader(
                                    dbc.Row(
                                        [
                                            dbc.Col(html.B("Prediction curve")),
                                            dbc.Col(
                                                html.Div(
                                                    id="predictions-last-updated",
                                                    className="text-muted small text-end",
                                                )
                                            ),
                                        ]
                                    )
                                ),
                                dbc.CardBody(
                                    dcc.Loading(
                                        type="circle",
                                        children=dcc.Graph(
                                            id="prediction-graph",
                                            config={"displayModeBar": True, "scrollZoom": True},
                                            style={"height": "520px"},
                                        ),
                                    )
                                ),
                            ],
                        ),
                    ],
                    width=12,
                    lg=9,
                ),
            ],
            className="g-3",
        ),
        dcc.Store(id="predictions-store"),

    ],
)


# ---------- Callback ----------
@callback(
    Output("prediction-graph", "figure"),
    Output("predictions-run-status", "children"),
    Output("predictions-run-status", "color"),
    Output("predictions-run-status", "is_open"),
    Output("predictions-last-updated", "children"),
    Output("predictions-store", "data"),
    Output("predictions-kpi-row", "children"),
    Input("predictions-run-btn", "n_clicks"),
    State("predictions-batch-dropdown", "value"),
    State("predictions-marker-toggle", "value"),
    State("predictions-smooth-toggle", "value"),
    State("predictions-smooth-window", "value"),
    State("predictions-target-low", "value"),
    State("predictions-target-high", "value"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, batch_number, marker_toggle, smooth_toggle, smooth_window, target_low, target_high):
    empty_fig = go.Figure()
    empty_kpis = _empty_kpis()

    def pack(fig=empty_fig, msg="", color="info", open_=False, updated="", store=None, kpis=empty_kpis):
        return fig, msg, color, open_, updated, store, kpis

    if batch_number is None:
        return pack(msg="⚠️ Please select a batch number.", color="warning", open_=True)

    try:
        data = get_predictions(batch_number)
        logger.info(f"getting predictions for batch number {batch_number}")
        preds = data.get("predictions_array") or []
        if len(preds) == 0:
            logger.info(f"getting predictions for batch number {batch_number}")
            return pack(msg="⚠️ API returned zero predictions.", color="warning", open_=True)

        y = [float(v) for v in preds]
        x_axis = np.arange(0.2, 240, 0.2)

        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x_axis, dtype=float)

        STOP_THRESHOLD = float(23.0)

        # first index where prediction >= threshold
        cross_idx = np.where(y_arr >= STOP_THRESHOLD)[0]

        if len(cross_idx) > 0:
            i0 = int(cross_idx[0])
            t_cross = float(x_arr[i0])  # hours
            y_cross = float(y_arr[i0])
            cross_text = f"{t_cross:.1f} h"
            cross_sub = f"ŷ={y_cross:.3f} ≥ {STOP_THRESHOLD}"
            cross_color = "warning"  # highlight
        else:
            cross_text = "—"
            cross_sub = f"Not reached (≥ {STOP_THRESHOLD})"
            cross_color = "light"

        show_markers = (marker_toggle is not None and "markers" in marker_toggle)
        do_smooth = (smooth_toggle is not None and "smooth" in smooth_toggle)
        smooth_window = int(smooth_window or 1)

        # RMSE key name may vary in your API response
        rmse = data.get("root mean squared error score")
        logger.info("getting rmse score from api",{rmse})

        logger.info("building prediction plot")
        fig = build_prediction_figure(
            x_axis,
            y,
            show_markers=show_markers,
            smooth=do_smooth,
            smooth_window=int(smooth_window or 1),
            target_low=target_low,
            target_high=target_high,
            batch_number=batch_number,
            rmse=rmse,
        )

        fig.update_xaxes(
            title="Time (hours)",
            type="linear",
            range=[float(x_axis[0]), float(x_axis[-1])],
            dtick=10.0,  # 12 minutes = 0.2 hours
            tickformat='.1f',
            tickangle=-45,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            automargin=True,
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        )

        fig.add_hline(
            y=STOP_THRESHOLD,
            line_width=1,
            line_dash="dot",
            line_color="rgba(0,0,0,0.25)",
            annotation_text=f"Threshold {STOP_THRESHOLD}",
            annotation_position="bottom right",
        )


        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store = {
            "batch": int(batch_number),
            "time_hours": [float(t) for t in x_axis],
            "prediction": y,
        }

        logger.info("setting kpi cards")
        kpis = [
            dbc.Col(kpi_card("Last value", f"{y[-1]:.3f}", "latest predicted concentration"), md=3),
            dbc.Col(kpi_card("Min / Max", f"{min(y):.3f} / {max(y):.3f}", "range"), md=3),
            dbc.Col(kpi_card("Stop time, threshold concentration", cross_text, cross_sub, color=cross_color), md=3),
            dbc.Col(kpi_card("Points", f"{len(y)}", f"0 → {x_axis[-1]} timestamp"), md=3),
        ]

        msg = f"✅ Batch {batch_number} • points={len(y)}"
        if rmse is not None:
            try:
                msg += f" • RMSE={float(rmse):.4f}"
            except Exception:
                pass

        return pack(
            fig=fig,
            msg=msg,
            color="success",
            open_=True,
            updated=f"Last updated: {now}",
            store=store,
            kpis=kpis,
        )

    except Exception as e:
        return pack(msg=f"❌ Error: {e}", color="danger", open_=True)
