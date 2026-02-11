import numpy as np
import plotly.graph_objects as go

def build_prediction_figure_finance(
    x_datetimes,
    y_pred,
    *,
    show_markers: bool = False,
    smooth: bool = False,
    smooth_window: int = 5,
    target_low=None,
    target_high=None,
    batch_number=None,
    rmse=None,
):
    y = np.asarray(y_pred, dtype=float)
    x = list(x_datetimes)  # keep datetimes as-is (no np.asarray(..., dtype=float))

    fig = go.Figure()

    # Optional target band (very subtle)
    if target_low is not None and target_high is not None:
        try:
            lo = float(target_low)
            hi = float(target_high)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                fig.add_hrect(
                    y0=lo, y1=hi,
                    fillcolor="rgba(0,0,0,0.05)",
                    line_width=0,
                    layer="below",
                )
        except Exception:
            pass

    mode = "lines+markers" if show_markers else "lines"
    fig.add_trace(go.Scattergl(
        x=x,
        y=y,
        mode=mode,
        name="Prediction",
        line={"width": 1},
        marker={"size": 4} if show_markers else None,
        fill="tozeroy",
        fillcolor="rgba(0,0,0,0.06)",
        hovertemplate="%{x|%H:%M}<br>ŷ=%{y:.3f}<extra></extra>",
    ))

    # Rolling mean overlay
    if smooth and smooth_window and smooth_window > 1 and len(y) >= smooth_window:
        w = int(smooth_window)
        kernel = np.ones(w) / w
        y_smooth = np.convolve(y, kernel, mode="same")

        fig.add_trace(go.Scatter(
            x=x,
            y=y_smooth,
            mode="lines",
            name=f"Rolling mean ({w})",
            line={"width": 2, "dash": "dot"},
            hovertemplate="%{x|%H:%M}<br>ŷ_smooth=%{y:.3f}<extra></extra>",
        ))

    title_parts = ["Batch prediction"]
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
        margin=dict(l=18, r=18, t=55, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )



    fig.update_yaxes(
        title="Predicted concentration",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)",
        zeroline=False,
    )

    return fig
