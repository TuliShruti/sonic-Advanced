import numpy as np
import plotly.graph_objects as go


def _to_meters(depth_arr):
    """
    Convert raw DLIS depth (0.1-inch units) to meters.
    Notebook formula: depth_m = depth_raw * 0.00254
    (1 unit = 0.1 inch = 0.00254 m)
    """
    return depth_arr * 0.00254


def plot_stc_heatmap(spr, depth_arr, slowness_axis, title="STC Semblance"):
    """
    STC heatmap:
        X → slowness (µs/ft)
        Y → depth (m)
        Color → semblance intensity
    """
    depth_m = _to_meters(depth_arr)

    fig = go.Figure(
        go.Heatmap(
            z=spr,
            x=slowness_axis,
            y=depth_m,
            colorscale="Jet",
            colorbar=dict(title="Semblance"),
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Slowness (µs/ft)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),
        height=700,
        margin=dict(l=60, r=40, t=50, b=50),
    )

    return fig


def plot_stc_with_picks(spr, depth_arr, slowness_axis, picked_slowness, title="STC + Picks"):
    """
    STC heatmap with the picked slowness curve overlaid in white.
    """
    fig     = plot_stc_heatmap(spr, depth_arr, slowness_axis, title)
    depth_m = _to_meters(depth_arr)

    fig.add_trace(
        go.Scatter(
            x=picked_slowness,
            y=depth_m,
            mode="lines",
            line=dict(color="white", width=1.5),
            name="Picked slowness",
        )
    )

    return fig