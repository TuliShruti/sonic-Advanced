"""Plotting helpers for waveform views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_waveform(wf_depth: np.ndarray, title: str = "Waveform") -> go.Figure:
    """Render one or more receiver traces as a Plotly line chart."""
    if wf_depth.ndim == 1:
        wf_depth = wf_depth[np.newaxis, :]

    fig = go.Figure()

    for i in range(wf_depth.shape[0]):
        trace = wf_depth[i]
        trace = trace / (np.max(np.abs(trace)) + 1e-6)

        fig.add_trace(
            go.Scatter(
                x=np.arange(trace.shape[0]),
                y=trace + i,
                mode="lines",
                name=f"R{i + 1}",
                line=dict(width=1.2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time samples",
        yaxis_title="Receiver",
        showlegend=False,
        height=420,
        margin=dict(l=50, r=20, t=50, b=50),
    )

    return fig
