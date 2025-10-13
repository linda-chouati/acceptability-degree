import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
import plotly.graph_objs as go

# ---------- helpers ----------
def _kde_2d(points2d: np.ndarray, nbins: int = 60, bandwidth: float = 0.08):
    x = points2d[:, 0]; y = points2d[:, 1]
    X = np.linspace(x.min(), x.max(), nbins)
    Y = np.linspace(y.min(), y.max(), nbins)
    Xg, Yg = np.meshgrid(X, Y)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(points2d)
    Z = np.exp(kde.score_samples(grid)).reshape(nbins, nbins)
    return Xg, Yg, Z

# ---------- 1D ----------
def fig_1d(values: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure([go.Histogram(x=values, nbinsx=40)])
    fig.update_layout(
        template="plotly_dark",
        title=f"Acceptability space (1D) — {label}",
        xaxis_title=label, yaxis_title="count",
        bargap=0.02, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------- 2D ----------
def fig_2d(points2d: np.ndarray, labels: Tuple[str, str], hull: Optional[ConvexHull],
           show_points: bool = True) -> go.Figure:
    Xg, Yg, Z = _kde_2d(points2d)
    traces = [go.Contour(
        x=Xg[0], y=Yg[:, 0], z=Z, contours_coloring="heatmap", showscale=True, name="KDE"
    )]
    if show_points:
        traces.append(go.Scatter(
            x=points2d[:, 0], y=points2d[:, 1],
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            name="points"
        ))
    if hull is not None:
        cyc = list(hull.vertices) + [hull.vertices[0]]
        traces.append(go.Scatter(
            x=points2d[cyc, 0], y=points2d[cyc, 1],
            mode="lines", name="convex hull"
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        template="plotly_dark",
        title="Acceptability space (2D)",
        xaxis_title=labels[0], yaxis_title=labels[1],
        xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1]),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------- 3D ----------
def fig_3d(points3d: np.ndarray, labels: tuple[str, str, str],
           hull: Optional[ConvexHull], show_points: bool = True) -> go.Figure:
    """Affiche un nuage 3D avec enveloppe convexe, fond blanc lisible."""
    fig = go.Figure()

    # --- Enveloppe convexe ---
    if hull is not None and len(points3d) >= 4:
        i, j, k = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]
        fig.add_trace(go.Mesh3d(
            x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2],
            i=i, j=j, k=k,
            opacity=0.35,
            color="rgba(65,105,225,0.45)", 
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3),
            lightposition=dict(x=60, y=100, z=80),
            name="convex hull",
            showscale=False,
            hoverinfo="none"  
        ))

    # --- Points (dessinés apres donc au dessus du hull) ---
    if show_points:
        fig.add_trace(go.Scatter3d(
            x=points3d[:, 0], y=points3d[:, 1], z=points3d[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                opacity=0.9,
                color="rgba(65,105,225,0.95)", 
            ),
            name="points",
            hovertemplate=(
                f"{labels[0]}: %{{x:.3f}}<br>"
                f"{labels[1]}: %{{y:.3f}}<br>"
                f"{labels[2]}: %{{z:.3f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_white",
        title="Acceptability space (3D)",
        paper_bgcolor="white",
        font=dict(color="#111"),
        hovermode="closest", 
        scene=dict(
            bgcolor="white",
            xaxis=dict(
                title=dict(text=labels[0], font=dict(size=15, color="#111")),
                tickfont=dict(color="#111"),
                showline=False, zeroline=False,
                showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1.1,
                showspikes=False,
            ),
            yaxis=dict(
                title=dict(text=labels[1], font=dict(size=15, color="#111")),
                tickfont=dict(color="#111"),
                showline=False, zeroline=False,
                showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1.1,
                showspikes=False,
            ),
            zaxis=dict(
                title=dict(text=labels[2], font=dict(size=15, color="#111")),
                tickfont=dict(color="#111"),
                showline=False, zeroline=False,
                showgrid=True, gridcolor="rgba(0,0,0,0.12)", gridwidth=1.1,
                showspikes=False,
            ),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(color="#111")),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
