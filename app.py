import streamlit as st
import numpy as np

from src.framework import ArgumentationFramework, parse_nodes, parse_edges
from src.sampler import sample_weights, transform_to_acceptability
from src.hull import convex_hull
from src.visuals import fig_1d, fig_2d, fig_3d

st.set_page_config(page_title="Acceptability Degree Space", page_icon="-", layout="wide")

# ---- Titre  ----
st.markdown("<h1 style='text-align:center;'>Acceptability Degree Space</h1>", unsafe_allow_html=True)

# -----------------------
# Sidebar with a FORM
# -----------------------
with st.sidebar:
    st.markdown("## Parameters")

    st.markdown("""
        <style>
        .side-help { font-size: 0.85rem; color: rgba(250,250,250,0.6); margin-top:-8px; }
        </style>
    """, unsafe_allow_html=True)

    with st.form("params", clear_on_submit=False):
        # ---  inputs ---
        nodes_text = st.text_input("Arguments A", value="a, b, c", help="Comma or space separated.")
        edges_text = st.text_area("Attacks R (u v per line)", value="a b\nb a\nc b",
                                  height=96, help="Example: 'a b' means a attacks b.")

        n_samples = st.slider("Number of samples (wᵢ)", 1_000, 100_000, 1_000, step=1_000)
        epsilon = st.number_input("ε (convergence)", min_value=1e-10, max_value=1e-2,
                                  value=1e-6, format="%.1e")
        seed = st.number_input("Random seed", value=0, step=1)
        st.divider()

        # --- Display options ---
        with st.expander("Display options", expanded=True):
            colA, colB = st.columns(2)
            with colA:
                show_points = st.checkbox("Show points", value=True)
            with colB:
                show_hull = st.checkbox("Show convex hull", value=False)

            subsample = st.slider("Subsample for display", 1_000, 50_000, 10_000, step=1_000,
                                  help="Max points rendered (computation still uses n_samples).")
            tol = st.slider("Tolerance for fixed dims (|A|>3)", 0.01, 0.20, 0.05, 0.01)

            if not show_points:
                subsample = 0

        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Update visualization", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)



# -----------------------
# Parse / validate
# -----------------------
A = parse_nodes(nodes_text)
R = parse_edges(edges_text)
if not A:
    st.stop()
af = ArgumentationFramework.from_lists(A, R)
m = len(af.A)

# -----------------------
# recalcule à chaque changement 
# -----------------------
@st.cache_data(show_spinner=False)
def compute_points_cached(A_tuple, R_tuple, m, n_samples, seed, epsilon):
    # rebuild AF inside cache to avoid non-hashable objects
    from src.framework import ArgumentationFramework
    af_local = ArgumentationFramework(A_tuple, R_tuple)
    W = sample_weights(m, n_samples, seed=int(seed))
    X = transform_to_acceptability(af_local, W, epsilon=epsilon)
    return X

# compute only when submitted or first load
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    submitted = True  # compute first time

if submitted:
    X = compute_points_cached(af.A, af.R, m, n_samples, seed, epsilon)
    st.session_state.X = X

if "X" not in st.session_state:
    st.stop()

X = st.session_state.X

# -----------------------
# Build hull once if needed
# -----------------------
hull_global = convex_hull(X) if (show_hull and m >= 2) else None

# -----------------------
# Visualization
# -----------------------

# for display => optional subsample
N = len(X)
if show_points and subsample < N:
    idx = np.random.default_rng(0).choice(N, size=subsample, replace=False)
    X_disp = X[idx]
else:
    X_disp = X

if m == 1:
    st.plotly_chart(fig_1d(X_disp[:, 0], A[0]), use_container_width=True)

elif m == 2:
    st.plotly_chart(
        fig_2d(X_disp[:, :2], (A[0], A[1]), hull_global if show_hull else None, show_points=show_points),
        use_container_width=True
    )

elif m == 3:
    hull_disp = convex_hull(X_disp) if show_hull else None
    st.plotly_chart(
        fig_3d(X_disp[:, :3], (A[0], A[1], A[2]), hull_disp, show_points=show_points),
        use_container_width=True
    )

else:
    cols = st.columns(2)
    with cols[0]:
        axes = st.multiselect("Choose 3 axes to display", A, max_selections=3)
    if len(axes) != 3:
        st.info("Select exactly 3 axes.")
        st.stop()
    fixed = [a for a in A if a not in axes]
    with cols[1]:
        st.markdown("**Fix other arguments (± tolerance)**")
        fixed_vals = {a: st.slider(a, 0.0, 1.0, 0.5, 0.01) for a in fixed}

    mask = np.ones(X.shape[0], dtype=bool)
    for a in fixed:
        j = A.index(a)
        mask &= np.abs(X[:, j] - fixed_vals[a]) <= tol

    Xf = X[mask] if mask.any() else X
    idx_axes = [A.index(a) for a in axes]
    pts = Xf[:, idx_axes]

    st.caption(f"{len(pts)} points shown after filtering.")
    st.plotly_chart(
        fig_3d(pts, tuple(axes), convex_hull(pts) if show_hull else None, show_points=show_points),
        use_container_width=True
    )

