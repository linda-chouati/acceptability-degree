import streamlit as st
import numpy as np

from src.framework import ArgumentationFramework, parse_nodes, parse_edges
from src.sampler import sample_weights, transform_to_acceptability
from src.hull import convex_hull
from src.visuals import fig_1d, fig_2d, fig_3d

st.set_page_config(page_title="Acceptability Degree Space", page_icon="-", layout="wide")

# ---- Title  ----
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

        n_samples = st.slider(
            "Number of samples (wᵢ)",
            1_000, 100_000, 10_000, step=1_000,
            help="Total number of random weight vectors generated and evaluated. "
        )
        epsilon = st.number_input(
            "ε (convergence)",
            min_value=1e-10, max_value=1e-2,
            value=1e-6, format="%.1e",
            help=(
                "Contrôle la précision de la convergence du calcul. "
                "Plus ε est petit → plus la sémantique est précise mais lente à calculer. "
                "Plus ε est grand → calcul plus rapide mais moins précis.\n\n"
                "Typiquement, 1e-6 donne un bon compromis."
            )
        )
        seed = st.number_input("Random seed", value=0, step=1, 
                               help=("Utile pour la reproductibilité des expériences.")
                               )
        st.divider()

        # --- Display options ---
        with st.expander("Display options", expanded=True):
            colA, colB = st.columns(2)
            with colA:
                show_points = st.checkbox("Show points", value=True)
            with colB:
                show_hull = st.checkbox("Show convex hull", value=False)
            subsample = st.slider(
                "Subsample for display",
                1_000, 50_000, 10_000, step=1_000,
                help="Maximum number of points shown in the plot - visualization only "
            )

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
    pts2d = X_disp[:, :2]     # points réellement affichés => après éventuel subsample

    # calcule la hull sur CES points-là, pas sur X complet
    hull_disp = convex_hull(pts2d) if show_hull else None

    st.plotly_chart(
        fig_2d(pts2d, (A[0], A[1]), hull_disp, show_points=show_points),
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

    # --- choix des 3 axes à afficher ---
    with cols[0]:
        axes = st.multiselect("Choose 3 axes to display", A, max_selections=3)
    if len(axes) != 3:
        st.info("Select exactly 3 axes.")
        st.stop()

    # --- sliders pour chaque argument non affiché ---
    fixed = [a for a in A if a not in axes]
    with cols[1]:
        st.markdown("**Fix other arguments**")
        fixed_vals = {a: st.slider(a, 0.0, 1.0, 0.5, 0.01) for a in fixed}

    # --- indices utilitaires ---
    idx_axes = [A.index(a) for a in axes]
    idx_fixed = [A.index(a) for a in fixed]

    if len(idx_fixed) == 0:
        # cas limite: rien à fixer → simple projection en 3D
        pts = X[:, idx_axes]
        caption = f"{len(pts)} points shown (no hidden arguments)."
    else:
        # k-NN dans l'espace des dimensions cachées :
        # on prend les K points les plus proches des valeurs fixées par les sliders
        V = np.array([fixed_vals[a] for a in fixed], dtype=float)   # (p,)
        D = X[:, idx_fixed]                                         # (N,p)
        dist = np.linalg.norm(D - V[None, :], axis=1)               # (N,)

        # Choix de K: assez petit pour que le nuage change réellement,
        # mais borné par le "subsample" d'affichage si activé.
        N = len(X)
        target = (
            subsample
            if (show_points and 'subsample' in locals() and subsample and subsample > 0)
            else 5000
        )
        # max 10% de N, min 500, et jamais plus que target
        K = max(500, min(int(0.10 * N), int(target), N))

        # si target >= N, garde quand même un K < N pour que ça bouge visiblement
        if target >= N:
            K = max(500, min(int(0.20 * N), N))

        nn_idx = np.argpartition(dist, K - 1)[:K]        # indices des K plus proches
        pts = X[nn_idx][:, idx_axes]

    st.plotly_chart(
        fig_3d(pts, tuple(axes), convex_hull(pts) if show_hull else None, show_points=show_points),
        use_container_width=True
    )
