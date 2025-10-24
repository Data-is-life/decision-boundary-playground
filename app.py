
"""
Decision Boundary Playground — Streamlit (Python) v2
Adds: Iris, Titanic, Breast Cancer, Penguins, Wine datasets + CSV upload,
"Try this!" callouts, multiclass-safe metrics, banners.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional seaborn for some datasets
import seaborn as sns

st.set_page_config(page_title="Decision Boundary Playground", page_icon="🧠", layout="wide")

# Top banner
st.markdown(
    '<p align="center"><img src="assets/banner_header.png" width="80%" alt="Decision Boundary Playground banner"/></p>',
    unsafe_allow_html=True
)

# ------------------------------------
# Header & quick intro
# ------------------------------------
st.title("🧠 Decision Boundary Playground")
st.markdown(
    """
Explore how different **classification models** cut the 2D feature space into decision regions.
- Pick a **dataset** (blobs, moons, circles, XOR, Iris, Titanic, Penguins, Wine, Breast Cancer), adjust **noise** and **class balance**.
- Toggle **feature engineering** (polynomial features, standardization).
- Choose a **model** and tweak its hyperparameters.
- See the **decision surface**, test **metrics**, **ROC curve** (binary only), and **compare models**.

💡 **Try this!**
- `XOR` + `Logistic Regression` + enable **Polynomial features (degree 3)** → see non-linear separation emerge.
- `Moons` + `SVM (RBF)` + low noise → clean curvy boundary.
- `Blobs` + `Random Forest` → blocky, stable partitions.
- `Titanic` + `Random Forest` → inspect precision/recall under imbalance.
- `Iris` + `Logistic Regression` → textbook multi-class separation.
- `Penguins` + `KNN (k=7)` → distinct species clusters from two measurements.
"""
)

# ------------------------------------
# Synthetic datasets
# ------------------------------------
def make_xor(n=400, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    X = np.vstack([x1 + noise * rng.normal(size=n), x2 + noise * rng.normal(size=n)]).T
    y = (np.sign(x1) * np.sign(x2) < 0).astype(int)  # XOR pattern
    return X, y

@dataclass
class DatasetSpec:
    name: str
    maker: Callable[[int, float, int], Tuple[np.ndarray, np.ndarray]]
    supports_noise: bool = True
    default_noise: float = 0.25
    is_multiclass: bool = False
    note: str = ""

DATASETS: Dict[str, DatasetSpec] = {
    "Blobs": DatasetSpec(
        name="Blobs",
        maker=lambda n, noise, seed: make_blobs(n_samples=n, centers=[(-1,-1),(1,1)],
                                                cluster_std=max(0.05, noise), random_state=seed),
        default_noise=0.4,
    ),
    "Moons": DatasetSpec(
        name="Moons",
        maker=lambda n, noise, seed: make_moons(n_samples=n, noise=noise, random_state=seed),
        default_noise=0.25,
    ),
    "Circles": DatasetSpec(
        name="Circles",
        maker=lambda n, noise, seed: make_circles(n_samples=n, noise=noise, factor=0.5, random_state=seed),
        default_noise=0.2,
    ),
    "XOR": DatasetSpec(
        name="XOR",
        maker=lambda n, noise, seed: make_xor(n, noise, seed),
        default_noise=0.25,
    ),
}

# ------------------------------------
# Real datasets (2D views)
# ------------------------------------
def load_iris_2d():
    iris = load_iris()
    X = iris.data[:, :2]  # first two features for visualization
    y = iris.target
    return X, y

def load_breast_cancer_2d():
    bc = load_breast_cancer()
    X = bc.data[:, :2]
    y = bc.target  # binary
    return X, y

def load_wine_2d():
    wine = load_wine()
    # pick two informative features: alcohol, color_intensity (idx 0, 9)
    X = wine.data[:, [0, 9]]
    y = wine.target  # multiclass (3)
    return X, y

def load_penguins_2d():
    df = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "flipper_length_mm", "species"])
    X = df[["bill_length_mm", "flipper_length_mm"]].values
    y = df["species"].astype("category").cat.codes.values  # multiclass 3
    return X, y

def load_titanic_2d():
    df = sns.load_dataset("titanic").dropna(subset=["age", "fare", "sex", "class", "survived"])
    # Use two numeric features for 2D viz; keep some categorical via encoding
    feats = pd.concat([
        df[["age", "fare"]].reset_index(drop=True),
        pd.get_dummies(df[["sex", "class"]], drop_first=True).reset_index(drop=True)
    ], axis=1)
    # take first two principal-ish columns for 2D; or simply age/fare
    X = df[["age", "fare"]].values
    y = df["survived"].values.astype(int)  # binary
    return X, y

DATASETS["Iris"] = DatasetSpec("Iris", maker=lambda n, noise, seed: load_iris_2d(), supports_noise=False, is_multiclass=True, note="3 classes")
DATASETS["Breast Cancer"] = DatasetSpec("Breast Cancer", maker=lambda n, noise, seed: load_breast_cancer_2d(), supports_noise=False, is_multiclass=False, note="binary")
DATASETS["Wine"] = DatasetSpec("Wine", maker=lambda n, noise, seed: load_wine_2d(), supports_noise=False, is_multiclass=True, note="3 classes")
DATASETS["Penguins"] = DatasetSpec("Penguins", maker=lambda n, noise, seed: load_penguins_2d(), supports_noise=False, is_multiclass=True, note="3 species")
DATASETS["Titanic"] = DatasetSpec("Titanic", maker=lambda n, noise, seed: load_titanic_2d(), supports_noise=False, is_multiclass=False, note="binary")

# ------------------------------------
# Sidebar controls
# ------------------------------------
with st.sidebar:
    st.title("Decision Boundary Lab")
    st.caption("Use the controls below. Hover labels for help.")

    # CSV upload option (dynamically adds to dataset list)
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    uploaded_name = None
    uploaded_Xy = None
    if uploaded is not None:
        try:
            udf = pd.read_csv(uploaded)
            st.write("Columns:", list(udf.columns))
            target_col = st.selectbox("Target column", udf.columns)
            feature_cols = st.multiselect(
                "Two feature columns (2D)",
                [c for c in udf.columns if c != target_col],
                default=[c for c in udf.columns if c != target_col][:2]
            )
            if len(feature_cols) == 2:
                uploaded_X = udf[feature_cols].values
                uploaded_y = udf[target_col].values
                uploaded_name = "Uploaded CSV"
                uploaded_Xy = (uploaded_X, uploaded_y)
        except Exception as e:
            st.warning(f"CSV parse error: {e}")

    ds_options = list(DATASETS.keys()) + ([uploaded_name] if uploaded_name else [])
    ds_idx_default = ds_options.index("Moons") if "Moons" in ds_options else 0
    ds_name = st.selectbox("Dataset", ds_options, index=ds_idx_default,
                           help="Pick a dataset. CSV upload appears as a temporary option when provided.")

    seed = st.number_input("Random seed", value=42, step=1, help="Reproducible samples and splits.")
    n_samples = st.slider("Samples (synthetic only)", min_value=200, max_value=3000, value=600, step=50,
                          help="Total samples. Ignored for real datasets and CSV.")
    noise_default = DATASETS.get(ds_name, DATASETS["Moons"]).default_noise if ds_name in DATASETS else 0.0
    noise = st.slider("Noise (synthetic only)", 0.0, 1.0, noise_default, 0.05,
                      help="Random jitter; increases overlap.")

    st.markdown("---")
    st.subheader("Class balance (synthetic only)")
    imbalance = st.slider("Class 1 proportion", 0.05, 0.95, 0.5, 0.05,
                          help="Downsample to this fraction for class=1.")

    st.markdown("---")
    st.subheader("Feature engineering")
    use_poly = st.checkbox("Polynomial features", value=False,
                           help="Add cross terms (x1^2, x2^2, x1·x2, ...)")
    poly_deg = st.slider("Degree", 2, 6, 3, help="Higher degree = more flexible boundary.") if use_poly else 2
    standardize = st.checkbox("Standardize features", value=True, help="Zero-mean / unit-variance scaling.")

    st.markdown("---")
    st.subheader("Model")
    model_name = st.selectbox("Estimator",
                              ["Logistic Regression", "SVM (RBF)", "KNN", "Random Forest"] + (["XGBoost"] if HAS_XGB else []))

    # Hyperparams per model
    if model_name == "Logistic Regression":
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01, help="Smaller C = stronger regularization.")
    elif model_name == "SVM (RBF)":
        C = st.slider("C (margin)", 0.01, 20.0, 2.0, 0.01, help="Larger C = tighter fit.")
        gamma = st.slider("gamma", 0.001, 2.0, 0.5, 0.001, help="Higher gamma = wigglier boundary.")
    elif model_name == "KNN":
        k = st.slider("k (neighbors)", 1, 50, 7, 1, help="More neighbors = smoother.")
    elif model_name == "Random Forest":
        n_estimators = st.slider("Trees", 10, 500, 200, 10, help="Number of trees.")
        max_depth = st.slider("Max depth", 1, 20, 6, 1, help="Tree depth.")
    elif model_name == "XGBoost":
        n_estimators = st.slider("Trees", 10, 800, 250, 10)
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        max_depth = st.slider("Max depth", 1, 12, 4, 1)

    st.markdown("---")
    st.subheader("Train/Test")
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, help="Fraction held-out for testing.")

# ------------------------------------
# Generate or load data
# ------------------------------------
if uploaded_name and ds_name == uploaded_name and uploaded_Xy is not None:
    X, y = uploaded_Xy
    ds_is_multiclass = len(np.unique(y)) > 2
else:
    X, y = DATASETS[ds_name].maker(n_samples, noise, seed) if ds_name in DATASETS else uploaded_Xy
    ds_is_multiclass = DATASETS[ds_name].is_multiclass if ds_name in DATASETS else (len(np.unique(y)) > 2)

# Synthetic imbalance control (binary only)
if (not ds_is_multiclass) and imbalance != 0.5 and ds_name in ["Blobs","Moons","Circles","XOR"]:
    cls0_idx = np.where(y == 0)[0]
    cls1_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    n1 = int(round(imbalance * len(y)))
    n0 = len(y) - n1
    sel0 = rng.choice(cls0_idx, size=min(n0, len(cls0_idx)), replace=False)
    sel1 = rng.choice(cls1_idx, size=min(n1, len(cls1_idx)), replace=False)
    idx = np.concatenate([sel0, sel1])
    rng.shuffle(idx)
    X = X[idx]; y = y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# ------------------------------------
# Build model pipeline
# ------------------------------------
steps = []
if use_poly:
    steps.append(("poly", PolynomialFeatures(degree=poly_deg, include_bias=False)))
if standardize:
    steps.append(("scaler", StandardScaler()))

if model_name == "Logistic Regression":
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=500, multi_class="auto")
elif model_name == "SVM (RBF)":
    clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
elif model_name == "KNN":
    clf = KNeighborsClassifier(n_neighbors=k)
elif model_name == "Random Forest":
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
elif model_name == "XGBoost" and HAS_XGB:
    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")
else:
    clf = LogisticRegression(max_iter=500)

pipe = Pipeline(steps + [("clf", clf)])
pipe.fit(X_train, y_train)

# ------------------------------------
# Evaluate
# ------------------------------------
y_pred = pipe.predict(X_test)

metrics = {"Accuracy": accuracy_score(y_test, y_pred)}
if ds_is_multiclass:
    metrics["F1 (macro)"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
else:
    metrics.update({
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    })
    # try proba for ROC
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)

# ------------------------------------
# Decision surface grid
# ------------------------------------
PAD = 0.6
x_min, x_max = X[:,0].min()-PAD, X[:,0].max()+PAD
y_min, y_max = X[:,1].min()-PAD, X[:,1].max()+PAD
RES = 250
xx, yy = np.meshgrid(np.linspace(x_min, x_max, RES), np.linspace(y_min, y_max, RES))
X_grid = np.c_[xx.ravel(), yy.ravel()]

if not ds_is_multiclass:
    # binary: probability surface
    try:
        zz = pipe.predict_proba(X_grid)[:, 1]
    except Exception:
        if hasattr(pipe, "decision_function"):
            scores = pipe.decision_function(X_grid)
            zz = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            zz = pipe.predict(X_grid).astype(float)
    zz = zz.reshape(xx.shape)
else:
    # multiclass: color by predicted class index normalized
    zz_labels = pipe.predict(X_grid).astype(float)
    zz = (zz_labels - zz_labels.min()) / (zz_labels.max() - zz_labels.min() + 1e-9)
    zz = zz.reshape(xx.shape)

# ------------------------------------
# Plotly figure
# ------------------------------------
fig = go.Figure()
fig.add_trace(go.Contour(
    x=np.linspace(x_min, x_max, RES),
    y=np.linspace(y_min, y_max, RES),
    z=zz,
    contours=dict(start=0.0, end=1.0, size=0.05, coloring="heatmap", showlines=False),
    opacity=0.8,
    showscale=False,
))

mask0 = (y_train == np.unique(y)[0])
fig.add_trace(go.Scatter(x=X_train[mask0,0], y=X_train[mask0,1], mode="markers", name="Class A",
                         marker=dict(size=6, symbol="circle")))
# plot others
for cls in np.unique(y_train)[1:]:
    mask = (y_train == cls)
    fig.add_trace(go.Scatter(x=X_train[mask,0], y=X_train[mask,1], mode="markers", name=f"Train {cls}",
                             marker=dict(size=6, symbol="x")))

# test set outline
fig.add_trace(go.Scatter(x=X_test[:,0], y=X_test[:,1], mode="markers", name="Test (outline)",
                         marker=dict(size=9, line=dict(width=2), symbol="circle-open")))

fig.update_layout(
    title=f"{ds_name} — {model_name}",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2",
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=640,
)

# ------------------------------------
# Tabs
# ------------------------------------
tab1, tab2, tab3 = st.tabs(["Decision Surface", "ROC Curve", "Compare Models"])

with tab1:
    st.markdown("**What you’re seeing:** Colored background = model score (binary) or predicted class (multiclass). The ~0.5 band is the decision boundary in binary case.")
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Metrics (test)")
        mt = pd.DataFrame({k:[v] for k,v in metrics.items()}).T
        mt.columns = ["Value"]
        st.dataframe(mt.style.format({"Value": "{:.3f}"}), use_container_width=True)

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

with tab2:
    if ds_is_multiclass:
        st.info("ROC is defined for binary tasks. Select a binary dataset (e.g., Breast Cancer, Titanic, Blobs) to view ROC.")
    else:
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
            fig_roc.update_layout(title="Receiver Operating Characteristic",
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  height=500, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("ROC curve unavailable for this model.")

with tab3:
    st.markdown("**Compare Models:** Trains several estimators with the same feature-engineering settings and split.")
    def make_pipe(estimator):
        steps_cmp = []
        if use_poly:
            steps_cmp.append(("poly", PolynomialFeatures(degree=poly_deg, include_bias=False)))
        if standardize:
            steps_cmp.append(("scaler", StandardScaler()))
        steps_cmp.append(("clf", estimator))
        return Pipeline(steps_cmp)

    comps = [
        ("LogReg", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", multi_class="auto")),
        ("SVM-RBF", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True)),
        ("KNN-7", KNeighborsClassifier(n_neighbors=7)),
        ("RF-200", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ]
    if HAS_XGB:
        comps.append(("XGB", XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4,
                                           subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")))

    rows = []
    for name, est in comps:
        p = make_pipe(est)
        p.fit(X_train, y_train)
        yhat = p.predict(X_test)
        row = {"Model": name, "Accuracy": accuracy_score(y_test, yhat)}
        if ds_is_multiclass:
            row["F1 (macro)"] = f1_score(y_test, yhat, average="macro", zero_division=0)
        else:
            try:
                ypr = p.predict_proba(X_test)[:,1]
            except Exception:
                if hasattr(p, "decision_function"):
                    s = p.decision_function(X_test)
                    ypr = (s - s.min())/(s.max()-s.min()+1e-9)
                else:
                    ypr = yhat
            row.update({
                "Precision": precision_score(y_test, yhat, zero_division=0),
                "Recall": recall_score(y_test, yhat, zero_division=0),
                "F1": f1_score(y_test, yhat, zero_division=0),
            })
            try:
                row["ROC AUC"] = roc_auc_score(y_test, ypr)
            except Exception:
                pass
        rows.append(row)

    cmp_df = pd.DataFrame(rows).set_index("Model")
    sort_cols = [c for c in ["ROC AUC","F1 (macro)","F1","Accuracy"] if c in cmp_df.columns]
    if sort_cols:
        cmp_df = cmp_df.sort_values(by=sort_cols[0], ascending=False)
    st.dataframe(cmp_df.style.format("{:.3f}"), use_container_width=True)
    top_metric = sort_cols[0] if sort_cols else "Accuracy"
    st.bar_chart(cmp_df[top_metric])

# ------------------------------------
# Download current dataset (for reproducibility)
# ------------------------------------
@st.cache_data
def to_csv_bytes(X, y):
    df = pd.DataFrame({"x1": X[:,0], "x2": X[:,1], "y": y})
    return df.to_csv(index=False).encode()

csv_bytes = to_csv_bytes(X, y)
st.download_button("Download dataset (CSV)", data=csv_bytes, file_name=f"{ds_name.lower().replace(' ','_')}_data.csv", mime="text/csv")

# Footer + hero banner for README/social (for local dev preview)
st.caption("Decision Boundary Playground · Streamlit · © You · Build: v2")
