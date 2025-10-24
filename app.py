
"""
Decision Boundary Playground — Streamlit (Python) MVP

Run locally:
  python -m venv .venv && source .venv/bin/activate
  python -m pip install -r requirements.txt
  streamlit run app.py
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataclasses import dataclass
from sklearn.datasets import make_blobs, make_moons, make_circles
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

st.set_page_config(page_title="Decision Boundary Playground", layout="wide")

# ------------------------------------
# Header & quick intro
# ------------------------------------
st.title("🧠 Decision Boundary Playground")
st.markdown(
    """
    Explore how different **classification models** cut the 2D feature space into decision regions.
    - Pick a **dataset** (blobs, moons, circles, XOR), adjust **noise** and **class balance**.
    - Toggle **feature engineering** (polynomial features, standardization).
    - Choose a **model** and tweak its hyperparameters.
    - See the **decision surface**, test **metrics**, and compare models.  
    Tip: Try `Moons` + `SVM (RBF)` or `XOR` + `Logistic Regression` **with** polynomial features.
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
    maker: callable
    supports_noise: bool = True
    default_noise: float = 0.25

DATASETS = {
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
# Sidebar controls
# ------------------------------------
with st.sidebar:
    st.title("Decision Boundary Lab")
    st.caption("Use the controls below. Hover labels for help.")
    ds_name = st.selectbox("Dataset", list(DATASETS.keys()), index=1,
                           help="Synthetic 2D datasets to illustrate different separability patterns.")
    seed = st.number_input("Random seed", value=42, step=1,
                           help="Fix randomness for reproducible samples and splits.")
    n_samples = st.slider("Samples", min_value=200, max_value=3000, value=600, step=50,
                          help="Number of total samples (train + test). Higher = denser cloud but slower surface compute.")
    noise = st.slider("Noise", min_value=0.0, max_value=1.0, value=DATASETS[ds_name].default_noise, step=0.05,
                      help="Random jitter added to points. More noise makes classes overlap and the task harder.")

    st.markdown("---")
    st.subheader("Class balance")
    imbalance = st.slider("Class 1 proportion", 0.05, 0.95, 0.5, 0.05,
                          help="Target fraction of class 1 after downsampling. Try skewed classes to see precision/recall tradeoffs.")

    st.markdown("---")
    st.subheader("Feature engineering")
    use_poly = st.checkbox("Polynomial features", value=False,
                           help="Adds non-linear feature crosses (x₁², x₂², x₁·x₂, …). Helps linear models learn curved boundaries.")
    poly_deg = st.slider("Degree", 2, 6, 3,
                         help="Higher degree = more complex curves. Beware overfitting.") if use_poly else 2
    standardize = st.checkbox("Standardize features", value=True,
                              help="Zero-mean / unit-variance scaling. Important for distance-based models and SVM.")

    st.markdown("---")
    st.subheader("Model")
    model_name = st.selectbox("Estimator",
                              ["Logistic Regression", "SVM (RBF)", "KNN", "Random Forest"] + (["XGBoost"] if HAS_XGB else []))

    # Hyperparams per model
    if model_name == "Logistic Regression":
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01,
                      help="Smaller C = stronger regularization (simpler boundary).")
    elif model_name == "SVM (RBF)":
        C = st.slider("C (margin)", 0.01, 20.0, 2.0, 0.01,
                      help="Larger C = tighter fit to training points.")
        gamma = st.slider("gamma", 0.001, 2.0, 0.5, 0.001,
                          help="Radius of influence. Higher gamma = wigglier boundary.")
    elif model_name == "KNN":
        k = st.slider("k (neighbors)", 1, 50, 7, 1, help="How many neighbors vote. Higher k = smoother boundary.")
    elif model_name == "Random Forest":
        n_estimators = st.slider("Trees", 10, 500, 200, 10,
                                 help="Number of trees. More can improve stability up to a point.")
        max_depth = st.slider("Max depth", 1, 20, 6, 1,
                              help="Tree depth. Deeper = more complex boundaries.")
    elif model_name == "XGBoost":
        n_estimators = st.slider("Trees", 10, 800, 250, 10)
        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        max_depth = st.slider("Max depth", 1, 12, 4, 1)

    st.markdown("---")
    st.subheader("Train/Test")
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, help="Fraction of data held out for testing.")

# ------------------------------------
# Generate data
# ------------------------------------
X, y = DATASETS[ds_name].maker(n_samples, noise, seed)

# Apply imbalance by downsampling class 1 proportion
if imbalance != 0.5:
    cls0_idx = np.where(y == 0)[0]
    cls1_idx = np.where(y == 1)[0]
    rng = np.random.default_rng(seed)
    n1 = int(round(imbalance * len(y)))
    n0 = len(y) - n1
    sel0 = rng.choice(cls0_idx, size=min(n0, len(cls0_idx)), replace=False)
    sel1 = rng.choice(cls1_idx, size=min(n1, len(cls1_idx)), replace=False)
    idx = np.concatenate([sel0, sel1])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

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
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=300)
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
    clf = LogisticRegression(max_iter=300)

pipe = Pipeline(steps + [("clf", clf)])
pipe.fit(X_train, y_train)

# ------------------------------------
# Evaluate
# ------------------------------------
y_pred = pipe.predict(X_test)
try:
    y_prob = pipe.predict_proba(X_test)[:, 1]
except Exception:
    y_scores = pipe.decision_function(X_test) if hasattr(pipe, "decision_function") else y_pred
    y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
}
try:
    metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)
except Exception:
    pass

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]).astype(int)

# ------------------------------------
# Decision boundary grid
# ------------------------------------
PAD = 0.6
x_min, x_max = X[:,0].min()-PAD, X[:,0].max()+PAD
y_min, y_max = X[:,1].min()-PAD, X[:,1].max()+PAD

RES = 250
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, RES),
    np.linspace(y_min, y_max, RES),
)
X_grid = np.c_[xx.ravel(), yy.ravel()]

try:
    zz = pipe.predict_proba(X_grid)[:, 1]
except Exception:
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_grid)
        zz = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        zz = pipe.predict(X_grid)
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

mask0 = (y_train == 0)
mask1 = (y_train == 1)
fig.add_trace(go.Scatter(x=X_train[mask0,0], y=X_train[mask0,1], mode="markers", name="Train 0",
                         marker=dict(size=6, symbol="circle")))
fig.add_trace(go.Scatter(x=X_train[mask1,0], y=X_train[mask1,1], mode="markers", name="Train 1",
                         marker=dict(size=6, symbol="x")))

mask0t = (y_test == 0)
mask1t = (y_test == 1)
fig.add_trace(go.Scatter(x=X_test[mask0t,0], y=X_test[mask0t,1], mode="markers", name="Test 0",
                         marker=dict(size=9, line=dict(width=2), symbol="circle-open")))
fig.add_trace(go.Scatter(x=X_test[mask1t,0], y=X_test[mask1t,1], mode="markers", name="Test 1",
                         marker=dict(size=9, line=dict(width=2), symbol="x-open")))

fig.update_layout(
    title=f"{DATASETS[ds_name].name} — {model_name}",
    xaxis_title="Feature 1",
    yaxis_title="Feature 2",
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=640,
)

# ------------------------------------
# Tabs: Decision Surface | ROC Curve | Compare Models
# ------------------------------------
tab1, tab2, tab3 = st.tabs(["Decision Surface", "ROC Curve", "Compare Models"])

with tab1:
    st.markdown("**What you’re seeing:** The colored background shows the model’s predicted probability for class 1. "
                "The line around ~0.5 is the decision boundary. Markers are train/test points.")
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

        st.markdown("---")
        st.caption("Tip: enable Polynomial features for Logistic/SVM to capture non-linear patterns like XOR/Circles.")

with tab2:
    st.markdown("**ROC curve** plots True Positive Rate vs False Positive Rate across thresholds. "
                "Higher AUC means better ranking quality irrespective of a single threshold.")
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(title="Receiver Operating Characteristic",
                              xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                              height=500, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.info("ROC curve unavailable for this model.")

with tab3:
    st.markdown("**Compare Models:** Trains several estimators with the same feature-engineering settings and split, "
                "then ranks by ROC AUC (or F1 if AUC not available).")
    def make_pipe(estimator):
        steps_cmp = []
        if use_poly:
            steps_cmp.append(("poly", PolynomialFeatures(degree=poly_deg, include_bias=False)))
        if standardize:
            steps_cmp.append(("scaler", StandardScaler()))
        steps_cmp.append(("clf", estimator))
        return Pipeline(steps_cmp)

    comps = [
        ("LogReg", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")),
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
        try:
            ypr = p.predict_proba(X_test)[:,1]
        except Exception:
            if hasattr(p, "decision_function"):
                s = p.decision_function(X_test)
                ypr = (s - s.min())/(s.max()-s.min()+1e-9)
            else:
                ypr = yhat
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, yhat),
            "Precision": precision_score(y_test, yhat, zero_division=0),
            "Recall": recall_score(y_test, yhat, zero_division=0),
            "F1": f1_score(y_test, yhat, zero_division=0),
        }
        try:
            row["ROC AUC"] = roc_auc_score(y_test, ypr)
        except Exception:
            pass
        rows.append(row)

    cmp_df = pd.DataFrame(rows).set_index("Model")
    sort_cols = [c for c in ["ROC AUC","F1","Accuracy"] if c in cmp_df.columns]
    if sort_cols:
        cmp_df = cmp_df.sort_values(by=sort_cols[0], ascending=False)
    st.dataframe(cmp_df.style.format("{:.3f}"), use_container_width=True)
    top_metric = "ROC AUC" if "ROC AUC" in cmp_df.columns else "F1"
    st.bar_chart(cmp_df[top_metric])

# ------------------------------------
# Download current dataset (for reproducibility)
# ------------------------------------
@st.cache_data
def to_csv_bytes(X, y):
    df = pd.DataFrame({"x1": X[:,0], "x2": X[:,1], "y": y})
    return df.to_csv(index=False).encode()

csv_bytes = to_csv_bytes(X, y)
st.download_button("Download dataset (CSV)", data=csv_bytes,
                   file_name=f"{DATASETS[ds_name].name.lower()}_data.csv", mime="text/csv")

# ------------------------------------
# Footer
# ------------------------------------
st.caption("Decision Boundary Playground · Streamlit · © You · Build: MVP")
