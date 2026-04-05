import os
import fastf1

# Create cache folder if not exists
os.makedirs("cache", exist_ok=True)

# Enable caching
fastf1.Cache.enable_cache("cache")
import warnings
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

CACHE_DIR = Path("fastf1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SESSIONS_TO_LOAD = [
    (2026, "Australian Grand Prix", "R"),
    (2026, "Chinese Grand Prix", "R"),
    (2026, "Japanese Grand Prix", "R"),
]

UPCOMING_RACES = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexican Grand Prix",
    "Brazilian Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
]


def timedelta_to_seconds(td):
    if pd.isna(td):
        return np.nan
    return td.total_seconds()


@st.cache_data(show_spinner="Loading F1 session data...")
def load_single_session(year, gp_name, session_type):
    try:
        session = fastf1.get_session(year, gp_name, session_type)
        session.load()
        return session
    except Exception as e:
        st.warning(f"Could not load {year} {gp_name}: {e}")
        return None


def extract_telemetry(lap):
    try:
        tel = lap.get_telemetry()
        if tel is None or tel.empty:
            return pd.Series({
                "AvgSpeed": np.nan, "MaxSpeed": np.nan,
                "MeanThrottle": np.nan, "MeanBrake": np.nan, "MeanRPM": np.nan
            })
        result = {
            "AvgSpeed": tel["Speed"].mean() if "Speed" in tel.columns else np.nan,
            "MaxSpeed": tel["Speed"].max() if "Speed" in tel.columns else np.nan,
            "MeanThrottle": tel["Throttle"].mean() if "Throttle" in tel.columns else np.nan,
            "MeanBrake": tel["Brake"].astype(float).mean() if "Brake" in tel.columns else np.nan,
            "MeanRPM": tel["RPM"].mean() if "RPM" in tel.columns else np.nan,
        }
        return pd.Series(result)
    except Exception:
        return pd.Series({
            "AvgSpeed": np.nan, "MaxSpeed": np.nan,
            "MeanThrottle": np.nan, "MeanBrake": np.nan, "MeanRPM": np.nan
        })


@st.cache_data(show_spinner="Extracting telemetry features (this takes a while)...")
def load_data():
    all_laps = []

    progress_bar = st.progress(0, text="Loading sessions...")
    total = len(SESSIONS_TO_LOAD)

    for idx, (year, gp_name, stype) in enumerate(SESSIONS_TO_LOAD):
        progress_bar.progress((idx) / total, text=f"Loading {gp_name} {year}...")
        session = load_single_session(year, gp_name, stype)
        if session is None:
            continue

        laps = session.laps.copy()
        laps = laps[laps["LapTime"].notna()].copy()
        laps = laps[laps["Sector1Time"].notna()].copy()
        laps = laps[laps["Sector2Time"].notna()].copy()
        laps = laps[laps["Sector3Time"].notna()].copy()
        laps = laps[laps["Compound"].notna()].copy()
        laps = laps[~laps["PitInTime"].notna()].copy()
        laps = laps[~laps["PitOutTime"].notna()].copy()

        if laps.empty:
            continue

        laps["LapTimeSec"] = laps["LapTime"].apply(timedelta_to_seconds)
        laps["Sector1Sec"] = laps["Sector1Time"].apply(timedelta_to_seconds)
        laps["Sector2Sec"] = laps["Sector2Time"].apply(timedelta_to_seconds)
        laps["Sector3Sec"] = laps["Sector3Time"].apply(timedelta_to_seconds)
        laps["Track"] = gp_name

        telemetry_data = []
        num_laps = len(laps)
        tel_bar = st.progress(0, text=f"Extracting telemetry for {gp_name} — 0/{num_laps} laps")
        for lap_idx, (_, lap_row) in enumerate(laps.iterrows()):
            try:
                tel_features = extract_telemetry(lap_row)
                telemetry_data.append(tel_features)
            except Exception:
                telemetry_data.append(pd.Series({
                    "AvgSpeed": np.nan, "MaxSpeed": np.nan,
                    "MeanThrottle": np.nan, "MeanBrake": np.nan, "MeanRPM": np.nan
                }))
            tel_bar.progress(
                (lap_idx + 1) / num_laps,
                text=f"Extracting telemetry for {gp_name} — {lap_idx + 1}/{num_laps} laps"
            )
        tel_bar.empty()

        tel_df = pd.DataFrame(telemetry_data, index=laps.index)
        laps = pd.concat([laps, tel_df], axis=1)

        cols_keep = [
            "LapTimeSec", "Sector1Sec", "Sector2Sec", "Sector3Sec",
            "Compound", "Track",
            "AvgSpeed", "MaxSpeed", "MeanThrottle", "MeanBrake", "MeanRPM"
        ]
        laps = laps[[c for c in cols_keep if c in laps.columns]]
        all_laps.append(laps)

    progress_bar.progress(1.0, text="All sessions loaded!")

    if not all_laps:
        st.error("No lap data could be loaded.")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)
    return df


def preprocess_data(df):
    df = df.dropna(subset=["LapTimeSec", "Sector1Sec", "Sector2Sec", "Sector3Sec",
                           "AvgSpeed", "MaxSpeed", "MeanThrottle", "MeanBrake"]).copy()

    if "MeanRPM" in df.columns:
        df["MeanRPM"] = df["MeanRPM"].fillna(df["MeanRPM"].median())

    for col in ["LapTimeSec", "Sector1Sec", "Sector2Sec", "Sector3Sec"]:
        q1 = df[col].quantile(0.02)
        q3 = df[col].quantile(0.98)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    compound_encoder = LabelEncoder()
    df["CompoundEncoded"] = compound_encoder.fit_transform(df["Compound"])

    track_encoder = LabelEncoder()
    df["TrackEncoded"] = track_encoder.fit_transform(df["Track"])

    numerical_features = ["AvgSpeed", "MaxSpeed", "MeanThrottle", "MeanBrake", "MeanRPM"]
    numerical_features = [f for f in numerical_features if f in df.columns]

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, compound_encoder, track_encoder, scaler, numerical_features


def get_feature_columns(numerical_features):
    return numerical_features + ["CompoundEncoded", "TrackEncoded"]


@st.cache_resource(show_spinner="Training models...")
def train_models(_df, numerical_features):
    df = _df.copy()
    feature_cols = get_feature_columns(numerical_features)

    X = df[feature_cols].values
    y_lap = df["LapTimeSec"].values
    y_sectors = df[["Sector1Sec", "Sector2Sec", "Sector3Sec"]].values

    X_train, X_test, y_lap_train, y_lap_test, y_sec_train, y_sec_test = train_test_split(
        X, y_lap, y_sectors, test_size=0.2, random_state=42
    )

    lr_lap = LinearRegression()
    lr_lap.fit(X_train, y_lap_train)
    lr_lap_pred = lr_lap.predict(X_test)
    lr_lap_mae = mean_absolute_error(y_lap_test, lr_lap_pred)
    lr_lap_rmse = np.sqrt(mean_squared_error(y_lap_test, lr_lap_pred))

    rf_lap = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_lap.fit(X_train, y_lap_train)
    rf_lap_pred = rf_lap.predict(X_test)
    rf_lap_mae = mean_absolute_error(y_lap_test, rf_lap_pred)
    rf_lap_rmse = np.sqrt(mean_squared_error(y_lap_test, rf_lap_pred))

    lr_sectors = MultiOutputRegressor(LinearRegression())
    lr_sectors.fit(X_train, y_sec_train)
    lr_sec_pred = lr_sectors.predict(X_test)
    lr_sec_mae = mean_absolute_error(y_sec_test, lr_sec_pred)
    lr_sec_rmse = np.sqrt(mean_squared_error(y_sec_test, lr_sec_pred))

    rf_sectors = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    )
    rf_sectors.fit(X_train, y_sec_train)
    rf_sec_pred = rf_sectors.predict(X_test)
    rf_sec_mae = mean_absolute_error(y_sec_test, rf_sec_pred)
    rf_sec_rmse = np.sqrt(mean_squared_error(y_sec_test, rf_sec_pred))

    metrics = {
        "LR Lap MAE": lr_lap_mae, "LR Lap RMSE": lr_lap_rmse,
        "RF Lap MAE": rf_lap_mae, "RF Lap RMSE": rf_lap_rmse,
        "LR Sector MAE": lr_sec_mae, "LR Sector RMSE": lr_sec_rmse,
        "RF Sector MAE": rf_sec_mae, "RF Sector RMSE": rf_sec_rmse,
    }

    test_results = {
        "y_lap_test": y_lap_test, "lr_lap_pred": lr_lap_pred, "rf_lap_pred": rf_lap_pred,
        "y_sec_test": y_sec_test, "lr_sec_pred": lr_sec_pred, "rf_sec_pred": rf_sec_pred,
    }

    models = {
        "lr_lap": lr_lap, "rf_lap": rf_lap,
        "lr_sectors": lr_sectors, "rf_sectors": rf_sectors,
    }

    return models, metrics, test_results


def predict(models, scaler, compound_encoder, track_encoder, numerical_features,
            compound, track, avg_speed, max_speed, throttle, brake, rpm):
    compound_enc = compound_encoder.transform([compound])[0]

    known_tracks = list(track_encoder.classes_)
    if track in known_tracks:
        track_enc = track_encoder.transform([track])[0]
        is_future = False
    else:
        track_enc = np.median(track_encoder.transform(known_tracks))
        is_future = True

    raw_features = []
    for feat_name in numerical_features:
        if feat_name == "AvgSpeed":
            raw_features.append(avg_speed)
        elif feat_name == "MaxSpeed":
            raw_features.append(max_speed)
        elif feat_name == "MeanThrottle":
            raw_features.append(throttle)
        elif feat_name == "MeanBrake":
            raw_features.append(brake)
        elif feat_name == "MeanRPM":
            raw_features.append(rpm)

    scaled = scaler.transform([raw_features])[0]

    feature_vec = list(scaled) + [compound_enc, track_enc]
    X_input = np.array([feature_vec])

    rf_lap = models["rf_lap"]
    rf_sectors = models["rf_sectors"]

    lap_pred = rf_lap.predict(X_input)[0]
    sector_pred = rf_sectors.predict(X_input)[0]

    return lap_pred, sector_pred, is_future


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


def plot_sector_distributions(df_raw):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    labels = ["Sector 1", "Sector 2", "Sector 3"]
    cols = ["Sector1Sec", "Sector2Sec", "Sector3Sec"]

    for i, (col, label, color) in enumerate(zip(cols, labels, colors)):
        axes[i].hist(df_raw[col], bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        axes[i].set_title(label, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Time (s)", fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Sector Time Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_raw):
    num_cols = ["AvgSpeed", "MaxSpeed", "MeanThrottle", "MeanBrake", "MeanRPM",
                "LapTimeSec", "Sector1Sec", "Sector2Sec", "Sector3Sec"]
    num_cols = [c for c in num_cols if c in df_raw.columns]
    corr = df_raw[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Telemetry Feature Correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_speed_vs_laptime(df_raw):
    fig, ax = plt.subplots(figsize=(10, 6))
    compounds = df_raw["Compound"].unique()
    palette = {"SOFT": "#e74c3c", "MEDIUM": "#f1c40f", "HARD": "#ecf0f1",
               "INTERMEDIATE": "#2ecc71", "WET": "#3498db"}

    for compound in sorted(compounds):
        subset = df_raw[df_raw["Compound"] == compound]
        color = palette.get(compound, "#95a5a6")
        ax.scatter(subset["AvgSpeed"], subset["LapTimeSec"],
                   alpha=0.5, s=20, label=compound, color=color, edgecolors="gray", linewidth=0.3)

    ax.set_xlabel("Average Speed (km/h)", fontsize=11)
    ax.set_ylabel("Lap Time (s)", fontsize=11)
    ax.set_title("Average Speed vs Lap Time by Compound", fontsize=14, fontweight="bold")
    ax.legend(title="Compound", title_fontsize=10, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(test_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    y_lap = test_results["y_lap_test"]
    rf_lap = test_results["rf_lap_pred"]

    axes[0].scatter(y_lap, rf_lap, alpha=0.4, s=15, color="#3498db", edgecolors="white", linewidth=0.3)
    lims = [min(y_lap.min(), rf_lap.min()) - 2, max(y_lap.max(), rf_lap.max()) + 2]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Perfect prediction")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Lap Time (s)", fontsize=11)
    axes[0].set_ylabel("Predicted Lap Time (s)", fontsize=11)
    axes[0].set_title("Lap Time: Actual vs Predicted (RF)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    y_sec = test_results["y_sec_test"]
    rf_sec = test_results["rf_sec_pred"]
    sector_labels = ["S1", "S2", "S3"]
    colors_s = ["#e74c3c", "#2ecc71", "#3498db"]

    for i in range(3):
        axes[1].scatter(y_sec[:, i], rf_sec[:, i], alpha=0.35, s=12,
                        color=colors_s[i], label=sector_labels[i], edgecolors="white", linewidth=0.2)

    all_vals = np.concatenate([y_sec.flatten(), rf_sec.flatten()])
    lims_s = [all_vals.min() - 1, all_vals.max() + 1]
    axes[1].plot(lims_s, lims_s, "r--", linewidth=1.5, alpha=0.7, label="Perfect prediction")
    axes[1].set_xlim(lims_s)
    axes[1].set_ylim(lims_s)
    axes[1].set_xlabel("Actual Sector Time (s)", fontsize=11)
    axes[1].set_ylabel("Predicted Sector Time (s)", fontsize=11)
    axes[1].set_title("Sector Times: Actual vs Predicted (RF)", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def build_streamlit_ui():
    st.set_page_config(page_title="F1 Lap & Sector Predictor", page_icon="🏎️", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Titillium Web', sans-serif; }
    .main { background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 50%, #f0f2f8 100%); }
    .stApp { background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 50%, #f0f2f8 100%); }
    h1, h2, h3 { color: #1a1a2e !important; }
    p, span, label, .stMarkdown { color: #2d2d44; }
    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(225, 6, 0, 0.12);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(225, 6, 0, 0.06), 0 1px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(225, 6, 0, 0.12), 0 2px 8px rgba(0,0,0,0.06);
        border-color: rgba(225, 6, 0, 0.35);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e10600, #ff2d2d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b6b80;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .sector-s1 .metric-value { background: linear-gradient(135deg, #e10600, #ff4040); -webkit-background-clip: text; }
    .sector-s2 .metric-value { background: linear-gradient(135deg, #1a1a2e, #3a3a5c); -webkit-background-clip: text; }
    .sector-s3 .metric-value { background: linear-gradient(135deg, #e10600, #ff6666); -webkit-background-clip: text; }
    .header-container {
        text-align: center;
        padding: 40px 0 20px 0;
        border-bottom: 2px solid rgba(225, 6, 0, 0.12);
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a1a2e, #2d2d44);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #6b6b80;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(225, 6, 0, 0.1);
    }
    .metrics-table th {
        background: #fff5f5;
        color: #6b6b80;
        padding: 12px 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
        text-align: left;
    }
    .metrics-table td {
        padding: 10px 16px;
        color: #2d2d44;
        border-top: 1px solid #f0f0f5;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 8px;
        color: #6b6b80;
        border: 1px solid rgba(225, 6, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e10600, #c10500) !important;
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #faf5f5);
        border-right: 1px solid rgba(225, 6, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg"
             alt="Formula 1 Logo"
             style="height: 60px; margin-bottom: 16px;">
        <div class="header-title">Lap & Sector Predictor</div>
        <div class="header-subtitle">Machine Learning · Telemetry Analytics · 2026 Season</div>
    </div>
    """, unsafe_allow_html=True)

    df_raw = load_data()

    if df_raw.empty:
        st.error("Failed to load any F1 data. Please check your internet connection and try again.")
        return

    df_processed, compound_enc, track_enc, scaler, num_feats = preprocess_data(df_raw.copy())

    if len(df_processed) < 20:
        st.error("Not enough data after preprocessing. Try loading more sessions.")
        return

    models, metrics, test_results = train_models(df_processed, num_feats)

    with st.sidebar:
        st.markdown("## ⚙️ Prediction Inputs")
        st.markdown("---")

        compounds = sorted(df_raw["Compound"].dropna().unique().tolist())
        completed_tracks = sorted(df_raw["Track"].dropna().unique().tolist())
        upcoming_tracks = [r for r in UPCOMING_RACES if r not in completed_tracks]
        all_tracks = completed_tracks + upcoming_tracks
        track_labels = [f"✅ {t}" for t in completed_tracks] + [f"🔮 {t}" for t in upcoming_tracks]

        compound = st.selectbox("🔵 Tyre Compound", compounds, index=0)
        selected_label = st.selectbox("🏁 Track", track_labels, index=0,
                                       help="✅ = completed race (trained on) · 🔮 = upcoming race (estimate)")
        track = all_tracks[track_labels.index(selected_label)]

        track_laps = df_raw[df_raw["Track"] == track]

        default_avg = float(track_laps["AvgSpeed"].median()) if not track_laps.empty else 200.0
        default_max = float(track_laps["MaxSpeed"].median()) if not track_laps.empty else 320.0
        default_throttle = float(track_laps["MeanThrottle"].median()) if not track_laps.empty else 60.0
        default_brake = float(track_laps["MeanBrake"].median()) if not track_laps.empty else 0.15
        default_rpm = float(track_laps["MeanRPM"].median()) if (not track_laps.empty and "MeanRPM" in track_laps.columns) else 10000.0

        st.markdown("### 📊 Telemetry")
        avg_speed = st.slider("Average Speed (km/h)", 100.0, 300.0, min(max(default_avg, 100.0), 300.0), 0.5)
        max_speed = st.slider("Max Speed (km/h)", 150.0, 370.0, min(max(default_max, 150.0), 370.0), 0.5)
        throttle = st.slider("Throttle %", 0.0, 100.0, min(max(default_throttle, 0.0), 100.0), 0.5)
        brake = st.slider("Brake %", 0.0, 1.0, min(max(default_brake, 0.0), 1.0), 0.01)
        rpm = st.slider("RPM", 5000.0, 15000.0, min(max(default_rpm, 5000.0), 15000.0), 100.0)

        predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")

    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📈 Analytics", "🔬 Model Performance"])

    with tab1:
        if predict_btn:
            try:
                lap_pred, sector_pred, is_future = predict(
                    models, scaler, compound_enc, track_enc, num_feats,
                    compound, track, avg_speed, max_speed, throttle, brake, rpm
                )

                if is_future:
                    st.warning(f"🔮 **{track}** is an upcoming race — prediction is an estimate based on telemetry inputs and data from completed 2026 races.")

                st.markdown("### Predicted Lap Time")
                col_main = st.columns([1, 2, 1])
                with col_main[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Lap Time</div>
                        <div class="metric-value">{format_time(lap_pred)}</div>
                        <div class="metric-label">{lap_pred:.3f} seconds</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### Predicted Sector Times")
                cols = st.columns(3)
                sector_names = ["Sector 1", "Sector 2", "Sector 3"]
                sector_classes = ["sector-s1", "sector-s2", "sector-s3"]

                for i, (col, name, cls) in enumerate(zip(cols, sector_names, sector_classes)):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card {cls}">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value">{sector_pred[i]:.3f}s</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### Input Summary")
                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.markdown(f"**Track:** {track}")
                    st.markdown(f"**Compound:** {compound}")
                    if is_future:
                        st.markdown("**Status:** 🔮 Upcoming")
                with summary_cols[1]:
                    st.markdown(f"**Avg Speed:** {avg_speed} km/h")
                    st.markdown(f"**Max Speed:** {max_speed} km/h")
                with summary_cols[2]:
                    st.markdown(f"**Throttle:** {throttle}%")
                    st.markdown(f"**Brake:** {brake*100:.1f}%")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.info("👈 Configure inputs in the sidebar and click **Predict** to see results.")

    with tab2:
        st.markdown("### 📊 Data Analytics")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### Sector Time Distributions")
            fig_sectors = plot_sector_distributions(df_raw)
            st.pyplot(fig_sectors)
            plt.close(fig_sectors)

        with viz_col2:
            st.markdown("#### Speed vs Lap Time")
            fig_speed = plot_speed_vs_laptime(df_raw)
            st.pyplot(fig_speed)
            plt.close(fig_speed)

        st.markdown("#### Telemetry Correlation Heatmap")
        fig_corr = plot_correlation_heatmap(df_raw)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

        st.markdown("#### Dataset Overview")
        col_stats = st.columns(4)
        with col_stats[0]:
            st.metric("Total Laps", f"{len(df_raw):,}")
        with col_stats[1]:
            st.metric("Tracks", f"{df_raw['Track'].nunique()}")
        with col_stats[2]:
            st.metric("Compounds", f"{df_raw['Compound'].nunique()}")
        with col_stats[3]:
            st.metric("Processed Laps", f"{len(df_processed):,}")

    with tab3:
        st.markdown("### 🔬 Model Evaluation")

        st.markdown("#### Prediction Metrics")

        metric_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "Linear Regression", "Random Forest"],
            "Target": ["Lap Time", "Lap Time", "Sector Times", "Sector Times"],
            "MAE (s)": [
                f"{metrics['LR Lap MAE']:.4f}",
                f"{metrics['RF Lap MAE']:.4f}",
                f"{metrics['LR Sector MAE']:.4f}",
                f"{metrics['RF Sector MAE']:.4f}",
            ],
            "RMSE (s)": [
                f"{metrics['LR Lap RMSE']:.4f}",
                f"{metrics['RF Lap RMSE']:.4f}",
                f"{metrics['LR Sector RMSE']:.4f}",
                f"{metrics['RF Sector RMSE']:.4f}",
            ],
        })
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.markdown("#### Actual vs Predicted")
        fig_avp = plot_actual_vs_predicted(test_results)
        st.pyplot(fig_avp)
        plt.close(fig_avp)

        st.markdown("#### Model Comparison")
        comp_cols = st.columns(2)
        with comp_cols[0]:
            st.markdown("**Lap Time MAE**")
            lr_val = metrics["LR Lap MAE"]
            rf_val = metrics["RF Lap MAE"]
            better = "Random Forest" if rf_val < lr_val else "Linear Regression"
            improvement = abs(lr_val - rf_val)
            st.markdown(f"- Linear Regression: **{lr_val:.4f}s**")
            st.markdown(f"- Random Forest: **{rf_val:.4f}s**")
            st.success(f"✅ {better} wins by {improvement:.4f}s")

        with comp_cols[1]:
            st.markdown("**Sector Time MAE**")
            lr_val_s = metrics["LR Sector MAE"]
            rf_val_s = metrics["RF Sector MAE"]
            better_s = "Random Forest" if rf_val_s < lr_val_s else "Linear Regression"
            improvement_s = abs(lr_val_s - rf_val_s)
            st.markdown(f"- Linear Regression: **{lr_val_s:.4f}s**")
            st.markdown(f"- Random Forest: **{rf_val_s:.4f}s**")
            st.success(f"✅ {better_s} wins by {improvement_s:.4f}s")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8b949e; font-size:0.85rem;'>"
        "Built with FastF1 · Scikit-Learn · Streamlit | 2026 F1 Season Data"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    build_streamlit_ui()
