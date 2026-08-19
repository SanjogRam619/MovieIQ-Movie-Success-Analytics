# ============================================================
# MOVIEIQ
# Movie Success Analysis & Prediction Platform
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.stats import ttest_ind, chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MOVIEIQ PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

html,
body {
    background-color: #080D1A !important;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(99, 102, 241, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 15% 90%,
            rgba(34, 211, 238, 0.05),
            transparent 30%
        ),
        #080D1A !important;

    color: #F8FAFC;
}


/* ==========================================================
   HEADER
   ========================================================== */

header[data-testid="stHeader"] {
    background-color: #080D1A !important;
    border-bottom: 1px solid #17233A !important;
}


/* ==========================================================
   MAIN CONTENT
   ========================================================== */

.main .block-container {
    max-width: 1400px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    width: 17rem !important;

    background:
        linear-gradient(
            180deg,
            #070C18 0%,
            #0B1020 100%
        ) !important;

    border-right: 1px solid #1E2A44 !important;
}

section[data-testid="stSidebar"] > div {
    width: 17rem !important;
    background: transparent !important;
}


/* ==========================================================
   SIDEBAR BRAND
   ========================================================== */

.brand {
    margin-top: 12px;
    margin-bottom: 5px;

    font-size: 29px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #22D3EE,
            #6366F1,
            #A855F7
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 10px;
    letter-spacing: 2.2px;
    color: #64748B;
    font-weight: 600;
    margin-bottom: 40px;
}

.nav-label {
    color: #64748B;
    font-size: 10px;
    letter-spacing: 1.8px;
    font-weight: 700;
    margin-bottom: 10px;
}


/* ==========================================================
   SIDEBAR CLOSED DROPDOWN
   ========================================================== */

section[data-testid="stSidebar"]
div[data-baseweb="select"] {
    background: #17233A !important;
    background-color: #17233A !important;

    border: 1px solid #33466A !important;
    border-radius: 10px !important;

    box-shadow: none !important;

    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] > div {
    background: #17233A !important;
    background-color: #17233A !important;

    border-radius: 10px !important;

    opacity: 1 !important;
}


/* ==========================================================
   SIDEBAR SELECTED TEXT — WHITE
   ========================================================== */

section[data-testid="stSidebar"]
div[data-baseweb="select"] div {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] span {
    color: #FFFFFF !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] p {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] input {
    color: #FFFFFF !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"] svg {
    fill: #CBD5E1 !important;
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="select"]:hover {
    background: #1D2D49 !important;
    background-color: #1D2D49 !important;

    border-color: #7C5CFC !important;
}


/* ==========================================================
   SIDEBAR OPEN DROPDOWN
   ========================================================== */

section[data-testid="stSidebar"]
div[data-baseweb="popover"] {
    background: #111A2D !important;
    background-color: #111A2D !important;

    border: 1px solid #33466A !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"]
div[data-baseweb="menu"] {
    background: #111A2D !important;
    background-color: #111A2D !important;
}

section[data-testid="stSidebar"]
ul[role="listbox"] {
    background: #111A2D !important;
    background-color: #111A2D !important;

    border: 1px solid #33466A !important;
    border-radius: 10px !important;

    padding: 6px !important;
}


/* ==========================================================
   SIDEBAR OPTIONS
   ========================================================== */

section[data-testid="stSidebar"]
li[role="option"] {
    background: #17233A !important;
    background-color: #17233A !important;

    color: #F8FAFC !important;

    border-radius: 7px !important;

    margin: 2px 0 !important;
    padding: 10px !important;
}

section[data-testid="stSidebar"]
li[role="option"] * {
    color: #F8FAFC !important;
}

section[data-testid="stSidebar"]
li[role="option"]:hover {
    background: #263B60 !important;
    background-color: #263B60 !important;
}

section[data-testid="stSidebar"]
li[role="option"]:hover * {
    color: #67E8F9 !important;
}

section[data-testid="stSidebar"]
li[role="option"][aria-selected="true"] {
    background: #4F46E5 !important;
    background-color: #4F46E5 !important;

    color: #FFFFFF !important;
}

section[data-testid="stSidebar"]
li[role="option"][aria-selected="true"] * {
    color: #FFFFFF !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero-section {
    position: relative;
    overflow: hidden;

    min-height: 310px;

    padding: 48px 52px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            #111C3A 0%,
            #1B2550 42%,
            #29235A 72%,
            #24173F 100%
        );

    border: 1px solid #3B4B86;

    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.35);

    margin-bottom: 38px;
}

.hero-glow {
    position: absolute;

    width: 420px;
    height: 420px;

    right: -140px;
    top: -190px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.38),
            rgba(124, 58, 237, 0.12),
            transparent 70%
        );

    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;

    max-width: 850px;
}

.hero-badge {
    display: inline-block;

    padding: 6px 12px;

    margin-bottom: 14px;

    border-radius: 20px;

    background: rgba(34, 211, 238, 0.08);

    border: 1px solid rgba(34, 211, 238, 0.25);

    color: #67E8F9;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 2px;
}

.hero-title {
    font-size: 52px;

    line-height: 1;

    font-weight: 850;

    margin-bottom: 12px;

    background:
        linear-gradient(
            90deg,
            #FFFFFF,
            #A5B4FC,
            #67E8F9
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 21px;

    font-weight: 600;

    color: #E2E8F0;

    margin-bottom: 14px;
}

.hero-description {
    max-width: 700px;

    font-size: 15px;

    line-height: 1.7;

    color: #9FB0CB;

    margin-bottom: 25px;
}


/* ==========================================================
   HERO STATS
   ========================================================== */

.hero-stats {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.hero-stat {
    display: flex;
    align-items: center;
    gap: 8px;

    padding: 9px 14px;

    border-radius: 9px;

    background: rgba(8, 13, 26, 0.35);

    border: 1px solid rgba(148, 163, 184, 0.15);

    color: #CBD5E1;

    font-size: 12px;

    font-weight: 500;
}

.stat-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #22D3EE;

    box-shadow:
        0 0 8px rgba(34, 211, 238, 0.7);
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    font-size: 25px;

    font-weight: 750;

    color: #F1F5F9;

    margin-top: 25px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #7F8DA6;

    font-size: 14px;

    margin-bottom: 20px;
}

h1,
h2,
h3,
h4 {
    color: #F8FAFC !important;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            #111A2D,
            #0D1526
        );

    border: 1px solid #263653;

    border-radius: 16px;

    padding: 20px;

    min-height: 125px;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.20);

    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: #6366F1;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;

    font-size: 29px;

    font-weight: 750;
}


/* ==========================================================
   INPUT LABELS
   ========================================================== */

label,
[data-testid="stWidgetLabel"] p {
    color: #A8B5C9 !important;
    font-weight: 500 !important;
}


/* ==========================================================
   INPUT BOXES
   ========================================================== */

div[data-baseweb="input"] {
    background: #17233A !important;
    background-color: #17233A !important;

    border: 1px solid #33466A !important;

    border-radius: 10px !important;
}

div[data-baseweb="input"] input {
    background: #17233A !important;
    background-color: #17233A !important;

    color: #F8FAFC !important;

    caret-color: #38BDF8 !important;
}


/* ==========================================================
   NUMBER INPUT
   ========================================================== */

div[data-testid="stNumberInput"] {
    background: transparent !important;
}

div[data-testid="stNumberInput"]
div[data-baseweb="input"] {
    background: #17233A !important;
    background-color: #17233A !important;
}

div[data-testid="stNumberInput"] input {
    background: #17233A !important;
    background-color: #17233A !important;

    color: #F8FAFC !important;
}

div[data-testid="stNumberInput"] button {
    background: #1D2D49 !important;
    background-color: #1D2D49 !important;

    color: #CBD5E1 !important;

    border-color: #33466A !important;
}

div[data-testid="stNumberInput"] button:hover {
    background: #263B60 !important;
    background-color: #263B60 !important;

    color: #38BDF8 !important;
}


/* ==========================================================
   GENRE DROPDOWN
   ========================================================== */

div[data-testid="stSelectbox"]
div[data-baseweb="select"] {
    background: #17233A !important;
    background-color: #17233A !important;

    border: 1px solid #33466A !important;

    border-radius: 10px !important;

    box-shadow: none !important;
}

div[data-testid="stSelectbox"]
div[data-baseweb="select"] > div {
    background: #17233A !important;
    background-color: #17233A !important;

    border-radius: 10px !important;
}

div[data-testid="stSelectbox"]
div[data-baseweb="select"] span {
    color: #FFFFFF !important;

    opacity: 1 !important;

    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-testid="stSelectbox"]
div[data-baseweb="select"] div {
    color: #FFFFFF !important;
}

div[data-testid="stSelectbox"]
div[data-baseweb="select"] svg {
    fill: #94A3B8 !important;
    color: #94A3B8 !important;
}

div[data-testid="stSelectbox"]
div[data-baseweb="select"]:hover {
    background: #263B60 !important;
    background-color: #263B60 !important;

    border-color: #6366F1 !important;
}


/* ==========================================================
   GENERAL SELECTBOX
   ========================================================== */

div[data-baseweb="select"] {
    background: #17233A !important;
    background-color: #17233A !important;

    border: 1px solid #33466A !important;

    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

div[data-baseweb="select"] svg {
    fill: #94A3B8 !important;
}


/* ==========================================================
   SELECTBOX POPUP
   ========================================================== */

div[data-baseweb="popover"] {
    background: #111A2D !important;
    background-color: #111A2D !important;

    border: 1px solid #33466A !important;

    border-radius: 10px !important;
}

div[data-baseweb="menu"] {
    background: #111A2D !important;
    background-color: #111A2D !important;
}

ul[role="listbox"] {
    background: #111A2D !important;
    background-color: #111A2D !important;

    padding: 6px !important;
}

li[role="option"] {
    background: #17233A !important;
    background-color: #17233A !important;

    color: #FFFFFF !important;

    border-radius: 7px !important;

    margin: 2px 0 !important;
}

li[role="option"] * {
    color: #FFFFFF !important;
}

li[role="option"]:hover {
    background: #263B60 !important;
    background-color: #263B60 !important;
}

li[role="option"]:hover * {
    color: #67E8F9 !important;
}

li[role="option"][aria-selected="true"] {
    background: #4F46E5 !important;
    background-color: #4F46E5 !important;

    color: #FFFFFF !important;
}


/* ==========================================================
   INFO CARDS
   ========================================================== */

.info-card {
    background:
        linear-gradient(
            145deg,
            #142039,
            #10192C
        );

    border: 1px solid #2F4265;

    border-radius: 15px;

    padding: 22px;

    margin-top: 12px;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.18);
}

.info-card-title {
    color: #67E8F9;

    font-size: 13px;

    letter-spacing: 1.5px;

    font-weight: 800;

    margin-bottom: 10px;
}

.info-card-text {
    color: #B6C2D9;

    font-size: 14px;

    line-height: 1.7;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {
    background:
        linear-gradient(
            90deg,
            #4F46E5,
            #7C3AED
        );

    color: #FFFFFF;

    border: none;

    border-radius: 11px;

    padding: 13px 20px;

    font-weight: 700;

    box-shadow:
        0 8px 25px
        rgba(99, 102, 241, 0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #6366F1,
            #8B5CF6
        );

    color: #FFFFFF;

    transform: translateY(-2px);
}


/* ==========================================================
   PREDICTION RESULT
   ========================================================== */

.prediction-box {
    background:
        linear-gradient(
            135deg,
            #11183B 0%,
            #18194A 50%,
            #21184A 100%
        );

    border: 1px solid #4F46E5;

    border-radius: 18px;

    padding: 30px;

    text-align: center;

    margin-top: 25px;

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.30);
}

.prediction-label {
    color: #94A3B8;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 2px;

    text-transform: uppercase;

    margin-bottom: 10px;
}

.prediction-result {
    color: #67E8F9;

    font-size: 30px;

    font-weight: 800;

    letter-spacing: 0.5px;
}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

div[data-testid="stProgressBar"] > div {
    background-color: #17233A !important;
}

div[data-testid="stProgressBar"] > div > div {
    background:
        linear-gradient(
            90deg,
            #4F46E5,
            #22D3EE
        ) !important;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {
    border: 1px solid #263653;

    border-radius: 12px;

    overflow: hidden;
}


/* ==========================================================
   ALERTS
   ========================================================== */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {
    border-color: #1E2A44;
}


/* ==========================================================
   SCROLLBAR
   ========================================================== */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #080D1A;
}

::-webkit-scrollbar-thumb {
    background: #263653;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3B4B86;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv("movies_cleaned.csv")


df = load_data()


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    features = [
        "budget",
        "popularity",
        "runtime",
        "vote_average",
        "genre"
    ]

    X = data[features]

    y = data["success"]


    numerical_features = [
        "budget",
        "popularity",
        "runtime",
        "vote_average"
    ]

    categorical_features = [
        "genre"
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                "passthrough",
                numerical_features
            ),

            (
                "genre",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                categorical_features
            )
        ]
    )


    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced"
    )


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    pipeline.fit(
        X_train,
        y_train
    )


    y_pred = pipeline.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )


    importances = (
        pipeline
        .named_steps["model"]
        .feature_importances_
    )


    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importances

    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    return (
        pipeline,
        X_test,
        y_test,
        y_pred,
        accuracy,
        precision,
        recall,
        cm,
        feature_importance
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="brand">MovieIQ</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="brand-subtitle">'
    'MOVIE ANALYTICS'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="nav-label">'
    'NAVIGATION'
    '</div>',
    unsafe_allow_html=True
)


page = st.sidebar.selectbox(

    "Navigation",

    [
        "Overview",
        "Exploratory Analysis",
        "Statistical Testing",
        "Predictive Modeling",
        "Movie Success Predictor"
    ],

    label_visibility="collapsed"
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
"""
<div class="hero-section">

<div class="hero-glow"></div>

<div class="hero-content">

<div class="hero-badge">
MOVIE ANALYTICS PLATFORM
</div>

<div class="hero-title">
MovieIQ
</div>

<div class="hero-subtitle">
Movie Success Analytics &amp; Prediction Platform
</div>

<div class="hero-description">
Transforming movie data into actionable insights
using exploratory analysis, statistical testing
and machine learning.
</div>

<div class="hero-stats">

<div class="hero-stat">
<span class="stat-dot"></span>
2,000+ Movies
</div>

<div class="hero-stat">
<span class="stat-dot"></span>
Statistical Analysis
</div>

<div class="hero-stat">
<span class="stat-dot"></span>
Random Forest ML
</div>

</div>

</div>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        'Analytics Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A high-level view of the movie dataset '
        'and success patterns.'
        '</div>',
        unsafe_allow_html=True
    )


    total_movies = len(df)

    successful_movies = (
        df["success"] == 1
    ).sum()

    unsuccessful_movies = (
        df["success"] == 0
    ).sum()

    success_rate = (
        successful_movies /
        total_movies *
        100
    )

    average_revenue = (
        df["revenue"].mean()
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Total Movies",
        f"{total_movies:,}"
    )

    c2.metric(
        "Successful",
        f"{successful_movies:,}"
    )

    c3.metric(
        "Unsuccessful",
        f"{unsuccessful_movies:,}"
    )

    c4.metric(
        "Success Rate",
        f"{success_rate:.1f}%"
    )

    c5.metric(
        "Average Revenue",
        f"${average_revenue / 1e6:.1f}M"
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    left, right = st.columns([1.5, 1])


    with left:

        st.markdown(
            '<div class="section-title">'
            'Movie Success Distribution'
            '</div>',
            unsafe_allow_html=True
        )


        success_counts = (
            df["success"].value_counts()
        )


        fig, ax = plt.subplots(
            figsize=(8, 4)
        )


        fig.patch.set_facecolor(
            "#080D1A"
        )

        ax.set_facecolor(
            "#080D1A"
        )


        sns.barplot(

            x=[
                "Not Successful",
                "Successful"
            ],

            y=[
                success_counts.get(0, 0),
                success_counts.get(1, 0)
            ],

            hue=[
                "Not Successful",
                "Successful"
            ],

            palette=[
                "#EF4444",
                "#22D3EE"
            ],

            legend=False,

            ax=ax
        )


        ax.set_xlabel(
            "Outcome",
            color="#94A3B8"
        )

        ax.set_ylabel(
            "Movies",
            color="#94A3B8"
        )

        ax.set_title(
            "Success Distribution",
            color="#F8FAFC"
        )

        ax.tick_params(
            colors="#94A3B8"
        )


        for spine in ax.spines.values():

            spine.set_color(
                "#263653"
            )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    with right:

        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            PROJECT OBJECTIVE
            </div>

            <div class="info-card-text">

            MovieIQ analyzes the relationship
            between movie characteristics and
            financial success.

            <br><br>

            The project combines exploratory
            analysis, statistical testing and
            machine learning to identify patterns
            and predict movie success.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            MACHINE LEARNING
            </div>

            <div class="info-card-text">

            Random Forest classification is used
            to predict whether a movie is likely
            to be successful based on budget,
            popularity, runtime, vote average
            and genre.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# EXPLORATORY ANALYSIS
# ============================================================

elif page == "Exploratory Analysis":

    st.markdown(
        '<div class="section-title">'
        'Exploratory Data Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore relationships between movie '
        'characteristics and financial success.'
        '</div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "Budget vs Revenue"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.scatterplot(
        data=df,
        x="budget",
        y="revenue",
        hue="success",
        alpha=0.65,
        ax=ax
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


    correlation = (
        df["budget"]
        .corr(df["revenue"])
    )


    st.info(
        f"Budget-Revenue correlation: "
        f"**{correlation:.3f}**"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Genre Distribution"
        )


        genre_counts = (
            df["genre"]
            .value_counts()
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.barplot(
            x=genre_counts.values,
            y=genre_counts.index,
            hue=genre_counts.index,
            palette="viridis",
            legend=False,
            ax=ax
        )


        ax.set_title(
            "Movies by Genre"
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    with col2:

        st.subheader(
            "Genre Success Rate"
        )


        genre_success = (
            df.groupby("genre")["success"]
            .mean()
            .sort_values(
                ascending=False
            )
            * 100
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.barplot(
            x=genre_success.values,
            y=genre_success.index,
            hue=genre_success.index,
            palette="magma",
            legend=False,
            ax=ax
        )


        ax.set_title(
            "Success Rate by Genre"
        )


        ax.set_xlim(
            0,
            100
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader(
            "Popularity"
        )


        fig, ax = plt.subplots(
            figsize=(6, 4)
        )


        sns.boxplot(
            data=df,
            x="success",
            y="popularity",
            ax=ax
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    with col2:

        st.subheader(
            "Runtime"
        )


        fig, ax = plt.subplots(
            figsize=(6, 4)
        )


        sns.boxplot(
            data=df,
            x="success",
            y="runtime",
            ax=ax
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    with col3:

        st.subheader(
            "Vote Average"
        )


        fig, ax = plt.subplots(
            figsize=(6, 4)
        )


        sns.boxplot(
            data=df,
            x="success",
            y="vote_average",
            ax=ax
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    st.subheader(
        "Correlation Heatmap"
    )


    numerical_columns = [
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "vote_average"
    ]


    correlation_matrix = (
        df[numerical_columns]
        .corr()
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        ax=ax
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


# ============================================================
# STATISTICAL TESTING
# ============================================================

elif page == "Statistical Testing":

    st.markdown(
        '<div class="section-title">'
        'Statistical Testing'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Testing whether observed differences are '
        'statistically significant.'
        '</div>',
        unsafe_allow_html=True
    )


    alpha = 0.05


    successful_popularity = df.loc[
        df["success"] == 1,
        "popularity"
    ]

    unsuccessful_popularity = df.loc[
        df["success"] == 0,
        "popularity"
    ]


    t_pop, p_pop = ttest_ind(
        successful_popularity,
        unsuccessful_popularity,
        equal_var=False
    )


    successful_runtime = df.loc[
        df["success"] == 1,
        "runtime"
    ]

    unsuccessful_runtime = df.loc[
        df["success"] == 0,
        "runtime"
    ]


    t_runtime, p_runtime = ttest_ind(
        successful_runtime,
        unsuccessful_runtime,
        equal_var=False
    )


    successful_vote = df.loc[
        df["success"] == 1,
        "vote_average"
    ]

    unsuccessful_vote = df.loc[
        df["success"] == 0,
        "vote_average"
    ]


    t_vote, p_vote = ttest_ind(
        successful_vote,
        unsuccessful_vote,
        equal_var=False
    )


    contingency_table = pd.crosstab(
        df["genre"],
        df["success"]
    )


    chi2, chi_p, dof, expected = (
        chi2_contingency(
            contingency_table
        )
    )


    results = pd.DataFrame({

        "Test": [
            "Popularity T-Test",
            "Runtime T-Test",
            "Vote Average T-Test",
            "Genre vs Success Chi-Square"
        ],

        "Statistic": [
            t_pop,
            t_runtime,
            t_vote,
            chi2
        ],

        "P-Value": [
            p_pop,
            p_runtime,
            p_vote,
            chi_p
        ],

        "Conclusion": [

            "Significant"
            if p_pop < alpha
            else "Not Significant",

            "Significant"
            if p_runtime < alpha
            else "Not Significant",

            "Significant"
            if p_vote < alpha
            else "Not Significant",

            "Significant"
            if chi_p < alpha
            else "Not Significant"
        ]
    })


    st.subheader(
        "Test Results"
    )


    st.dataframe(
        results.style.format({
            "Statistic": "{:.4f}",
            "P-Value": "{:.4f}"
        }),
        use_container_width=True
    )


    st.info(
        "Significance threshold: α = 0.05"
    )


    cols = st.columns(4)


    tests = [
        ("Popularity", p_pop),
        ("Runtime", p_runtime),
        ("Vote Average", p_vote),
        ("Genre", chi_p)
    ]


    for col, (name, pvalue) in zip(
        cols,
        tests
    ):

        result = (
            "Significant"
            if pvalue < alpha
            else "Not Significant"
        )


        col.metric(
            name,
            result,
            f"p = {pvalue:.4f}"
        )


# ============================================================
# PREDICTIVE MODELING
# ============================================================

elif page == "Predictive Modeling":

    st.markdown(
        '<div class="section-title">'
        'Random Forest Predictive Modeling'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Evaluate model performance and identify '
        'the most influential features.'
        '</div>',
        unsafe_allow_html=True
    )


    (
        model,
        X_test,
        y_test,
        y_pred,
        accuracy,
        precision,
        recall,
        cm,
        feature_importance
    ) = train_model(df)


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Confusion Matrix"
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="mako",
            xticklabels=[
                "Not Successful",
                "Successful"
            ],
            yticklabels=[
                "Not Successful",
                "Successful"
            ],
            ax=ax
        )


        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    with col2:

        st.subheader(
            "Top Feature Importance"
        )


        top_features = (
            feature_importance
            .head(10)
        )


        fig, ax = plt.subplots(
            figsize=(8, 5)
        )


        sns.barplot(
            data=top_features,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="viridis",
            legend=False,
            ax=ax
        )


        ax.set_title(
            "Top 10 Features"
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    baseline = (
        y_test
        .value_counts(
            normalize=True
        )
        .max()
    )


    st.subheader(
        "Model vs Baseline"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Baseline Accuracy",
        f"{baseline * 100:.2f}%"
    )

    c2.metric(
        "Random Forest Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# MOVIE SUCCESS PREDICTOR
# ============================================================

elif page == "Movie Success Predictor":

    st.markdown(
        '<div class="section-title">'
        'Movie Success Predictor'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-subtitle">'
        'Enter movie characteristics and let the Random '
        'Forest estimate its success probability.'
        '</div>',
        unsafe_allow_html=True
    )


    (
        model,
        X_test,
        y_test,
        y_pred,
        accuracy,
        precision,
        recall,
        cm,
        feature_importance
    ) = train_model(df)


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Movie Information"
        )


        budget = st.number_input(
            "Budget",
            min_value=0.0,
            value=float(
                df["budget"].median()
            ),
            step=1000000.0,
            format="%.2f"
        )


        popularity = st.number_input(
            "Popularity",
            min_value=0.0,
            value=float(
                df["popularity"].median()
            ),
            step=1.0,
            format="%.2f"
        )


        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=1,
            max_value=300,
            value=int(
                df["runtime"].median()
            ),
            step=1
        )


    with col2:

        st.subheader(
            "Audience Information"
        )


        vote_average = st.number_input(
            "Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=float(
                df["vote_average"].median()
            ),
            step=0.1,
            format="%.2f"
        )


        genre = st.selectbox(
            "Genre",
            sorted(
                df["genre"].unique()
            )
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    if st.button(
        "Predict Movie Success",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "budget": [budget],

            "popularity": [popularity],

            "runtime": [runtime],

            "vote_average": [vote_average],

            "genre": [genre]
        })


        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0]


        success_probability = (
            probability[1] * 100
        )


        if prediction == 1:

            result_text = "LIKELY SUCCESSFUL"

        else:

            result_text = "LIKELY NOT SUCCESSFUL"


        # ====================================================
        # PREDICTION CARD
        # IMPORTANT:
        # No nested HTML is used here.
        # This prevents HTML tags from appearing visibly.
        # ====================================================

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">
                    MODEL PREDICTION
                </div>
                <div class="prediction-result">
                    {result_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        c1, c2 = st.columns(2)


        c1.metric(
            "Success Probability",
            f"{success_probability:.2f}%"
        )


        c2.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )


        st.progress(
            min(
                max(
                    int(success_probability),
                    0
                ),
                100
            )
        )


        st.caption(
            "The prediction is generated using "
            "the Random Forest model trained on "
            "the MovieIQ dataset."
        )
