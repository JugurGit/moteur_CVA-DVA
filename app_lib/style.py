# app_lib/style.py
import streamlit as st

def apply_page_config(title: str = "XVA HW1F++ — Streamlit", icon: str = "📊"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")

def apply_css():
    css = """
    <style>
    /* Layout */
    .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
    section[data-testid="stSidebar"] { width: 340px !important; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

    /* Headings */
    h1, h2, h3 { letter-spacing: -0.02em; }
    h1 { font-weight: 800; }
    h2 { font-weight: 750; }

    /* Cards / Metrics */
    [data-testid="stMetric"] {
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }
    [data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.9; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }

    /* Tables */
    div[data-testid="stDataFrame"]{
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox div, .stMultiSelect div {
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 14px;
        padding: 0.55rem 0.85rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    /* Code blocks */
    pre {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03) !important;
    }

    /* Subtle separators */
    hr { opacity: 0.25; }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
