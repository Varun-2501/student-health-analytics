import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import os

# Auto-install markdown if missing (needed to convert AI responses to HTML)
try:
    import markdown as md_lib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "-q"])
    import markdown as md_lib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pdfplumber

from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# RAG IMPORTS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# FIX 1: Updated deprecated import



# -------------------------
# PAGE CONFIG & CUSTOM CSS
# -------------------------

st.set_page_config(
    page_title="Student Health Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Design Tokens ─────────────────────────────────────────────────────── */
    :root {
        --navy:      #0B2545;
        --navy-mid:  #13315C;
        --teal:      #0E9AA7;
        --teal-dark: #0C7A84;
        --teal-soft: #E0F7FA;
        --blue:      #1565C0;
        --blue-soft: #E3F2FD;
        --green:     #2E7D32;
        --green-soft:#E8F5E9;
        --amber:     #E65100;
        --amber-soft:#FFF3E0;
        --red:       #C62828;
        --red-soft:  #FFEBEE;
        --slate:     #F4F6F9;
        --border:    #DDE3EC;
        --text-main: #0B2545;
        --text-sub:  #4A5568;
        --white:     #FFFFFF;
        --shadow-sm: 0 1px 4px rgba(11,37,69,0.08);
        --shadow-md: 0 4px 16px rgba(11,37,69,0.10);
        --shadow-lg: 0 8px 32px rgba(11,37,69,0.13);
        --radius:    10px;
    }

    /* ── Global ────────────────────────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .main .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1280px; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ── Sidebar ───────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.92rem; padding: 0.45rem 0.6rem;
        border-radius: 6px; transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10); }

    /* ── Page header banner ────────────────────────────────────────────────── */
    .page-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #1a4a7a 100%);
        padding: 2rem 2.5rem;
        border-radius: var(--radius);
        color: white;
        margin-bottom: 1.75rem;
        box-shadow: var(--shadow-lg);
        border-left: 5px solid var(--teal);
        position: relative;
        overflow: hidden;
    }
    .page-header::after {
        content: '';
        position: absolute; top: -40px; right: -40px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(14,154,167,0.12);
    }
    .page-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.3px; }
    .page-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 0.95rem; }

    /* ── Section label ─────────────────────────────────────────────────────── */
    .section-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; color: var(--teal);
        margin: 1.75rem 0 0.5rem;
        display: flex; align-items: center; gap: 6px;
    }
    .section-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, var(--teal), transparent);
        opacity: 0.3;
    }

    /* ── Metric cards ──────────────────────────────────────────────────────── */
    .kpi-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        display: flex; align-items: center; gap: 1rem;
    }
    .kpi-icon {
        width: 46px; height: 46px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; flex-shrink: 0;
    }
    .kpi-icon.teal   { background: var(--teal-soft); }
    .kpi-icon.blue   { background: var(--blue-soft); }
    .kpi-icon.amber  { background: var(--amber-soft); }
    .kpi-icon.red    { background: var(--red-soft); }
    .kpi-value { font-size: 1.65rem; font-weight: 700; color: var(--text-main); line-height: 1; }
    .kpi-label { font-size: 0.8rem; color: var(--text-sub); margin-top: 2px; }

    /* ── Info / insight cards ──────────────────────────────────────────────── */
    .info-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        border-top: 3px solid var(--teal);
        margin: 0.5rem 0;
    }
    .info-card h4  { margin: 0 0 0.6rem; font-size: 0.88rem; font-weight: 600; color: var(--text-main) !important; }
    .info-card p   { margin: 0.2rem 0; font-size: 0.85rem; color: var(--text-sub) !important; }
    /* Cover AI-generated markdown: lists, spans, links, bold, italic */
    .info-card ul, .info-card ol {
        margin: 0.5rem 0 0.25rem 1.25rem; padding: 0;
        color: var(--text-sub) !important;
    }
    .info-card li  { font-size: 0.88rem; line-height: 1.7; color: var(--text-sub) !important; margin: 0.1rem 0; }
    .info-card strong, .info-card b { color: var(--text-main) !important; }
    .info-card em, .info-card span  { color: var(--text-sub)  !important; }
    .info-card a   { color: var(--teal) !important; }
    /* ai-answer variant — slightly more generous padding for long responses */
    .info-card.ai-answer { padding: 1.5rem 1.75rem; }
    .info-card.ai-answer p, .info-card.ai-answer li { font-size: 0.9rem; line-height: 1.75; }

    /* ── AI response text — force dark color on all markdown output ─────────── */
    .ai-resp, .ai-resp * {
        color: var(--text-main) !important;
        font-size: 0.9rem; line-height: 1.75;
    }
    .ai-resp ul, .ai-resp ol { margin: 0.4rem 0 0.4rem 1.3rem; padding: 0; }
    .ai-resp li  { margin: 0.2rem 0; }
    .ai-resp p   { margin: 0.4rem 0; }
    .ai-resp strong, .ai-resp b { font-weight: 700; color: var(--text-main) !important; }
    .ai-resp h1, .ai-resp h2, .ai-resp h3 { color: var(--navy) !important; margin: 0.75rem 0 0.3rem; }

    /* ── Risk badges ───────────────────────────────────────────────────────── */
    .risk-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 0.45rem 1.1rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; letter-spacing: 0.2px;
    }
    .risk-high     { background: var(--red-soft);   color: var(--red);   border: 1px solid #FFCDD2; }
    .risk-moderate { background: var(--amber-soft);  color: var(--amber); border: 1px solid #FFE0B2; }
    .risk-low      { background: var(--green-soft);  color: var(--green); border: 1px solid #C8E6C9; }

    /* ── Result banners (prediction) ───────────────────────────────────────── */
    .result-banner {
        border-radius: var(--radius); padding: 2rem;
        text-align: center; margin: 1rem 0;
        box-shadow: var(--shadow-md);
    }
    .result-banner.high   { background: linear-gradient(135deg, #C62828, #E53935); color: white; }
    .result-banner.moderate { background: linear-gradient(135deg, #E65100, #F4511E); color: white; }
    .result-banner.low    { background: linear-gradient(135deg, #2E7D32, #388E3C); color: white; }
    .result-banner .result-icon  { font-size: 2.8rem; margin-bottom: 0.5rem; }
    .result-banner .result-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-banner .result-sub   { opacity: 0.85; font-size: 0.95rem; margin-top: 0.3rem; }

    /* ── Buttons ───────────────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--teal) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important; font-size: 0.88rem !important;
        letter-spacing: 0.2px;
        transition: background 0.2s, box-shadow 0.2s, transform 0.15s !important;
        box-shadow: 0 2px 8px rgba(14,154,167,0.25) !important;
    }
    .stButton > button:hover {
        background: var(--teal-dark) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(14,154,167,0.35) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Quick action buttons ──────────────────────────────────────────────── */
    .qa-btn > button {
        background: var(--white) !important;
        color: var(--navy) !important;
        border: 1.5px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
        font-weight: 500 !important;
    }
    .qa-btn > button:hover {
        background: var(--teal-soft) !important;
        border-color: var(--teal) !important;
        color: var(--teal-dark) !important;
        box-shadow: 0 2px 8px rgba(14,154,167,0.18) !important;
    }

    /* ── Text inputs ───────────────────────────────────────────────────────── */
    .stTextArea textarea, .stTextInput input {
        border: 1.5px solid var(--border) !important;
        border-radius: 8px !important;
        background: var(--white) !important;
        color: var(--text-main) !important;
        font-size: 0.92rem !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(14,154,167,0.12) !important;
    }

    /* ── Select / slider ───────────────────────────────────────────────────── */
    .stSelectbox > div > div { border-radius: 8px !important; }

    /* ── Dataframe ─────────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius); overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    /* ── Expander ──────────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--slate) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
    }

    /* ── Chat bubbles ──────────────────────────────────────────────────────── */
    .chat-wrap { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1rem; }
    .chat-row { display: flex; gap: 0.75rem; align-items: flex-start; }
    .chat-row.user-row { flex-direction: row-reverse; }
    .chat-avatar {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0; margin-top: 2px;
    }
    .chat-avatar.user-av  { background: var(--blue-soft);  }
    .chat-avatar.bot-av   { background: var(--teal-soft);  }
    .chat-bubble {
        max-width: 78%; padding: 0.85rem 1.1rem;
        border-radius: 14px; font-size: 0.9rem;
        line-height: 1.65; color: var(--text-main) !important;
        box-shadow: var(--shadow-sm);
    }
    .chat-bubble * { color: var(--text-main) !important; }
    .bubble-user {
        background: var(--blue-soft);
        border-bottom-right-radius: 4px;
        border: 1px solid #BBDEFB;
    }
    .bubble-bot {
        background: var(--teal-soft);
        border-bottom-left-radius: 4px;
        border: 1px solid #B2EBF2;
    }
    .bubble-label {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.5px; text-transform: uppercase;
        margin-bottom: 4px; opacity: 0.6;
    }

    /* ── Alert / notice box ────────────────────────────────────────────────── */
    .notice-box {
        background: var(--teal-soft); border: 1px solid #80DEEA;
        border-left: 4px solid var(--teal);
        border-radius: var(--radius); padding: 1rem 1.25rem;
        color: var(--teal-dark) !important; font-size: 0.88rem;
    }
    .no-data-box {
        background: var(--slate); border: 2px dashed var(--border);
        border-radius: var(--radius); padding: 3.5rem 2rem;
        text-align: center;
    }
    .no-data-box h3 { color: var(--text-main); margin: 0 0 0.5rem; }
    .no-data-box p  { color: var(--text-sub);  margin: 0; font-size: 0.9rem; }

    /* ── Student profile ───────────────────────────────────────────────────── */
    .profile-card {
        background: var(--white); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 1.5rem 2rem;
        box-shadow: var(--shadow-md);
        border-top: 4px solid var(--teal);
    }

    /* ── Upload zone ───────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--teal) !important;
        border-radius: var(--radius) !important;
        background: var(--teal-soft) !important;
    }

    /* ── Sidebar status pill ───────────────────────────────────────────────── */
    .status-pill {
        display: flex; align-items: center; gap: 8px;
        padding: 0.6rem 0.9rem; border-radius: 8px;
        font-size: 0.82rem; font-weight: 600;
    }
    .status-ok  { background: rgba(46,125,50,0.2);  color: #81C784; border: 1px solid rgba(129,199,132,0.3); }
    .status-err { background: rgba(198,40,40,0.2);  color: #EF9A9A; border: 1px solid rgba(239,154,154,0.3); }

    /* ── Sidebar nav logo ──────────────────────────────────────────────────── */
    .sidebar-logo {
        padding: 1.25rem 1rem 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.5rem;
    }
    .sidebar-logo h2 {
        color: white !important; margin: 0;
        font-size: 1.1rem; font-weight: 700; letter-spacing: -0.2px;
    }
    .sidebar-logo p {
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.75rem; margin: 2px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# FIX 2: API KEY — load from env or st.secrets, never hardcoded
# -------------------------

def _load_api_key():
    # 1. Environment variable (most reliable)
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # 2. Streamlit secrets (only if file exists and key present)
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 3. Hard-coded fallback — replace with your actual key if not using env vars
    return ""

api_key = _load_api_key()

if not api_key:
    st.sidebar.warning("⚠️ Set GROQ_API_KEY in environment or Streamlit secrets.")

# FIX: Always create llm — guard was hiding real errors
try:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )
except Exception as e:
    llm = None
    st.sidebar.error(f"LLM init failed: {e}")


# -------------------------
# HEALTH GUIDELINES
# -------------------------

def health_guidelines(query):
    guidelines = {
        "stress": "Reduce workload, meditate, maintain sleep cycle.",
        "sleep": "Maintain 7-8 hours sleep.",
        "heart": "Exercise regularly and maintain balanced diet.",
        "mental": "Consult counselor and maintain work-life balance."
    }
    results = [v for k, v in guidelines.items() if k in query.lower()]
    return "\n".join(results) if results else "Maintain balanced diet, sleep properly and exercise."


# -------------------------
# LANGGRAPH AI ASSISTANT
# -------------------------

# FIX: Use plain TypedDict without Annotated reducer — the add reducer
# causes messages to be merged unexpectedly when passing {**state, "messages": [...]}
# and produces duplicate/empty message lists that break llm.invoke()
class HealthState(TypedDict):
    query: str
    context: str
    messages: list
    response: str


def create_prompt_node(state: HealthState) -> HealthState:
    query = state["query"]
    context = health_guidelines(query)
    prompt = (
        "You are an AI Student Health Assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query:\n{query}\n\n"
        "Provide clear, helpful health advice."
    )
    return {
        "query": query,
        "context": context,
        "messages": [HumanMessage(content=prompt)],
        "response": ""
    }


def generate_response_node(state: HealthState) -> HealthState:
    if llm is None:
        return {**state, "response": "⚠️ AI assistant unavailable: missing or invalid API key."}
    try:
        response = llm.invoke(state["messages"])
        text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        text = f"⚠️ Error getting response: {str(e)}"
    return {
        **state,
        "response": text,
        "messages": state["messages"] + [AIMessage(content=text)]
    }


def build_graph():
    graph = StateGraph(HealthState)
    graph.add_node("prompt", create_prompt_node)
    graph.add_node("response", generate_response_node)
    graph.add_edge(START, "prompt")
    graph.add_edge("prompt", "response")
    graph.add_edge("response", END)
    return graph.compile()


# Cache the compiled workflow so it isn't rebuilt on every Streamlit rerun
@st.cache_resource
def get_workflow():
    return build_graph()

workflow = get_workflow()


# -------------------------
# FIX 4: Cache model training so it doesn't retrain on every interaction
# -------------------------

@st.cache_resource
def train_mental_model(_df_hash, df):
    df_model = df.copy()
    df_model["Sleep_Quality"] = df_model["Sleep_Quality"].replace({"Poor": 0, "Average": 1, "Good": 2})
    df_model["Mood"] = df_model["Mood"].replace({"Sad": 0, "Neutral": 1, "Happy": 2})
    df_model["Health_Risk_Level"] = df_model["Health_Risk_Level"].replace({"Low": 0, "Moderate": 1, "High": 2})
    df_model = df_model.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df_model[["Stress_Level_Biosensor", "Sleep_Quality", "Physical_Activity"]]
    y = df_model["Health_Risk_Level"]
    # FIX 5: Guard against too-small datasets
    if len(df_model) < 5:
        st.error("Dataset too small to train a model (need at least 5 rows).")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# -------------------------
# FIX 6: PDF extractor — wrap UploadedFile in BytesIO for reliability
# -------------------------

def extract_pdf_text(uploaded_file):
    try:
        file_bytes = io.BytesIO(uploaded_file.read())
        text = ""
        with pdfplumber.open(file_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text.strip() else "No readable text found."
    except Exception as e:
        return f"PDF reading error: {str(e)}"


# -------------------------
# FIX 7: Cache RAG index in session_state so it isn't rebuilt on every question
# -------------------------

def create_rag_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    return docs


def rag_query(docs, question):
    if llm is None:
        return "⚠️ AI assistant unavailable"

    # simple retrieval (top 3 chunks)
    context = "\n".join([d.page_content for d in docs[:3]])

    prompt = f"""
You are a medical AI assistant.

Context:
{context}

Question:
{question}

Give clear medical insights.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🏥 HealthAnalytics</h2>
        <p>Student Wellness Platform</p>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Advanced Analytics", "👤 Student Analysis",
         "🧠 Mental Health Prediction", "📄 PDF Report Analyzer", "🤖 AI Health Assistant"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if "data" in st.session_state:
        st.markdown('<div class="status-pill status-ok">✅ &nbsp;Dataset Loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-err">⚠️ &nbsp;No Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; padding: 0.9rem; background: rgba(255,255,255,0.06);
                border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
        <p style="color:rgba(255,255,255,0.5) !important; font-size:0.72rem;
                  text-transform:uppercase; letter-spacing:1px; margin:0 0 0.6rem;">Quick Tips</p>
        <p style="color:rgba(255,255,255,0.7) !important; font-size:0.82rem; margin:0; line-height:1.7;">
            1. Upload a CSV on the Dashboard<br>
            2. Explore analytics &amp; trends<br>
            3. Predict health risk scores<br>
            4. Ask the AI assistant
        </p>
    </div>
    """, unsafe_allow_html=True)



# -------------------------
# SAFE COLUMN HELPERS
# -------------------------

# Ordinal maps for columns that may be stored as strings in the CSV
_ORDINAL_MAPS = {
    "Physical_Activity": {"Low": 1, "Moderate": 2, "High": 3},
    "Sleep_Quality":     {"Poor": 1, "Average": 2, "Good": 3},
    "Mood":              {"Sad": 1, "Neutral": 2, "Happy": 3},
    "Health_Risk_Level": {"Low": 1, "Moderate": 2, "High": 3},
}

def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Return a numeric version of a Series.
    If it's already numeric, return as-is.
    If it maps to a known ordinal, convert via that map.
    Otherwise attempt pd.to_numeric, dropping failures.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    col = series.name
    if col in _ORDINAL_MAPS:
        return series.map(_ORDINAL_MAPS[col]).fillna(0)
    return pd.to_numeric(series, errors="coerce").fillna(0)

def _safe_mean(df: pd.DataFrame, col: str):
    """Return mean of a column as a rounded float, or 'N/A' if impossible."""
    if col not in df.columns:
        return "N/A"
    try:
        return round(_to_numeric_series(df[col]).mean(), 1)
    except Exception:
        return "N/A"

def _safe_idxmax(df: pd.DataFrame, col: str):
    """Return the row with the max value of col (numeric-safe)."""
    return df.loc[_to_numeric_series(df[col]).idxmax()]

def _activity_label(df: pd.DataFrame) -> tuple:
    """Return (display_value, status_string) for Physical_Activity insight card."""
    col = "Physical_Activity"
    if col not in df.columns:
        return "N/A", "N/A"
    s = _to_numeric_series(df[col])
    avg = s.mean()
    # If original data was ordinal (1-3 scale) show label, else show raw number
    if df[col].dtype == object:
        label = f"{round(avg, 2)} / 3"
        status = "Good ✅" if avg >= 2 else "Needs Improvement ⚠️"
    else:
        label = str(round(avg, 2))
        status = "Good ✅" if avg > 5 else "Needs Improvement ⚠️"
    return label, status


def _ai_card(title: str, answer: str, margin_top: str = "1rem"):
    """Render an AI response as a self-contained white card with dark text.
    Uses md_lib to convert markdown → HTML, all in one st.markdown() call.
    """
    try:
        body_html = md_lib.markdown(answer, extensions=["nl2br", "sane_lists"])
    except Exception:
        # Fallback: plain text wrapped in paragraphs
        body_html = "".join(f"<p>{line}</p>" for line in answer.split("\n") if line.strip())

    html = (
        '<div style="background:#ffffff !important; border:1px solid #DDE3EC !important;'
        'border-top:3px solid #0E9AA7 !important; border-radius:10px;'
        'padding:1.25rem 1.6rem 1.4rem; box-shadow:0 2px 8px rgba(11,37,69,0.10);'
        f'margin-top:{margin_top}">'
        '<span style="display:block; font-size:0.72rem !important; font-weight:700 !important;'
        'letter-spacing:1.1px; text-transform:uppercase; color:#0E9AA7 !important;'
        'margin-bottom:0.9rem">' + title + '</span>'
        '<div style="color:#0B2545 !important; font-size:0.9rem; line-height:1.75">'
        + body_html +
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# -------------------------
# DASHBOARD
# -------------------------

if menu == "🏠 Dashboard":
    st.markdown("""
    <div class="page-header">
        <h1>🏥 Student Health Analytics</h1>
        <p>Comprehensive health monitoring &amp; insights platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📁 Upload Dataset</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file = st.file_uploader("Drag and drop your CSV file here", type=["csv"],
                                help="Upload a CSV file containing student health data")

    if file:
        df = pd.read_csv(file)
        st.session_state["data"] = df
        st.markdown('<div class="notice-box">✅ &nbsp;<strong>Dataset successfully loaded.</strong></div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="section-label">📈 Key Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon teal">👥</div>
                <div><div class="kpi-value">{len(df)}</div><div class="kpi-label">Total Students</div></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            hr_avg = _safe_mean(df, "Heart_Rate")
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon red">❤️</div>
                <div><div class="kpi-value">{hr_avg}</div><div class="kpi-label">Avg Heart Rate (bpm)</div></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            sl_avg = _safe_mean(df, "Stress_Level_Biosensor")
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon amber">😰</div>
                <div><div class="kpi-value">{sl_avg}</div><div class="kpi-label">Avg Stress Level</div></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            high_risk = len(df[df["Health_Risk_Level"] == "High"]) if "Health_Risk_Level" in df.columns else 0
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-icon blue">⚠️</div>
                <div><div class="kpi-value">{high_risk}</div><div class="kpi-label">High Risk Students</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Data Preview</div>', unsafe_allow_html=True)
        with st.expander("View Dataset", expanded=True):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label">📊 Visual Analytics</div>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)

        PLOT_LAYOUT = dict(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0B2545", size=12),
            margin=dict(t=30, b=20, l=10, r=10)
        )

        with chart_col1:
            st.markdown("**❤️ Heart Rate Distribution**")
            if "Heart_Rate" in df.columns:
                fig1 = px.histogram(df, x="Heart_Rate", nbins=20, color_discrete_sequence=["#0E9AA7"])
                fig1.update_layout(**PLOT_LAYOUT, showlegend=False)
                fig1.update_traces(marker_line_color="white", marker_line_width=1)
                st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            st.markdown("**🎯 Health Risk Distribution**")
            if "Health_Risk_Level" in df.columns:
                fig3 = px.pie(df, names="Health_Risk_Level", color="Health_Risk_Level",
                              color_discrete_map={"Low": "#2E7D32", "Moderate": "#E65100", "High": "#C62828"},
                              hole=0.45)
                fig3.update_layout(**PLOT_LAYOUT)
                fig3.update_traces(textfont_size=13)
                st.plotly_chart(fig3, use_container_width=True)

        if all(c in df.columns for c in ["Physical_Activity", "Sleep_Quality", "Stress_Level_Biosensor", "Heart_Rate", "Student_ID"]):
            st.markdown("**🏃 Physical Activity vs Sleep Quality**")
            # Convert ordinal columns to numeric for scatter plot
            scatter_df = df.copy()
            scatter_df["_activity_num"] = _to_numeric_series(df["Physical_Activity"])
            scatter_df["_sleep_num"]    = _to_numeric_series(df["Sleep_Quality"])
            scatter_df["_stress_num"]   = _to_numeric_series(df["Stress_Level_Biosensor"])
            fig2 = px.scatter(scatter_df, x="_activity_num", y="_sleep_num",
                              color="_stress_num", size="Heart_Rate",
                              color_continuous_scale="Teal", hover_data=["Student_ID"],
                              labels={"_activity_num": "Physical Activity",
                                      "_sleep_num": "Sleep Quality",
                                      "_stress_num": "Stress Level"})
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown("""
        <div class="no-data-box">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">📂</div>
            <h3>No Dataset Loaded</h3>
            <p>Upload a CSV file above to start exploring student health analytics.</p>
        </div>""", unsafe_allow_html=True)


# -------------------------
# ADVANCED ANALYTICS
# -------------------------

elif menu == "📊 Advanced Analytics":
    st.markdown("""<div class="page-header">
        <h1>📊 Advanced Analytics</h1>
        <p>Deep-dive correlations, distributions, and health patterns</p>
    </div>""", unsafe_allow_html=True)

    PLOT_LAYOUT = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0B2545", size=12), margin=dict(t=30, b=20, l=10, r=10)
    )

    if "data" not in st.session_state:
        st.markdown("""<div class="no-data-box"><div style="font-size:2rem;margin-bottom:.5rem">📊</div>
            <h3>Dataset Required</h3><p>Please upload a dataset on the Dashboard first.</p></div>""",
            unsafe_allow_html=True)
    else:
        df = st.session_state["data"]

        st.markdown('<div class="section-label">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        # Build a fully numeric DataFrame: keep existing numerics + convert known ordinals
        numeric_df = df.select_dtypes(include=np.number).copy()
        for col, mapping in _ORDINAL_MAPS.items():
            if col in df.columns and col not in numeric_df.columns:
                numeric_df[col] = df[col].map(mapping).fillna(0)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale=[[0,"#C62828"],[0.5,"#F4F6F9"],[1,"#0E9AA7"]])
            fig.update_layout(**PLOT_LAYOUT, height=480)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">📈 Distribution Charts</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**😊 Mood vs Stress Level**")
            if all(c in df.columns for c in ["Mood", "Stress_Level_Biosensor"]):
                box_df = df.copy()
                box_df["_stress_num"] = _to_numeric_series(df["Stress_Level_Biosensor"])
                fig2 = px.box(box_df, x="Mood", y="_stress_num", color="Mood",
                              color_discrete_map={"Happy": "#2E7D32", "Neutral": "#E65100", "Sad": "#C62828"},
                              labels={"_stress_num": "Stress Level"})
                fig2.update_layout(**PLOT_LAYOUT, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("**📚 Study Hours vs Stress**")
            if all(c in df.columns for c in ["Study_Hours", "Stress_Level_Biosensor", "Sleep_Quality"]):
                try:
                    import statsmodels; trendline = "ols"
                except ImportError:
                    trendline = None
                scatter2_df = df.copy()
                scatter2_df["_stress_num"] = _to_numeric_series(df["Stress_Level_Biosensor"])
                scatter2_df["_study_num"]  = pd.to_numeric(df["Study_Hours"], errors="coerce").fillna(0)
                fig3 = px.scatter(scatter2_df, x="_study_num", y="_stress_num",
                                  color="Sleep_Quality",
                                  color_discrete_map={"Good": "#2E7D32", "Average": "#E65100", "Poor": "#C62828"},
                                  trendline=trendline,
                                  labels={"_study_num": "Study Hours", "_stress_num": "Stress Level"})
                fig3.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-label">💡 Key Insights</div>', unsafe_allow_html=True)
        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            if "Stress_Level_Biosensor" in df.columns:
                highest_stress = _safe_idxmax(df, "Stress_Level_Biosensor")
                st.markdown(f"""<div class="info-card" style="border-top-color:#C62828">
                    <h4>🔴 Highest Stress Student</h4>
                    <p><strong>ID:</strong> {highest_stress.get("Student_ID", "N/A")}</p>
                    <p><strong>Stress Score:</strong> {highest_stress["Stress_Level_Biosensor"]}</p>
                </div>""", unsafe_allow_html=True)

        with insight_col2:
            if "Physical_Activity" in df.columns:
                act_val, act_status = _activity_label(df)
                st.markdown(f"""<div class="info-card">
                    <h4>🏃 Average Physical Activity</h4>
                    <p><strong>Score:</strong> {act_val}</p>
                    <p><strong>Status:</strong> {act_status}</p>
                </div>""", unsafe_allow_html=True)

        with insight_col3:
            if "Sleep_Quality" in df.columns:
                poor_sleep = len(df[df["Sleep_Quality"] == "Poor"])
                st.markdown(f"""<div class="info-card" style="border-top-color:#E65100">
                    <h4>😴 Poor Sleep Quality</h4>
                    <p><strong>Count:</strong> {poor_sleep} students</p>
                    <p><strong>Percentage:</strong> {round(poor_sleep / len(df) * 100, 1)}%</p>
                </div>""", unsafe_allow_html=True)


# -------------------------
# STUDENT ANALYSIS
# -------------------------

elif menu == "👤 Student Analysis":
    st.markdown("""<div class="page-header">
        <h1>👤 Individual Student Analysis</h1>
        <p>Detailed health profile and personalised recommendations</p>
    </div>""", unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.markdown("""<div class="no-data-box"><div style="font-size:2rem;margin-bottom:.5rem">👤</div>
            <h3>Dataset Required</h3><p>Please upload a dataset on the Dashboard first.</p></div>""",
            unsafe_allow_html=True)
    else:
        df = st.session_state["data"]

        st.markdown('<div class="section-label">🔍 Select Student</div>', unsafe_allow_html=True)
        sel_col1, sel_col2 = st.columns([1, 3])
        with sel_col1:
            student_id = st.selectbox("Student ID", df["Student_ID"], label_visibility="collapsed")

        student = df[df["Student_ID"] == student_id].iloc[0]
        risk = student.get("Health_Risk_Level", "Unknown")

        with sel_col2:
            if risk == "High":
                badge = '<span class="risk-badge risk-high">🔴 High Risk</span>'
            elif risk == "Moderate":
                badge = '<span class="risk-badge risk-moderate">🟡 Moderate Risk</span>'
            else:
                badge = '<span class="risk-badge risk-low">🟢 Low Risk</span>'
            st.markdown(f'<div style="text-align:right; padding-top:0.4rem">{badge}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-label">📋 Health Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("❤️ Heart Rate",    f"{student.get('Heart_Rate','N/A')} bpm")
            st.metric("😰 Stress Level",  student.get("Stress_Level_Biosensor","N/A"))
        with p2:
            st.metric("😴 Sleep Quality", student.get("Sleep_Quality","N/A"))
            st.metric("🏃 Activity",       student.get("Physical_Activity","N/A"))
        with p3:
            st.metric("📚 Study Hours",   f"{student.get('Study_Hours','N/A')} hrs")
            st.metric("😊 Mood",           student.get("Mood","N/A"))
        with p4:
            st.metric("🎯 Risk Level",    student.get("Health_Risk_Level","N/A"))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">💊 Recommendations</div>', unsafe_allow_html=True)
        if risk == "High":
            st.error("**⚠️ Immediate Attention Required**")
            st.markdown("- 🏥 Schedule a health check-up with campus medical services\n- 🧘 Consider stress management counselling\n- 😴 Prioritise improving sleep quality\n- 🏃 Gradually increase physical activity")
        elif risk == "Moderate":
            st.warning("**⚡ Proactive Measures Recommended**")
            st.markdown("- 📅 Maintain a consistent sleep schedule\n- ⏰ Take regular breaks during study sessions\n- 🥗 Focus on balanced nutrition\n- 🧘 Practice relaxation techniques")
        else:
            st.success("**✅ Keep Up the Good Work!**")
            st.markdown("- 🎯 Continue current healthy habits\n- 📊 Regular self-monitoring\n- 🤝 Support peers who may need help\n- 🌟 Consider mentoring others")


# -------------------------
# MENTAL HEALTH PREDICTION
# -------------------------

elif menu == "🧠 Mental Health Prediction":
    st.markdown("""<div class="page-header">
        <h1>🧠 Mental Health Risk Prediction</h1>
        <p>AI-powered risk assessment based on key health indicators</p>
    </div>""", unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.markdown("""<div class="no-data-box"><div style="font-size:2rem;margin-bottom:.5rem">🧠</div>
            <h3>Dataset Required</h3><p>Please upload a dataset on the Dashboard to train the model.</p></div>""",
            unsafe_allow_html=True)
    else:
        df = st.session_state["data"]
        model = train_mental_model(id(df), df)

        if model is not None:
            st.markdown('<div class="section-label">📝 Enter Health Parameters</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                stress = st.slider("😰 Stress Level (1–10)", min_value=1, max_value=10, value=5)
            with col2:
                sleep_option = st.selectbox("😴 Sleep Quality", ["Poor", "Average", "Good"], index=1)
            with col3:
                activity = st.slider("🏃 Physical Activity (0–10)", min_value=0, max_value=10, value=5)

            st.markdown("<br>", unsafe_allow_html=True)
            _, btn_col, _ = st.columns([1, 1, 1])
            with btn_col:
                predict_button = st.button("🔮 Predict Risk Level", use_container_width=True)

            if predict_button:
                sleep_map = {"Poor": 0, "Average": 1, "Good": 2}
                pred = model.predict([[stress, sleep_map[sleep_option], activity]])[0]
                risk = {0: "Low", 1: "Moderate", 2: "High"}.get(int(pred), "Low")

                st.markdown("<br>", unsafe_allow_html=True)
                if risk == "High":
                    st.markdown("""<div class="result-banner high">
                        <div class="result-icon">🔴</div>
                        <div class="result-title">High Risk</div>
                        <div class="result-sub">Immediate attention recommended</div>
                    </div>""", unsafe_allow_html=True)
                    st.error("**Recommended Actions:**")
                    st.markdown("- 🏥 Consult with a mental health professional\n- 🧘 Practice daily stress-relief activities\n- 😴 Prioritise 7–8 hours of quality sleep\n- 🏃 Incorporate regular physical exercise")
                elif risk == "Moderate":
                    st.markdown("""<div class="result-banner moderate">
                        <div class="result-icon">🟡</div>
                        <div class="result-title">Moderate Risk</div>
                        <div class="result-sub">Preventive measures advised</div>
                    </div>""", unsafe_allow_html=True)
                    st.warning("**Recommended Actions:**")
                    st.markdown("- 📅 Establish a consistent daily routine\n- 🧘 Practice mindfulness or meditation\n- 🥗 Maintain a balanced diet\n- 💬 Stay connected with friends and family")
                else:
                    st.markdown("""<div class="result-banner low">
                        <div class="result-icon">🟢</div>
                        <div class="result-title">Low Risk</div>
                        <div class="result-sub">Great health indicators!</div>
                    </div>""", unsafe_allow_html=True)
                    st.success("**Keep Up These Habits:**")
                    st.markdown("- ✅ Continue your current healthy lifestyle\n- 📊 Regular self-assessment\n- 🎯 Set and achieve personal goals\n- 🤝 Help others maintain good health")


# -------------------------
# PDF HEALTH REPORT ANALYZER
# -------------------------

elif menu == "📄 PDF Report Analyzer":
    st.markdown("""<div class="page-header">
        <h1>📄 PDF Health Report Analyzer</h1>
        <p>Upload medical reports and get AI-powered insights instantly</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">📤 Upload Report</div>', unsafe_allow_html=True)
    upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])
    with upload_col2:
        pdf_file = st.file_uploader("Drop your PDF file here", type=["pdf"],
                                    help="Upload a health report in PDF format")

    if pdf_file:
        with st.spinner("📖 Extracting text…"):
            text = extract_pdf_text(pdf_file)

        st.markdown('<div class="section-label">📋 Extracted Content</div>', unsafe_allow_html=True)
        with st.expander("View Extracted Text", expanded=False):
            preview = text[:2000] + ("…" if len(text) > 2000 else "")
            # Use a styled pre block — st.text_area with disabled=True renders
            # white text on white background in Streamlit's dark theme
            import html as html_lib
            safe_text = html_lib.escape(preview)
            st.markdown(
                '<div style="background:#f8f9fa !important; border:1px solid #DDE3EC !important;'
                'border-radius:8px; padding:1rem 1.25rem; max-height:200px; overflow-y:auto">'
                '<pre style="margin:0; white-space:pre-wrap; word-wrap:break-word;'
                'font-family:monospace; font-size:0.82rem; color:#0B2545 !important;'
                'line-height:1.6; background:transparent !important">'
                + safe_text +
                '</pre></div>',
                unsafe_allow_html=True
            )

        if "error" not in text.lower() and text != "No readable text found.":
            pdf_key = f"vectorstore_{pdf_file.name}"
            if pdf_key not in st.session_state:
                with st.spinner("🧠 Building AI knowledge base…"):
                    st.session_state[pdf_key] = create_rag_index(text)

            st.markdown('<div class="notice-box">✅ &nbsp;<strong>AI Knowledge Base Ready.</strong> Ask questions below.</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="section-label">💬 Ask Questions</div>', unsafe_allow_html=True)
            st.markdown("**Quick Questions:**")
            s1, s2, s3 = st.columns(3)

            def _ask_pdf(q):
                st.session_state["pdf_question"] = q

            with s1:
                if st.button("📊 Summarise Findings"):
                    _ask_pdf("Summarize the key findings from this health report")
            with s2:
                if st.button("⚠️ Any Concerns?"):
                    _ask_pdf("Are there any health concerns mentioned in this report?")
            with s3:
                if st.button("💊 Recommendations"):
                    _ask_pdf("What recommendations are given in this report?")

            current_q = st.session_state.get("pdf_question", "")
            question = st.text_input("Or type your question:", value=current_q,
                                     placeholder="e.g., What are my cholesterol levels?")
            if "pdf_question" in st.session_state:
                del st.session_state["pdf_question"]

            if question:
                with st.spinner("🤔 Analysing…"):
                    answer = rag_query(st.session_state[pdf_key], question)
                _ai_card("🤖 AI Health Insight", answer)
        else:
            st.warning(f"⚠️ {text}")
    else:
        st.markdown("""<div class="no-data-box">
            <div style="font-size:2.5rem;margin-bottom:.75rem">📄</div>
            <h3>Upload a PDF Health Report</h3>
            <p>Supported: lab results, medical reports, health summaries</p>
        </div>""", unsafe_allow_html=True)


# -------------------------
# AI HEALTH ASSISTANT
# -------------------------

elif menu == "🤖 AI Health Assistant":
    st.markdown("""<div class="page-header">
        <h1>🤖 AI Health Assistant</h1>
        <p>Your personal AI-powered health advisor — ask anything</p>
    </div>""", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def _send_to_ai(user_query: str):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("🤔 Thinking…"):
            try:
                result = workflow.invoke({"query": user_query, "context": "", "messages": [], "response": ""})
                reply = result.get("response", "").strip()
                if not reply:
                    reply = "⚠️ No response returned. Please check your API key and try again."
            except Exception as e:
                reply = f"⚠️ Error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # ── Quick Action buttons ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚡ Quick Actions</div>', unsafe_allow_html=True)
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("😰 Stress Relief", use_container_width=True):
            _send_to_ai("What are some effective stress relief techniques for students?")
            st.rerun()
    with qa2:
        if st.button("😴 Sleep Better", use_container_width=True):
            _send_to_ai("How can I improve my sleep quality as a student?")
            st.rerun()
    with qa3:
        if st.button("🧠 Mental Wellness", use_container_width=True):
            _send_to_ai("Tips for maintaining good mental health as a student")
            st.rerun()
    with qa4:
        if st.button("🏃 Stay Active", use_container_width=True):
            _send_to_ai("What are easy exercises for busy students?")
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Conversation</div>', unsafe_allow_html=True)

    # ── Chat bubble styles — !important everywhere to beat Streamlit dark theme ─
    st.markdown("""
    <style>
        .sha-user-bubble {
            background: #DBEAFE !important;
            border: 1px solid #BFDBFE !important;
            border-radius: 14px !important;
            border-bottom-right-radius: 4px !important;
            padding: 0.85rem 1.1rem !important;
            max-width: 78%;
        }
        .sha-bot-bubble {
            background: #CCFBF1 !important;
            border: 1px solid #99F6E4 !important;
            border-radius: 14px !important;
            border-bottom-left-radius: 4px !important;
            padding: 0.85rem 1.1rem !important;
            max-width: 78%;
        }
        .sha-user-bubble, .sha-user-bubble *,
        .sha-bot-bubble,  .sha-bot-bubble * {
            color: #0B2545 !important;
            font-size: 0.9rem !important;
            line-height: 1.7 !important;
        }
        .sha-bubble-label {
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            opacity: 0.5 !important;
            margin-bottom: 5px !important;
            display: block !important;
            color: #0B2545 !important;
        }
        .sha-user-bubble p, .sha-bot-bubble p   { margin: 0.3rem 0 !important; }
        .sha-user-bubble li, .sha-bot-bubble li  { margin: 0.15rem 0 !important; }
        .sha-user-bubble ul, .sha-bot-bubble ul,
        .sha-user-bubble ol, .sha-bot-bubble ol  { margin-left: 1.2rem !important; margin-top: 0.3rem !important; }
        .sha-bot-bubble strong, .sha-user-bubble strong { font-weight: 700 !important; }
        .sha-chat-row { display: flex; gap: 0.75rem; align-items: flex-start; margin: 0.5rem 0; }
        .sha-chat-av {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem; flex-shrink: 0; margin-top: 3px;
        }
        .sha-av-user { background: #DBEAFE !important; }
        .sha-av-bot  { background: #CCFBF1 !important; }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:2rem; background:#F4F6F9 !important;
                    border-radius:10px; border:1px solid #DDE3EC;">
            <div style="font-size:2rem; margin-bottom:0.5rem">💬</div>
            <span style="color:#4A5568 !important; font-size:0.9rem">
                Ask a question below or use a Quick Action to get started.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"""
                <div class="sha-chat-row" style="flex-direction:row-reverse">
                    <div class="sha-chat-av sha-av-user">👤</div>
                    <div class="sha-user-bubble">
                        <span class="sha-bubble-label">You</span>
                        <div style="color:#0B2545 !important">{chat["content"]}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                try:
                    body_html = md_lib.markdown(chat["content"], extensions=["nl2br", "sane_lists"])
                except Exception:
                    body_html = "".join(f"<p>{l}</p>" for l in chat["content"].split("\n") if l.strip())
                html = (
                    '<div class="sha-chat-row">'
                    '<div class="sha-chat-av sha-av-bot">🤖</div>'
                    '<div class="sha-bot-bubble">'
                    '<span class="sha-bubble-label">AI Assistant</span>'
                    '<div style="color:#0B2545 !important">' + body_html + '</div>'
                    '</div></div>'
                )
                st.markdown(html, unsafe_allow_html=True)

    # ── Input bar ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    # Use a separate internal buffer key (_val) — never the widget key itself.
    # Writing to a widget's own key after render raises StreamlitAPIException.
    # The on_change callback syncs the widget value into the buffer each keystroke.
    if "_chat_buf" not in st.session_state:
        st.session_state["_chat_buf"] = ""

    def _on_input_change():
        st.session_state["_chat_buf"] = st.session_state["_chat_widget"]

    st.text_area(
        "Your message:",
        key="_chat_widget",
        placeholder="e.g., How can I manage exam stress effectively?",
        height=90,
        label_visibility="collapsed",
        on_change=_on_input_change
    )

    btn_c1, btn_c2, btn_c3 = st.columns([3, 1, 1])
    with btn_c2:
        send_button = st.button("🚀 Send", use_container_width=True)
    with btn_c3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state["_chat_buf"] = ""
            st.rerun()

    # Read from the buffer, not from the widget key
    pending = st.session_state.get("_chat_buf", "").strip()
    if send_button and pending:
        st.session_state["_chat_buf"] = ""          # clear buffer (safe — not the widget key)
        _send_to_ai(pending)
        st.rerun()

    st.markdown("""
    <div style="margin-top:1.25rem; padding:0.8rem 1rem; background:#F4F6F9;
                border-radius:8px; border:1px solid #DDE3EC; font-size:0.8rem; color:#4A5568;">
        <strong>⚠️ Disclaimer:</strong> This AI assistant provides general health information only.
        For medical concerns, please consult a qualified healthcare professional.
    </div>""", unsafe_allow_html=True)


# -------------------------
# FOOTER
# -------------------------

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem; border-top:1px solid #DDE3EC;">
    <p style="color:#4A5568; font-size:0.78rem; margin:0;">
        🏥 <strong>Student Health Analytics</strong> &nbsp;·&nbsp;
        Built with Streamlit, LangGraph &amp; Groq AI &nbsp;·&nbsp;
        <em>For educational purposes only</em>
    </p>
</div>""", unsafe_allow_html=True)
