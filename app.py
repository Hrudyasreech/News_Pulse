import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import hashlib
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NewsLens",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  GLOBAL CSS  –  chocolate-brown luxury editorial
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

/* ── tokens ── */
:root {
    --bg:        #1a1208;
    --surface:   #231a0e;
    --surface2:  #2e2212;
    --border:    #4a3520;
    --accent:    #c8954a;
    --accent2:   #e8c090;
    --text:      #e8ddd0;
    --muted:     #9a8570;
    --neg:       #c04a4a;
    --pos:       #6aab6a;
    --neu:       #8a8a6a;
}

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* inputs */
.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(200,149,74,.2) !important;
}

/* primary button */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #1a1208 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    font-size: .75rem !important;
    padding: .65rem 1.5rem !important;
    transition: background .2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent2) !important;
}
.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 2px !important;
    font-size: .75rem !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    transition: border-color .2s, color .2s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 2px !important;
    font-size: .75rem !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
}

/* metrics */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--accent) !important;
    padding: 1rem !important;
    border-radius: 2px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .7rem !important; letter-spacing: .1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: var(--accent2) !important; font-family: 'Cormorant Garamond', serif !important; font-size: 2rem !important; }

/* dataframe */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 2px !important; }
iframe { background: var(--surface) !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    padding: .6rem 1.2rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* divider */
hr { border-color: var(--border) !important; }

/* info / success / error boxes */
.stAlert { border-radius: 2px !important; border-left: 3px solid var(--accent) !important; background: var(--surface) !important; }

/* radio in sidebar */
.stRadio > div { gap: .3rem !important; }
.stRadio label { font-size: .8rem !important; letter-spacing: .06em !important; }

/* plotly */
.js-plotly-plot .plotly { background: transparent !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HELPERS – branded HTML components
# ============================================================
def brand_header():
    st.markdown("""
    <div style="padding:2rem 0 1.5rem; border-bottom:1px solid var(--border); margin-bottom:2rem;">
        <p style="font-family:'Jost',sans-serif; font-size:.65rem; letter-spacing:.25em;
                  color:var(--muted); text-transform:uppercase; margin:0 0 .3rem;">
            NLP · Intelligence Platform
        </p>
        <h1 style="font-family:'Cormorant Garamond',serif; font-size:3rem; font-weight:300;
                   color:var(--accent2); margin:0; line-height:1;">
            NewsLens
        </h1>
        <p style="font-family:'Jost',sans-serif; font-size:.8rem; color:var(--muted);
                  letter-spacing:.08em; margin:.5rem 0 0;">
            Semantic news analysis · Trends · Sentiment
        </p>
    </div>
    """, unsafe_allow_html=True)

def section_title(text, sub=None):
    st.markdown(f"""
    <div style="margin:2rem 0 1.2rem; padding-bottom:.8rem; border-bottom:1px solid var(--border);">
        <h2 style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:300;
                   color:var(--accent2); margin:0; letter-spacing:.02em;">{text}</h2>
        {"<p style='font-size:.75rem; color:var(--muted); margin:.3rem 0 0; letter-spacing:.06em;'>"+sub+"</p>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def article_card(title, source, keyword, score=None):
    score_color = "var(--pos)" if score and score >= 0.2 else ("var(--neg)" if score and score <= -0.2 else "var(--neu)")
    score_html = f"<span style='color:{score_color}; font-size:.7rem;'>● {score:.3f}</span>" if score is not None else ""
    st.markdown(f"""
    <div style="background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--accent);
                padding:1rem 1.2rem; margin:.5rem 0; border-radius:2px;">
        <p style="font-family:'Cormorant Garamond',serif; font-size:1rem; color:var(--accent2);
                  margin:0 0 .4rem; line-height:1.4;">{title}</p>
        <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">
            <span style="font-size:.65rem; letter-spacing:.1em; color:var(--muted); text-transform:uppercase;">{source}</span>
            <span style="font-size:.65rem; letter-spacing:.1em; color:var(--accent); text-transform:uppercase; 
                         border:1px solid var(--border); padding:.1rem .4rem;">{keyword}</span>
            {score_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def plotly_theme():
    """Return common plotly layout overrides for the brown theme."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Jost, sans-serif', color='#9a8570', size=11),
        title_font=dict(family='Cormorant Garamond, serif', color='#e8c090', size=16),
        colorway=['#c8954a','#e8c090','#a06030','#6aab6a','#c04a4a','#8a8a6a'],
        xaxis=dict(gridcolor='#2e2212', linecolor='#4a3520', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#2e2212', linecolor='#4a3520', tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#4a3520', borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
    )


# ============================================================
#  NLTK SETUP
# ============================================================
@st.cache_resource
def download_nltk_data():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon']:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

download_nltk_data()

@st.cache_resource
def get_preprocessing_resources():
    stop_words = set(stopwords.words('english'))
    extra = {"new","said","say","year","world","could","one","make","day","watch","wa","ha","may","also","would","like","get","us","time"}
    stop_words |= extra
    return stop_words, WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = get_preprocessing_resources()


# ============================================================
#  AUTH  (simple hash-based; no DB required)
# ============================================================
USERS = {
    "admin": {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "user":  {"password": hashlib.sha256("user123".encode()).hexdigest(),  "role": "user"},
}

def check_login(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    user = USERS.get(username)
    if user and user["password"] == hashed:
        return user["role"]
    return None


# ============================================================
#  CONFIG
# ============================================================
try:
    API_KEY = st.secrets.get("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY_HERE")
except:
    API_KEY = "YOUR_NEWSAPI_KEY_HERE"


# ============================================================
#  TEXT PROCESSING
# ============================================================
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_nlp(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text):
    try:
        tokens = word_tokenize(text)
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
        tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
        return ' '.join(tokens)
    except:
        return text


# ============================================================
#  API
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_news(keywords_tuple, from_date, to_date):
    articles_list = []
    for keyword in keywords_tuple:
        for page in range(1, 3):
            try:
                url = (f"https://newsapi.org/v2/everything?q={keyword}"
                       f"&from={from_date}&to={to_date}&pageSize=100&page={page}&apiKey={API_KEY}")
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    arts = data.get("articles", [])
                    if not arts:
                        break
                    for a in arts:
                        articles_list.append({
                            "Title":          a.get("title", ""),
                            "Description":    a.get("description", ""),
                            "Source":         a.get("source", {}).get("name", ""),
                            "Published Date": a.get("publishedAt", ""),
                            "Keyword":        keyword,
                            "URL":            a.get("url", ""),
                        })
                    time.sleep(0.2)
                else:
                    break
            except:
                break
    return pd.DataFrame(articles_list)

def clean_dataset(df):
    df["Title"]       = df["Title"].apply(clean_text)
    df["Description"] = df["Description"].apply(clean_text)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Title"], inplace=True)
    df["Published Date"] = pd.to_datetime(df["Published Date"], utc=True)
    df["Date"]           = df["Published Date"].dt.date
    df['news']           = df['Title'] + ' ' + df['Description']
    df['clean_news']     = df['news'].apply(clean_text_for_nlp)
    df['preprocessed_news'] = df['clean_news'].apply(preprocess_text)
    df = df[df['Title'].str.strip() != '']
    df = df[df['Title'] != '[Removed]']
    return df.reset_index(drop=True)


# ============================================================
#  ANALYSIS
# ============================================================
def perform_tfidf(df):
    vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    mat = vec.fit_transform(df['clean_news'])
    names = vec.get_feature_names_out()
    scores = np.asarray(mat.mean(axis=0)).ravel()
    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return {w: float(s) for w, s in pairs[:15]}

def perform_lda(df, n_topics=3):
    vec = CountVectorizer(max_features=500, max_df=0.6, min_df=3, ngram_range=(1, 2), stop_words='english')
    X = vec.fit_transform(df['preprocessed_news'])
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, random_state=42)
    lda.fit(X)
    words = vec.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(lda.components_):
        top = [words[j] for j in comp.argsort()[-8:][::-1]]
        topics[f"Topic {i+1}"] = top
    return topics

def perform_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['clean_news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment']       = df['sentiment_score'].apply(
        lambda s: 'Positive' if s >= 0.2 else ('Negative' if s <= -0.2 else 'Neutral'))
    return df

def keyword_trend(df):
    """Daily keyword mentions"""
    trend = df.groupby(['Date', 'Keyword']).size().reset_index(name='Count')
    return trend

def sentiment_trend(df):
    """Daily average sentiment"""
    trend = df.groupby('Date')['sentiment_score'].mean().reset_index()
    trend.columns = ['Date', 'Avg Sentiment']
    return trend

def balanced_sample(df, n=2):
    """n articles per keyword"""
    return df.groupby('Keyword').head(n)


# ============================================================
#  SESSION STATE INIT
# ============================================================
for key in ['df','tfidf','lda','from_date','to_date','logged_in','role','username']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


# ============================================================
#  LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("""
    <div style="max-width:420px; margin:6rem auto 0; padding:3rem; 
                background:var(--surface); border:1px solid var(--border); border-radius:2px;">
        <p style="font-size:.6rem; letter-spacing:.3em; color:var(--muted);
                  text-transform:uppercase; margin:0 0 .3rem;">NLP Intelligence</p>
        <h1 style="font-family:'Cormorant Garamond',serif; font-size:2.8rem; font-weight:300;
                   color:var(--accent2); margin:0 0 .3rem;">NewsLens</h1>
        <p style="font-size:.75rem; color:var(--muted); margin:0 0 2rem; letter-spacing:.05em;">
            Sign in to access the dashboard
        </p>
        <hr style="border-color:var(--border); margin-bottom:1.5rem;">
    </div>
    """, unsafe_allow_html=True)

    # Center the form
    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.container():
            username = st.text_input("Username", placeholder="admin or user")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            if st.button("Sign In →", use_container_width=True, type="primary"):
                role = check_login(username, password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.role      = role
                    st.session_state.username  = username
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            
            st.markdown("""
            <p style="font-size:.65rem; color:var(--muted); text-align:center; margin-top:1rem; letter-spacing:.05em;">
                Demo — admin / admin123 &nbsp;·&nbsp; user / user123
            </p>
            """, unsafe_allow_html=True)


# ============================================================
#  ADMIN PAGE
# ============================================================
def show_admin(df):
    section_title("Admin Console", "System overview · Data management")

    tab1, tab2 = st.tabs(["📊  System Stats", "🗄️  Data Management"])

    with tab1:
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Articles", len(df))
            col2.metric("Unique Sources", df['Source'].nunique())
            col3.metric("Keywords", df['Keyword'].nunique())
            col4.metric("Date Span", f"{df['Date'].min()} → {df['Date'].max()}")

            st.markdown("<br>", unsafe_allow_html=True)
            section_title("Keyword Distribution")
            kw = df['Keyword'].value_counts().reset_index()
            kw.columns = ['Keyword','Count']
            fig = px.bar(kw, x='Keyword', y='Count', title="Articles per Keyword")
            fig.update_layout(**plotly_theme())
            fig.update_traces(marker_color='#c8954a')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data loaded yet. Fetch data from the sidebar.")

    with tab2:
        if df is not None:
            section_title("Duplicate Check")
            dupes = df.duplicated(subset=['Title']).sum()
            col1, col2 = st.columns(2)
            col1.metric("Duplicate Titles", int(dupes))
            col2.metric("Clean Articles", len(df) - int(dupes))

            st.markdown("<br>", unsafe_allow_html=True)
            section_title("Raw Data Preview")
            st.dataframe(df[['Title','Source','Keyword','Date']].head(20), use_container_width=True)

            if st.button("🗑  Clear Dataset", type="primary"):
                st.session_state.df = None
                st.session_state.tfidf = None
                st.session_state.lda = None
                st.success("Dataset cleared.")
                st.rerun()
        else:
            st.info("No dataset in memory.")


# ============================================================
#  MAIN APP
# ============================================================
def show_app():
    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1.2rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:1rem;">
            <p style="font-size:.6rem; letter-spacing:.2em; color:var(--muted); text-transform:uppercase; margin:0;">
                Signed in as
            </p>
            <p style="font-size:.9rem; color:var(--accent2); font-family:'Cormorant Garamond',serif; margin:.2rem 0 0;">
                {st.session_state.username} 
                <span style="font-size:.65rem; color:var(--muted);">({st.session_state.role})</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Search Keywords**")
        kw_input = st.text_input(
            "Enter up to 5 keywords (comma-separated)",
            value="AI, climate, economy, healthcare, elections",
            label_visibility="collapsed"
        )
        keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()][:5]
        
        if keywords:
            pills = " ".join([f'<span style="border:1px solid var(--border);padding:.1rem .5rem;'
                              f'font-size:.6rem;letter-spacing:.08em;color:var(--accent);'
                              f'text-transform:uppercase;">{k}</span>' for k in keywords])
            st.markdown(f'<div style="margin:.5rem 0 1rem;display:flex;flex-wrap:wrap;gap:.3rem;">{pills}</div>',
                        unsafe_allow_html=True)

        st.markdown("**Date Range**")
        today         = datetime.utcnow().date()
        default_start = today - timedelta(days=14)
        col1, col2    = st.columns(2)
        from_date = col1.date_input("From", value=default_start, max_value=today, label_visibility="collapsed")
        to_date   = col2.date_input("To",   value=today,         max_value=today, label_visibility="collapsed")

        if from_date > to_date:
            st.error("From date is after To date.")
        else:
            days = (to_date - from_date).days
            st.caption(f"Range: {days} days")

        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("Fetch & Analyse →", use_container_width=True, type="primary",
                              disabled=(from_date > to_date))

        if fetch_btn:
            if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
                st.error("Set your NewsAPI key in st.secrets['NEWSAPI_KEY']")
            else:
                prog = st.progress(0)
                with st.spinner("Fetching articles…"):
                    raw = fetch_news(tuple(keywords), from_date.isoformat(), to_date.isoformat())
                prog.progress(30)
                if raw.empty:
                    st.error("No articles found.")
                else:
                    with st.spinner("Processing…"):
                        df = clean_dataset(raw)
                    prog.progress(60)
                    with st.spinner("Analysing…"):
                        tfidf = perform_tfidf(df)
                        lda   = perform_lda(df)
                        df    = perform_sentiment(df)
                    prog.progress(100)
                    st.session_state.df        = df
                    st.session_state.tfidf     = tfidf
                    st.session_state.lda       = lda
                    st.session_state.from_date = from_date
                    st.session_state.to_date   = to_date
                    st.success(f"✓ {len(df)} articles analysed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        # Navigation
        if st.session_state.df is not None:
            nav_options = ["Summary", "Trends", "Topics", "Sentiment", "Export"]
            if st.session_state.role == "admin":
                nav_options.append("Admin")

            section = st.radio("Navigate", nav_options, label_visibility="collapsed")
        else:
            st.caption("Fetch data to unlock views.")
            section = None

        st.divider()
        if st.button("Sign Out", use_container_width=False):
            for k in ['logged_in','role','username','df','tfidf','lda']:
                st.session_state[k] = None
            st.session_state.logged_in = False
            st.rerun()

    # ── MAIN CONTENT ─────────────────────────────────────────
    brand_header()

    if st.session_state.df is None:
        st.markdown("""
        <div style="text-align:center; padding:5rem 0;">
            <p style="font-family:'Cormorant Garamond',serif; font-size:2rem; color:var(--muted); font-weight:300;">
                Enter keywords and fetch articles to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df    = st.session_state.df
    tfidf = st.session_state.tfidf
    lda   = st.session_state.lda

    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════
    if section == "Summary":
        section_title("Summary", "Overview of fetched articles")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles", len(df))
        c2.metric("Sources", df['Source'].nunique())
        c3.metric("Keywords", df['Keyword'].nunique())
        pos_pct = round(len(df[df['sentiment']=='Positive'])/len(df)*100, 1)
        c4.metric("Positive Sentiment", f"{pos_pct}%")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section_title("Articles by Keyword")
            kw_cnt = df['Keyword'].value_counts().reset_index()
            kw_cnt.columns = ['Keyword','Count']
            fig = px.bar(kw_cnt, x='Keyword', y='Count')
            fig.update_layout(**plotly_theme(), showlegend=False)
            fig.update_traces(marker_color='#c8954a', marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Top 10 Sources")
            src = df['Source'].value_counts().head(10).reset_index()
            src.columns = ['Source','Count']
            fig2 = px.bar(src, x='Count', y='Source', orientation='h')
            fig2.update_layout(**plotly_theme(), showlegend=False,
                               yaxis=dict(autorange='reversed', gridcolor='#2e2212'))
            fig2.update_traces(marker_color='#a06030', marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Article Sample", f"2 articles per keyword")
        sample = balanced_sample(df, n=2)
        for _, row in sample.iterrows():
            article_card(row['Title'], row['Source'], row['Keyword'], row.get('sentiment_score'))

    # ══════════════════════════════════════════════════════════
    #  TRENDS
    # ══════════════════════════════════════════════════════════
    elif section == "Trends":
        section_title("Trend Analysis", "Keyword frequency and sentiment over time")

        col1, col2 = st.columns(2)

        with col1:
            section_title("Keyword Frequency Over Time")
            trend = keyword_trend(df)
            fig = px.line(trend, x='Date', y='Count', color='Keyword',
                          markers=True, title="Daily Keyword Mentions")
            fig.update_layout(**plotly_theme())
            fig.update_traces(line_width=2, marker_size=5)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Sentiment Trend Over Time")
            s_trend = sentiment_trend(df)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=s_trend['Date'], y=s_trend['Avg Sentiment'],
                mode='lines+markers',
                line=dict(color='#c8954a', width=2),
                marker=dict(size=5, color='#e8c090'),
                fill='tozeroy', fillcolor='rgba(200,149,74,.08)',
                name='Avg Sentiment'
            ))
            fig2.add_hline(y=0.2,  line_dash='dot', line_color='#6aab6a', annotation_text="Positive", annotation_font_size=9)
            fig2.add_hline(y=-0.2, line_dash='dot', line_color='#c04a4a', annotation_text="Negative", annotation_font_size=9)
            fig2.update_layout(title="Daily Average Sentiment Score", **plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Top TF-IDF Keywords", "Extracted from full corpus")
        tfidf_df = pd.DataFrame(list(tfidf.items()), columns=['Keyword','Score']).sort_values('Score', ascending=False).head(10)
        fig3 = px.bar(tfidf_df.sort_values('Score'), x='Score', y='Keyword', orientation='h')
        fig3.update_layout(**plotly_theme(), showlegend=False,
                           yaxis=dict(autorange='reversed', gridcolor='#2e2212'))
        fig3.update_traces(marker_color='#c8954a', marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  TOPICS
    # ══════════════════════════════════════════════════════════
    elif section == "Topics":
        section_title("Topic Modelling", "LDA-discovered latent topics")

        for topic_name, words in lda.items():
            with st.container():
                st.markdown(f"""
                <div style="background:var(--surface); border:1px solid var(--border);
                            border-left:3px solid var(--accent); padding:1.2rem 1.5rem; margin:.6rem 0; border-radius:2px;">
                    <p style="font-size:.6rem; letter-spacing:.2em; color:var(--muted); 
                              text-transform:uppercase; margin:0 0 .5rem;">{topic_name}</p>
                    <div style="display:flex; flex-wrap:wrap; gap:.5rem;">
                        {"".join([f'<span style="background:var(--surface2);border:1px solid var(--border);padding:.2rem .7rem;font-size:.75rem;letter-spacing:.05em;color:var(--accent2);">{w}</span>' for w in words])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Topics", len(lda))
        c2.metric("Words per Topic", 8)
        c3.metric("Total Vocab Terms", sum(len(v) for v in lda.values()))

    # ══════════════════════════════════════════════════════════
    #  SENTIMENT
    # ══════════════════════════════════════════════════════════
    elif section == "Sentiment":
        section_title("Sentiment Analysis", "VADER — compound score · Positive ≥0.2 · Negative ≤–0.2")

        dist = df['sentiment'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("😊 Positive", int(dist.get('Positive',0)),
                  f"{dist.get('Positive',0)/len(df)*100:.1f}%")
        c2.metric("😐 Neutral",  int(dist.get('Neutral',0)),
                  f"{dist.get('Neutral',0)/len(df)*100:.1f}%")
        c3.metric("😞 Negative", int(dist.get('Negative',0)),
                  f"{dist.get('Negative',0)/len(df)*100:.1f}%")
        c4.metric("Mean Score", f"{df['sentiment_score'].mean():.3f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            section_title("Distribution")
            fig = px.pie(
                values=dist.values, names=dist.index,
                color=dist.index,
                color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'},
                hole=0.55
            )
            fig.update_layout(**plotly_theme(), showlegend=True)
            fig.update_traces(textinfo='label+percent', textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("By Keyword")
            sk = pd.crosstab(df['Keyword'], df['sentiment'])
            fig2 = px.bar(sk.reset_index(), x='Keyword',
                          y=[c for c in ['Positive','Neutral','Negative'] if c in sk.columns],
                          barmode='group',
                          color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'})
            fig2.update_layout(**plotly_theme())
            st.plotly_chart(fig2, use_container_width=True)

        section_title("Score Distribution")
        fig3 = px.histogram(df, x='sentiment_score', nbins=40,
                            color_discrete_sequence=['#c8954a'])
        fig3.add_vline(x=0.2,  line_dash='dot', line_color='#6aab6a')
        fig3.add_vline(x=-0.2, line_dash='dot', line_color='#c04a4a')
        fig3.update_layout(**plotly_theme(), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Most Positive & Negative Articles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p style='font-size:.7rem;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin:0 0 .8rem;'>Most Positive</p>", unsafe_allow_html=True)
            for _, row in df.nlargest(5,'sentiment_score').iterrows():
                article_card(row['Title'], row['Source'], row['Keyword'], row['sentiment_score'])
        with col2:
            st.markdown("<p style='font-size:.7rem;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin:0 0 .8rem;'>Most Negative</p>", unsafe_allow_html=True)
            for _, row in df.nsmallest(5,'sentiment_score').iterrows():
                article_card(row['Title'], row['Source'], row['Keyword'], row['sentiment_score'])

    # ══════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════
    elif section == "Export":
        section_title("Export", "Download your analysis results")
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        exports = {
            "Full Dataset": (df, f"full_{ts}.csv"),
            "Sentiment Results": (df[['Title','Source','Keyword','Date','sentiment_score','sentiment']], f"sentiment_{ts}.csv"),
            "Cleaned Text": (df[['Title','clean_news','preprocessed_news','Keyword']], f"cleaned_{ts}.csv"),
            "Keywords (TF-IDF)": (pd.DataFrame(list(tfidf.items()), columns=['Keyword','Score']).sort_values('Score',ascending=False), f"keywords_{ts}.csv"),
            "LDA Topics": (pd.DataFrame([{'Topic':t,'Words':', '.join(w)} for t,w in lda.items()]), f"topics_{ts}.csv"),
        }

        for label, (data, fname) in exports.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);
                            padding:.8rem 1rem; border-radius:2px;">
                    <p style="margin:0; font-size:.75rem; letter-spacing:.08em; color:var(--accent2);">{label}</p>
                    <p style="margin:.2rem 0 0; font-size:.65rem; color:var(--muted);">{len(data)} rows · {len(data.columns)} cols</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.download_button(f"↓ Download", data=data.to_csv(index=False),
                                   file_name=fname, mime="text/csv", use_container_width=True)
            st.markdown("<div style='margin:.4rem 0;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  ADMIN
    # ══════════════════════════════════════════════════════════
    elif section == "Admin" and st.session_state.role == "admin":
        show_admin(df)


# ============================================================
#  ENTRY POINT
# ============================================================
if not st.session_state.logged_in:
    show_login()
else:
    show_app()
