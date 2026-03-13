import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import hashlib
import json
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
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  ENHANCED CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

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

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(200,149,74,.2) !important;
}

.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #1a1208 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    font-size: .75rem !important;
    padding: .75rem 1.5rem !important;
    transition: all .2s ease !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(200,149,74,.3) !important;
}

.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 6px !important;
    font-size: .75rem !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    transition: all .2s !important;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--accent) !important;
    padding: 1.5rem !important;
    border-radius: 6px !important;
}

[data-testid="stMetricLabel"] { 
    color: var(--muted) !important; 
    font-size: .7rem !important; 
    letter-spacing: .1em !important; 
    text-transform: uppercase !important; 
}

[data-testid="stMetricValue"] { 
    color: var(--accent2) !important; 
    font-family: 'Cormorant Garamond', serif !important; 
    font-size: 2.8rem !important; 
    font-weight: 600 !important;
}

.stDataFrame { 
    border: 1px solid var(--border) !important; 
    border-radius: 6px !important; 
}

iframe { background: var(--surface) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: .75rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    padding: .8rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

hr { border-color: var(--border) !important; }

.stAlert { 
    border-radius: 6px !important; 
    border-left: 3px solid var(--accent) !important; 
    background: var(--surface) !important; 
}

.stRadio > div { gap: .4rem !important; }
.stRadio label { font-size: .8rem !important; letter-spacing: .06em !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

h1 { font-size: 3.5rem !important; font-weight: 300 !important; }
h2 { font-size: 2.4rem !important; font-weight: 400 !important; margin: 2rem 0 1rem !important; }
h3 { font-size: 1.6rem !important; font-weight: 500 !important; margin: 1.5rem 0 .8rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  NLTK & SETUP
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
#  AUTHENTICATION
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
#  LLM TOPIC NAMING
# ============================================================
@st.cache_data
def generate_topic_name(keywords_str):
    """Generate topic name from keywords"""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": st.secrets.get("ANTHROPIC_API_KEY", "")
            },
            json={
                "model": "claude-opus-4-20250805",
                "max_tokens": 50,
                "messages": [{
                    "role": "user",
                    "content": f"Create a short, 2-3 word topic name for these keywords: {keywords_str}. Just the name, nothing else."
                }]
            },
            timeout=5
        )
        if response.status_code == 200:
            content = response.json()['content'][0]['text'].strip()
            return content[:30]
    except:
        pass
    return ' & '.join([w.title() for w in keywords_str.split(', ')[:2]])

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
#  NEWS API
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
#  NLP ANALYSIS
# ============================================================
def perform_tfidf(df):
    vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    mat = vec.fit_transform(df['clean_news'])
    names = vec.get_feature_names_out()
    scores = np.asarray(mat.mean(axis=0)).ravel()
    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return {w: float(s) for w, s in pairs[:20]}

def perform_lda(df, n_topics=5):
    vec = CountVectorizer(max_features=500, max_df=0.6, min_df=3, ngram_range=(1, 2), stop_words='english')
    X = vec.fit_transform(df['preprocessed_news'])
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, random_state=42)
    lda.fit(X)
    words = vec.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(lda.components_):
        top = [words[j] for j in comp.argsort()[-8:][::-1]]
        keywords_str = ', '.join(top[:3])
        topic_name = generate_topic_name(keywords_str)
        topics[topic_name] = top
    return topics

def perform_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['clean_news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment']       = df['sentiment_score'].apply(
        lambda s: 'Positive' if s >= 0.2 else ('Negative' if s <= -0.2 else 'Neutral'))
    return df

def keyword_trend(df):
    trend = df.groupby(['Date', 'Keyword']).size().reset_index(name='Count')
    return trend

def sentiment_trend(df):
    trend = df.groupby('Date')['sentiment_score'].mean().reset_index()
    trend.columns = ['Date', 'Avg Sentiment']
    return trend

def balanced_sample(df, n=3):
    return df.groupby('Keyword').head(n)

# ============================================================
#  PLOTLY THEME
# ============================================================
def plotly_theme(yaxis_override=None):
    theme = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Jost, sans-serif', color='#9a8570', size=12),
        title_font=dict(family='Cormorant Garamond, serif', color='#e8c090', size=18),
        colorway=['#c8954a','#e8c090','#a06030','#6aab6a','#c04a4a','#8a8a6a'],
        xaxis=dict(gridcolor='#2e2212', linecolor='#4a3520', tickfont=dict(size=10)),
        yaxis=dict(gridcolor='#2e2212', linecolor='#4a3520', tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#4a3520', borderwidth=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if yaxis_override:
        theme['yaxis'].update(yaxis_override)
    return theme

# ============================================================
#  COMPONENTS
# ============================================================
def section_title(text, sub=None):
    st.markdown(f"""
    <div style="margin:2.5rem 0 1.5rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
        <h2 style="font-family:'Cormorant Garamond',serif; font-size:2.4rem; font-weight:300;
                   color:var(--accent2); margin:0; letter-spacing:.02em;">{text}</h2>
        {"<p style='font-size:.85rem; color:var(--muted); margin:.5rem 0 0; letter-spacing:.06em;'>"+sub+"</p>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def article_card(title, source, keyword, score=None, idx=None, url=None):
    """Article card with STAR BOOKMARK"""
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = {}
    
    score_color = "var(--pos)" if score and score >= 0.2 else ("var(--neg)" if score and score <= -0.2 else "var(--neu)")
    sentiment_label = "Positive" if score and score >= 0.2 else ("Negative" if score and score <= -0.2 else "Neutral")
    
    article_id = f"{idx}_{title[:20]}_{source}"
    is_bookmarked = article_id in st.session_state.bookmarks
    
    # Bookmark button with star icon
    col1, col2 = st.columns([0.92, 0.08])
    
    with col1:
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-left:4px solid {score_color};
                    padding:1.5rem; margin:.8rem 0; border-radius:6px; min-height:200px; display:flex; flex-direction:column;">
            <div style="flex:1;">
                <h4 style="font-family:'Cormorant Garamond',serif; font-size:1.1rem; color:var(--accent2);
                          margin:0 0 .8rem; line-height:1.5;">{title}</h4>
                <p style="font-size:.75rem; color:var(--muted); margin:0; line-height:1.6;">
                    {source} • {keyword.upper()}
                </p>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem; padding-top:1rem; border-top:1px solid var(--border);">
                <span style="font-size:.7rem; color:{score_color}; text-transform:uppercase; letter-spacing:.06em; font-weight:500;">
                    ● {sentiment_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        bookmark_symbol = "★" if is_bookmarked else "☆"
        bookmark_color = "#c8954a" if is_bookmarked else "#9a8570"
        
        if st.button(bookmark_symbol, key=f"bookmark_{article_id}", 
                    help="Click to bookmark"):
            if is_bookmarked:
                del st.session_state.bookmarks[article_id]
            else:
                st.session_state.bookmarks[article_id] = {
                    'Title': title,
                    'Source': source,
                    'Keyword': keyword,
                    'Sentiment': sentiment_label,
                    'Score': score,
                    'URL': url,
                    'Saved_At': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
            st.rerun()

# ============================================================
#  SESSION STATE
# ============================================================
for key in ['df','tfidf','lda','from_date','to_date','logged_in','role','username']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = {}

# ============================================================
#  LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("""
    <div style="min-height:100vh; display:flex; align-items:center; justify-content:center; padding:2rem;">
        <div style="max-width:600px; width:100%;">
            <div style="text-align:center; margin-bottom:3rem;">
                <div style="font-size:4rem; margin-bottom:1.5rem;">📰</div>
                <h1 style="font-family:'Cormorant Garamond',serif; font-size:3.5rem; font-weight:300;
                           color:var(--accent2); margin:0 0 .5rem; letter-spacing:.02em;">NewsLens</h1>
                <p style="font-size:.95rem; color:var(--muted); margin:0; letter-spacing:.05em;">
                    NLP-powered news analysis platform
                </p>
            </div>

            <div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:3rem;">
                <p style="font-size:1.1rem; color:var(--text); margin:0 0 .8rem; font-weight:500;">Welcome back</p>
                <p style="font-size:.9rem; color:var(--muted); margin:0 0 2rem;">Sign in to your dashboard</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown('<div style="background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:2.5rem;">', unsafe_allow_html=True)
        
        username = st.text_input("", placeholder="you@example.com", label_visibility="collapsed")
        password = st.text_input("", placeholder="Enter your password", type="password", label_visibility="collapsed")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.checkbox("Remember me")
        with col2:
            st.markdown('<p style="text-align:right; font-size:.75rem; color:var(--accent); cursor:pointer;">Forgot password?</p>', unsafe_allow_html=True)
        
        if st.button("Sign in", use_container_width=True, type="primary"):
            role = check_login(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin/admin123 or user/user123")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
#  ADMIN DASHBOARD
# ============================================================
def show_admin(df):
    section_title("Analytics Dashboard", "System overview & data management")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Articles", len(df), "+12%")
    col2.metric("Unique Sources", df['Source'].nunique())
    col3.metric("Keywords", df['Keyword'].nunique())
    col4.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🗄️ Management", "📈 Insights"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            section_title("Keyword Distribution")
            kw = df['Keyword'].value_counts().reset_index()
            kw.columns = ['Keyword','Count']
            fig = px.bar(kw, x='Keyword', y='Count')
            fig.update_layout(**plotly_theme(), height=400)
            fig.update_traces(marker_color='#c8954a')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_title("Source Distribution")
            src = df['Source'].value_counts().head(10).reset_index()
            src.columns = ['Source','Count']
            fig2 = px.bar(src, x='Count', y='Source', orientation='h')
            theme = plotly_theme(yaxis_override={'autorange': 'reversed'})
            fig2.update_layout(**theme, height=400)
            fig2.update_traces(marker_color='#6aab6a')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_title("Data Quality")
        col1, col2, col3 = st.columns(3)
        dupes = df.duplicated(subset=['Title']).sum()
        col1.metric("Total Records", len(df))
        col2.metric("Duplicates", int(dupes))
        col3.metric("Clean Data", len(df) - int(dupes))

        st.dataframe(df[['Title','Source','Keyword','Date']].head(20), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        col1.metric("Bookmarks", len(st.session_state.bookmarks), "📌")
        col2.metric("Data Freshness", "Real-time", "🔄")

# ============================================================
#  MAIN APP
# ============================================================
def show_app():
    # TOP HEADER
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown('<div style="font-size:2.2rem;">📰</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p style="font-size:.6rem; letter-spacing:.2em; color:var(--muted); text-transform:uppercase; margin:0;">NLP Analytics</p><h1 style="font-family:\'Cormorant Garamond\',serif; font-size:2.2rem; color:var(--accent2); margin:.3rem 0 0; font-weight:400;">NewsLens</h1>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<p style="text-align:right; font-size:.65rem; color:var(--muted);">👤 {st.session_state.username}</p>', unsafe_allow_html=True)
    
    st.divider()

    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="text-align:center; margin-bottom:2rem;"><div style="font-size:2.5rem;">📰</div><h2 style="font-family:\'Cormorant Garamond\',serif; font-size:1.8rem; color:var(--accent2); margin:0;">NewsLens</h2></div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### Navigation")
        
        if st.session_state.df is not None:
            nav_options = ["Summary", "Trends", "Topics", "Sentiment", "Reading List"]
            if st.session_state.role == "admin":
                nav_options.append("Admin")
            section = st.radio("", nav_options, label_visibility="collapsed")
        else:
            section = None

        st.divider()
        st.markdown("### Search Keywords")
        kw_input = st.text_input("Enter keywords", value="AI, climate, economy", label_visibility="collapsed")
        keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()][:5]
        
        st.markdown("### Date Range")
        today = datetime.utcnow().date()
        default_start = today - timedelta(days=14)
        col1, col2 = st.columns(2)
        from_date = col1.date_input("From", value=default_start, max_value=today, label_visibility="collapsed")
        to_date = col2.date_input("To", value=today, max_value=today, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🔍 Fetch & Analyze", use_container_width=True, type="primary", disabled=(from_date > to_date))

        if fetch_btn:
            if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
                st.error("⚠️ Add NewsAPI key")
            else:
                with st.spinner("Fetching…"):
                    raw = fetch_news(tuple(keywords), from_date.isoformat(), to_date.isoformat())
                if raw.empty:
                    st.error("No articles found")
                else:
                    with st.spinner("Processing…"):
                        df = clean_dataset(raw)
                    with st.spinner("Analyzing…"):
                        tfidf = perform_tfidf(df)
                        lda = perform_lda(df)
                        df = perform_sentiment(df)
                    st.session_state.df = df
                    st.session_state.tfidf = tfidf
                    st.session_state.lda = lda
                    st.session_state.from_date = from_date
                    st.session_state.to_date = to_date
                    st.success(f"✅ {len(df)} articles analyzed!")
                    st.rerun()

        st.divider()
        if st.button("Logout", use_container_width=True):
            for k in ['logged_in','role','username','df','tfidf','lda']:
                st.session_state[k] = None
            st.session_state.logged_in = False
            st.rerun()

    # MAIN CONTENT
    if st.session_state.df is None:
        st.markdown('<div style="text-align:center; padding:10rem 2rem;"><h1 style="color:var(--muted);">📰 Start Your Analysis</h1></div>', unsafe_allow_html=True)
        return

    df = st.session_state.df
    tfidf = st.session_state.tfidf
    lda = st.session_state.lda

    # SUMMARY
    if section == "Summary":
        section_title("Summary", "Overview of your news analysis")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", len(df))
        c2.metric("Unique Sources", df['Source'].nunique())
        c3.metric("Keywords", df['Keyword'].nunique())
        pos_pct = round(len(df[df['sentiment']=='Positive'])/len(df)*100, 1)
        c4.metric("Positive %", f"{pos_pct}%")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section_title("Articles by Keyword")
            kw_cnt = df['Keyword'].value_counts().reset_index()
            kw_cnt.columns = ['Keyword','Count']
            fig = px.bar(kw_cnt, x='Keyword', y='Count')
            fig.update_layout(**plotly_theme(), height=420)
            fig.update_traces(marker_color='#c8954a')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Top News Sources")
            src = df['Source'].value_counts().head(10).reset_index()
            src.columns = ['Source','Count']
            fig2 = px.bar(src, x='Count', y='Source', orientation='h')
            theme = plotly_theme(yaxis_override={'autorange': 'reversed'})
            fig2.update_layout(**theme, height=420)
            fig2.update_traces(marker_color='#6aab6a')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Recent Articles")
        
        sample = balanced_sample(df, n=3)
        cols = st.columns(3)
        for idx, (i, row) in enumerate(sample.iterrows()):
            with cols[idx % 3]:
                article_card(row['Title'], row['Source'], row['Keyword'], row.get('sentiment_score'), idx=i, url=row.get('URL'))

    # TRENDS
    elif section == "Trends":
        section_title("Trend Analysis", "Keyword frequency and sentiment")

        st.markdown("#### Filter by Date")
        col1, col2 = st.columns(2)
        with col1:
            trend_start = st.date_input("Trend Start", value=st.session_state.from_date)
        with col2:
            trend_end = st.date_input("Trend End", value=st.session_state.to_date)

        trend_data = keyword_trend(df)
        trend_data = trend_data[(trend_data['Date'] >= trend_start) & (trend_data['Date'] <= trend_end)]

        col1, col2 = st.columns(2)
        with col1:
            section_title("Keyword Frequency")
            fig = px.line(trend_data, x='Date', y='Count', color='Keyword', markers=True)
            fig.update_layout(**plotly_theme(), height=450)
            fig.update_traces(line_width=2.5, marker_size=7)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Sentiment Over Time")
            s_trend = sentiment_trend(df)
            s_trend = s_trend[(s_trend['Date'] >= trend_start) & (s_trend['Date'] <= trend_end)]
            fig2 = px.line(s_trend, x='Date', y='Avg Sentiment', markers=True)
            fig2.update_layout(**plotly_theme(), height=450)
            fig2.update_traces(line_color='#c8954a', marker_size=7)
            st.plotly_chart(fig2, use_container_width=True)

    # TOPICS
    elif section == "Topics":
        section_title("Discovered Topics", "AI-named semantic topics")

        cols = st.columns(2)
        for idx, (topic_name, words) in enumerate(lda.items()):
            with cols[idx % 2]:
                st.markdown(f'<div style="background:var(--surface); border:1px solid var(--border); border-left:4px solid var(--accent); padding:2rem; margin:1rem 0; border-radius:6px;"><h3 style="color:var(--accent2); margin:0 0 1.5rem;">{topic_name}</h3><div style="display:flex; flex-wrap:wrap; gap:.8rem;">{"".join([f"<span style=\'background:var(--surface2);border:1px solid var(--border);padding:.4rem 1rem;border-radius:4px;color:var(--accent2);\'>{w}</span>" for w in words])}</div></div>', unsafe_allow_html=True)

    # SENTIMENT
    elif section == "Sentiment":
        section_title("Sentiment Analysis", "Positive/Negative distribution")

        dist = df['sentiment'].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("😊 Positive", int(dist.get('Positive',0)))
        c2.metric("😐 Neutral", int(dist.get('Neutral',0)))
        c3.metric("😞 Negative", int(dist.get('Negative',0)))

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            section_title("Distribution")
            fig = px.pie(values=dist.values, names=dist.index, 
                        color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'})
            fig.update_layout(**plotly_theme(), height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("By Keyword")
            sk = pd.crosstab(df['Keyword'], df['sentiment'])
            fig2 = px.bar(sk.reset_index(), x='Keyword', y=[c for c in ['Positive','Neutral','Negative'] if c in sk.columns], barmode='group',
                         color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'})
            fig2.update_layout(**plotly_theme(), height=450)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Most Positive & Negative")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='color:var(--pos);'>📈 Most Positive</h3>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(df.nlargest(3,'sentiment_score').iterrows()):
                article_card(row['Title'], row['Source'], row['Keyword'], row['sentiment_score'], idx=i, url=row.get('URL'))
        with col2:
            st.markdown("<h3 style='color:var(--neg);'>📉 Most Negative</h3>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(df.nsmallest(3,'sentiment_score').iterrows()):
                article_card(row['Title'], row['Source'], row['Keyword'], row['sentiment_score'], idx=100+i, url=row.get('URL'))

    # READING LIST (BOOKMARKS)
    elif section == "Reading List":
        section_title("Your Bookmarks", "Saved articles")
        
        if st.session_state.bookmarks:
            st.markdown(f'<div style="padding:2rem; background:var(--surface); border:1px solid var(--border); border-radius:6px; margin-bottom:2rem;"><p style="margin:0; color:var(--muted);">📌 Total Bookmarked</p><p style="font-size:2.8rem; color:var(--accent2); margin:.8rem 0 0; font-weight:600;">{len(st.session_state.bookmarks)}</p></div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            for idx, (article_id, article) in enumerate(st.session_state.bookmarks.items()):
                with cols[idx % 3]:
                    article_card(article['Title'], article['Source'], article['Keyword'], article.get('Score'), idx=idx, url=article.get('URL'))
        else:
            st.info("📝 No bookmarks yet. Click ★ on article cards to save them!")

    # ADMIN
    elif section == "Admin" and st.session_state.role == "admin":
        show_admin(df)

# ============================================================
#  ENTRY POINT
# ============================================================
if not st.session_state.logged_in:
    show_login()
else:
    show_app()
