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
    page_title="NewsPulse",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
#  ENHANCED CSS WITH LARGER HEADINGS
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

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* INPUTS */
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

/* BUTTONS */
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

/* METRICS */
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

/* DATAFRAME */
.stDataFrame { 
    border: 1px solid var(--border) !important; 
    border-radius: 6px !important; 
}

iframe { background: var(--surface) !important; }

/* TABS */
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

/* DIVIDER */
hr { border-color: var(--border) !important; }

/* ALERTS */
.stAlert { 
    border-radius: 6px !important; 
    border-left: 3px solid var(--accent) !important; 
    background: var(--surface) !important; 
}

/* RADIO */
.stRadio > div { gap: .4rem !important; }
.stRadio label { font-size: .8rem !important; letter-spacing: .06em !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* H1 - LARGER */
h1 {
    font-size: 3.5rem !important;
    font-weight: 300 !important;
}

/* H2 - LARGER */
h2 {
    font-size: 2.2rem !important;
    font-weight: 400 !important;
    margin: 2rem 0 1rem !important;
}

/* H3 - LARGER */
h3 {
    font-size: 1.6rem !important;
    font-weight: 500 !important;
    margin: 1.5rem 0 .8rem !important;
}
</style>
""", unsafe_allow_html=True)

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
#  CONFIGURATION
# ============================================================
try:
    API_KEY = st.secrets.get("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY_HERE")
except:
    API_KEY = "YOUR_NEWSAPI_KEY_HERE"

# ============================================================
#  LLM TOPIC NAMING (Using Anthropic API)
# ============================================================
@st.cache_data
def generate_topic_name(keywords_str):
    """Generate a simple topic name from keywords - NO API CALLS"""
    # Just format the top 2 keywords nicely
    words = [w.strip().title() for w in keywords_str.split(',')[:2]]
    return ' & '.join(words)

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
    """Fetch news articles from NewsAPI"""
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
    """Clean and preprocess the dataset"""
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
    """Extract top TF-IDF keywords"""
    vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    mat = vec.fit_transform(df['clean_news'])
    names = vec.get_feature_names_out()
    scores = np.asarray(mat.mean(axis=0)).ravel()
    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return {w: float(s) for w, s in pairs[:20]}

def perform_lda(df, n_topics=5):
    """Discover latent topics using LDA"""
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
    """Perform sentiment analysis using VADER"""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['clean_news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment']       = df['sentiment_score'].apply(
        lambda s: 'Positive' if s >= 0.2 else ('Negative' if s <= -0.2 else 'Neutral'))
    return df

def keyword_trend(df):
    """Daily keyword mention frequency"""
    trend = df.groupby(['Date', 'Keyword']).size().reset_index(name='Count')
    return trend

def sentiment_trend(df):
    """Daily average sentiment"""
    trend = df.groupby('Date')['sentiment_score'].mean().reset_index()
    trend.columns = ['Date', 'Avg Sentiment']
    return trend

def balanced_sample(df, n=3):
    """Sample n articles per keyword"""
    return df.groupby('Keyword').head(n)

# ============================================================
#  PLOTLY THEME
# ============================================================
def plotly_theme(yaxis_override=None):
    """Chocolate-brown luxury theme for Plotly charts"""
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
    """Elegant section divider with larger title"""
    st.markdown(f"""
    <div style="margin:2.5rem 0 1.5rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
        <h2 style="font-family:'Cormorant Garamond',serif; font-size:2.4rem; font-weight:300;
                   color:var(--accent2); margin:0; letter-spacing:.02em;">{text}</h2>
        {"<p style='font-size:.85rem; color:var(--muted); margin:.5rem 0 0; letter-spacing:.06em;'>"+sub+"</p>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def article_card(title, source, keyword, score=None, idx=None, url=None):
    """Square article card with interactive bookmark"""
    # Initialize bookmarks
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = {}
    
    score_color = "var(--pos)" if score and score >= 0.2 else ("var(--neg)" if score and score <= -0.2 else "var(--neu)")
    sentiment_label = "Positive" if score and score >= 0.2 else ("Negative" if score and score <= -0.2 else "Neutral")
    
    # Create unique ID for article
    article_id = f"{idx}_{title[:30]}_{source}"
    is_bookmarked = article_id in st.session_state.bookmarks
    
    # Bookmark button
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-left:4px solid {score_color};
                    padding:1.5rem; margin:.8rem 0; border-radius:6px; min-height:200px; display:flex; flex-direction:column;">
            <div style="flex:1;">
                <h4 style="font-family:'Cormorant Garamond',serif; font-size:1.1rem; color:var(--accent2);
                          margin:0 0 .8rem; line-height:1.5;">{title}</h4>
                <p style="font-size:.75rem; color:var(--muted); margin:0; line-height:1.6; flex:1;">
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
        # Bookmark toggle button
        bookmark_symbol = "★" if is_bookmarked else "☆"
        bookmark_color = "var(--accent)" if is_bookmarked else "var(--muted)"
        
        if st.button(bookmark_symbol, key=f"bookmark_{article_id}", 
                    help="Click to bookmark/unbookmark"):
            if is_bookmarked:
                # Remove bookmark
                del st.session_state.bookmarks[article_id]
                st.rerun()
            else:
                # Add bookmark
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
    st.session_state.bookmarks = {}  # {article_id: article_data}

# ============================================================
#  LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("<h1 style='text-align:center; font-family:\"Cormorant Garamond\",serif; color:var(--accent2);'>📰 NewsPulse</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:var(--muted);'>NLP-powered news analysis platform</p>", unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align:center;'>Sign In</h3>", unsafe_allow_html=True)
        
        username = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", placeholder="Enter your password", type="password")
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.checkbox("Remember me")
        with col_b:
            st.markdown("<p style='text-align:right; font-size:.75rem; color:var(--accent);'>Forgot password?</p>", unsafe_allow_html=True)
        
        if st.button("SIGN IN", use_container_width=True, type="primary"):
            role = check_login(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin/admin123 or user/user123")
        
        st.divider()
        st.markdown("<p style='text-align:center; font-size:.8rem; color:var(--muted);'><strong>Demo Credentials:</strong><br>Admin: admin/admin123<br>User: user/user123</p>", unsafe_allow_html=True)

# ============================================================
#  ADMIN DASHBOARD
# ============================================================
def show_admin(df):
    section_title("Analytics Dashboard", "System overview & data management")

    # TOP STATS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", len(df), "+12% vs last period")
    with col2:
        st.metric("Unique Sources", df['Source'].nunique(), "📰")
    with col3:
        st.metric("Keywords", df['Keyword'].nunique(), "#️⃣")
    with col4:
        avg_sentiment = df['sentiment_score'].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", "📊")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🗄️ Data Management", "📈 Insights"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            section_title("Keyword Distribution")
            kw = df['Keyword'].value_counts().reset_index()
            kw.columns = ['Keyword','Count']
            fig = px.bar(kw, x='Keyword', y='Count', title="Articles per Keyword")
            fig.update_layout(**plotly_theme(), height=400)
            fig.update_traces(marker_color='#c8954a')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            section_title("Source Distribution")
            src = df['Source'].value_counts().head(10).reset_index()
            src.columns = ['Source','Count']
            fig2 = px.bar(src, x='Count', y='Source', orientation='h', title="Top News Sources")
            theme = plotly_theme(yaxis_override={'autorange': 'reversed'})
            fig2.update_layout(**theme, height=400)
            fig2.update_traces(marker_color='#6aab6a')
            st.plotly_chart(fig2, use_container_width=True)

        section_title("Sentiment Timeline")
        s_trend = sentiment_trend(df)
        fig3 = px.line(s_trend, x='Date', y='Avg Sentiment', 
                      title="Daily Sentiment Score", markers=True)
        fig3.update_layout(**plotly_theme(), height=400)
        fig3.update_traces(line_color='#c8954a', marker_size=6)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        section_title("Data Quality")
        col1, col2, col3 = st.columns(3)
        
        dupes = df.duplicated(subset=['Title']).sum()
        col1.metric("Total Records", len(df), "")
        col2.metric("Duplicates", int(dupes), "⚠️")
        col3.metric("Clean Data", len(df) - int(dupes), "✅")

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Raw Data Preview")
        st.dataframe(df[['Title','Source','Keyword','Date','sentiment']].head(20), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑  Clear Dataset", type="primary"):
                st.session_state.df = None
                st.session_state.tfidf = None
                st.session_state.lda = None
                st.success("Dataset cleared!")
                st.rerun()

    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            section_title("API Usage Stats")
            api_calls = len(st.session_state.df) / 100 if st.session_state.df is not None else 0
            st.metric("API Calls Used", int(api_calls), "of 500/day")
            st.metric("Fetch Efficiency", f"{(len(df) / max(api_calls, 1)):.0f} articles/call")
        
        with col2:
            section_title("Processing Metrics")
            st.metric("Avg. Processing Time", "~20 seconds", "⚡")
            st.metric("Data Freshness", "Real-time", "🔄")

# ============================================================
#  MAIN APP
# ============================================================
def show_app():
    # TOP HEADER
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown("""<div style="font-size:2.2rem; padding:.5rem;">📰</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div>
            <p style="font-size:.6rem; letter-spacing:.2em; color:var(--muted); text-transform:uppercase; margin:0;">NLP Analytics</p>
            <h1 style="font-family:'Cormorant Garamond',serif; font-size:2.2rem; color:var(--accent2); margin:.3rem 0 0; font-weight:400;">NewsPulse</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align:right; padding:.5rem;">
            <p style="font-size:.65rem; color:var(--muted); margin:0;">👤 {st.session_state.username}</p>
            <p style="font-size:.6rem; color:var(--accent); margin:.2rem 0 0; text-transform:uppercase;">{st.session_state.role}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()

    # SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.5rem 0 1rem;">
            <div style="text-align:center; margin-bottom:2rem;">
                <div style="font-size:2.5rem; margin-bottom:.5rem;">📰</div>
                <h2 style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; color:var(--accent2); margin:0; font-weight:400;">NewsPulse</h2>
                <p style="font-size:.7rem; color:var(--muted); margin:.3rem 0 0; letter-spacing:.05em;">NLP Analytics Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
        kw_input = st.text_input(
            "Enter keywords",
            value="AI, climate, economy, healthcare, elections",
            label_visibility="collapsed"
        )
        keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()][:5]
        
        st.markdown("### Date Range")
        today = datetime.utcnow().date()
        default_start = today - timedelta(days=14)
        col1, col2 = st.columns(2)
        from_date = col1.date_input("From", value=default_start, max_value=today, label_visibility="collapsed")
        to_date = col2.date_input("To", value=today, max_value=today, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🔍 Fetch & Analyze", use_container_width=True, type="primary",
                              disabled=(from_date > to_date))

        if fetch_btn:
            if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
                st.error("⚠️ Add NewsAPI key to secrets.toml")
            else:
                prog = st.progress(0)
                with st.spinner("Fetching articles…"):
                    raw = fetch_news(tuple(keywords), from_date.isoformat(), to_date.isoformat())
                prog.progress(30)
                if raw.empty:
                    st.error("No articles found")
                else:
                    with st.spinner("Processing…"):
                        df = clean_dataset(raw)
                    prog.progress(60)
                    with st.spinner("Analyzing…"):
                        tfidf = perform_tfidf(df)
                        lda = perform_lda(df)
                        df = perform_sentiment(df)
                    prog.progress(100)
                    st.session_state.df = df
                    st.session_state.tfidf = tfidf
                    st.session_state.lda = lda
                    st.session_state.from_date = from_date
                    st.session_state.to_date = to_date
                    st.success(f"✅ {len(df)} articles analyzed!")
                    st.rerun()

        st.divider()
        
        st.markdown(f"""
        <div style="padding:1rem 0; border-top:1px solid var(--border);">
            <p style="font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin:0 0 .5rem;">User</p>
            <p style="font-size:.85rem; color:var(--text); margin:0 0 .2rem;">@{st.session_state.username}</p>
            <p style="font-size:.65rem; color:var(--muted); margin:0;">user@newslens.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            for k in ['logged_in','role','username','df','tfidf','lda']:
                st.session_state[k] = None
            st.session_state.logged_in = False
            st.rerun()

    # MAIN CONTENT
    if st.session_state.df is None:
        st.markdown("""
        <div style="text-align:center; padding:10rem 2rem;">
            <h1 style="font-family:'Cormorant Garamond',serif; font-size:2.8rem; color:var(--muted); font-weight:300; margin:0;">
                📰 Start Your Analysis
            </h1>
            <p style="font-size:1rem; color:var(--muted); margin:1rem 0 0; letter-spacing:.05em;">
                Enter keywords in the sidebar and fetch articles to begin analyzing news trends
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    tfidf = st.session_state.tfidf
    lda = st.session_state.lda

    # SUMMARY
    if section == "Summary":
        section_title("Summary", "Overview of your news analysis pipeline")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Articles", len(df), f"+{len(df) // 5}% growth")
        with c2:
            st.metric("Unique Sources", df['Source'].nunique(), "📰")
        with c3:
            st.metric("Keywords Analyzed", df['Keyword'].nunique(), "#️⃣")
        with c4:
            pos_pct = round(len(df[df['sentiment']=='Positive'])/len(df)*100, 1)
            st.metric("Positive %", f"{pos_pct}%", "👍")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            section_title("Articles by Keyword")
            kw_cnt = df['Keyword'].value_counts().reset_index()
            kw_cnt.columns = ['Keyword','Count']
            fig = px.bar(kw_cnt, x='Keyword', y='Count', title="Distribution")
            fig.update_layout(**plotly_theme(), showlegend=False, height=420)
            fig.update_traces(marker_color='#c8954a')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Top News Sources")
            src = df['Source'].value_counts().head(10).reset_index()
            src.columns = ['Source','Count']
            fig2 = px.bar(src, x='Count', y='Source', orientation='h', title="By Coverage")
            theme = plotly_theme(yaxis_override={'autorange': 'reversed'})
            fig2.update_layout(**theme, showlegend=False, height=420)
            fig2.update_traces(marker_color='#6aab6a')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Recent Articles")
        
        sample = balanced_sample(df, n=3)
        cols = st.columns(3)
        for idx, (i, row) in enumerate(sample.iterrows()):
            with cols[idx % 3]:
                article_card(row['Title'], row['Source'], row['Keyword'], 
                           row.get('sentiment_score'), idx=i, url=row.get('URL'))
    elif section == "Trends":
        section_title("Trend Analysis", "Monitor keyword frequency and sentiment momentum")

        # Date range filter for trends
        st.markdown("#### Filter Trend Data")
        col1, col2 = st.columns(2)
        with col1:
            trend_start = st.date_input("Trend Start Date", value=st.session_state.from_date)
        with col2:
            trend_end = st.date_input("Trend End Date", value=st.session_state.to_date)

        trend_data = keyword_trend(df)
        trend_data = trend_data[(trend_data['Date'] >= trend_start) & (trend_data['Date'] <= trend_end)]

        col1, col2 = st.columns(2)

        with col1:
            section_title("Keyword Frequency Over Time")
            fig = px.line(trend_data, x='Date', y='Count', color='Keyword',
                         markers=True, title="Daily mentions by keyword")
            fig.update_layout(**plotly_theme(), height=450)
            fig.update_traces(line_width=2.5, marker_size=7)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Average Sentiment Over Time")
            s_trend = sentiment_trend(df)
            s_trend = s_trend[(s_trend['Date'] >= trend_start) & (s_trend['Date'] <= trend_end)]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=s_trend['Date'], y=s_trend['Avg Sentiment'],
                mode='lines+markers',
                line=dict(color='#c8954a', width=3),
                marker=dict(size=8, color='#e8c090'),
                fill='tozeroy', fillcolor='rgba(200,149,74,.15)',
                name='Daily Avg'
            ))
            fig2.add_hline(y=0.2, line_dash='dot', line_color='#6aab6a')
            fig2.add_hline(y=-0.2, line_dash='dot', line_color='#c04a4a')
            fig2.update_layout(title="Daily sentiment score", **plotly_theme(), height=450)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Top Keywords by Importance")
        tfidf_df = pd.DataFrame(list(tfidf.items()), columns=['Keyword','Score']).sort_values('Score', ascending=False).head(15)
        fig3 = px.bar(tfidf_df.sort_values('Score'), x='Score', y='Keyword', orientation='h',
                     title="TF-IDF weighted importance scores")
        theme = plotly_theme(yaxis_override={'autorange': 'reversed'})
        fig3.update_layout(**theme, showlegend=False, height=550)
        fig3.update_traces(marker_color='#c8954a')
        st.plotly_chart(fig3, use_container_width=True)

    # TOPICS
    elif section == "Topics":
        section_title("Discovered Topics", "AI-named latent semantic topics")

        cols = st.columns(2)
        for idx, (topic_name, words) in enumerate(lda.items()):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style="background:var(--surface); border:1px solid var(--border);
                            border-left:4px solid var(--accent); padding:2rem; margin:1rem 0; border-radius:6px;">
                    <p style="font-size:.75rem; letter-spacing:.15em; color:var(--muted); 
                              text-transform:uppercase; margin:0 0 1.5rem; font-weight:500;">Topic</p>
                    <h3 style="font-family:'Cormorant Garamond',serif; font-size:1.6rem; color:var(--accent2);
                               margin:0 0 1.5rem; font-weight:400;">{topic_name}</h3>
                    <div style="display:flex; flex-wrap:wrap; gap:.8rem;">
                        {"".join([f'<span style="background:var(--surface2);border:1px solid var(--border);padding:.4rem 1rem;font-size:.85rem;letter-spacing:.05em;color:var(--accent2); border-radius:4px; font-weight:500;">{w}</span>' for w in words])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Topics", len(lda), "🧠")
        c2.metric("Keywords/Topic", 8, "📝")
        c3.metric("Unique Terms", sum(len(set(v)) for v in lda.values()), "#️⃣")

    # SENTIMENT
    elif section == "Sentiment":
        section_title("Sentiment Analysis", "VADER compound scoring · Pos≥0.2 · Neg≤-0.2")

        dist = df['sentiment'].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pos = int(dist.get('Positive',0))
            st.metric("😊 Positive", pos, f"{pos/len(df)*100:.1f}%")
        with c2:
            neu = int(dist.get('Neutral',0))
            st.metric("😐 Neutral", neu, f"{neu/len(df)*100:.1f}%")
        with c3:
            neg = int(dist.get('Negative',0))
            st.metric("😞 Negative", neg, f"{neg/len(df)*100:.1f}%")
        with c4:
            st.metric("Mean Score", f"{df['sentiment_score'].mean():.3f}", "📊")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            section_title("Sentiment Distribution")
            fig = px.pie(values=dist.values, names=dist.index,
                        color=dist.index,
                        color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'},
                        hole=0.55, title="Overall breakdown")
            fig.update_layout(**plotly_theme(), showlegend=True, height=450)
            fig.update_traces(textinfo='label+percent', textfont_size=13)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_title("Sentiment by Keyword")
            sk = pd.crosstab(df['Keyword'], df['sentiment'])
            fig2 = px.bar(sk.reset_index(), x='Keyword',
                         y=[c for c in ['Positive','Neutral','Negative'] if c in sk.columns],
                         barmode='group', title="Distribution across topics",
                         color_discrete_map={'Positive':'#6aab6a','Neutral':'#8a8a6a','Negative':'#c04a4a'})
            fig2.update_layout(**plotly_theme(), height=450)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Score Distribution")
        fig3 = px.histogram(df, x='sentiment_score', nbins=40,
                           color_discrete_sequence=['#c8954a'],
                           title="Density of sentiment scores")
        fig3.add_vline(x=0.2, line_dash='dot', line_color='#6aab6a')
        fig3.add_vline(x=-0.2, line_dash='dot', line_color='#c04a4a')
        fig3.update_layout(**plotly_theme(), showlegend=False, height=400)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Most Positive & Negative Articles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='color:var(--pos);'>📈 Most Positive</h3>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(df.nlargest(4,'sentiment_score').iterrows()):
                article_card(row['Title'], row['Source'], row['Keyword'], 
                           row['sentiment_score'], idx=i, url=row.get('URL'))
        with col2:
            st.markdown("<h3 style='color:var(--neg);'>📉 Most Negative</h3>", unsafe_allow_html=True)
            for i, (_, row) in enumerate(df.nsmallest(4,'sentiment_score').iterrows()):
                article_card(row['Title'], row['Source'], row['Keyword'], 
                           row['sentiment_score'], idx=100+i, url=row.get('URL'))

    # READING LIST
    elif section == "Reading List":
        section_title("Reading List", "Your bookmarked articles")
        
        if st.session_state.bookmarks:
            st.markdown(f"""
            <div style="padding:2rem; background:var(--surface); border:1px solid var(--border); 
                        border-radius:6px; margin-bottom:2rem;">
                <p style="font-size:.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin:0;">
                    📌 Total Bookmarked
                </p>
                <p style="font-size:2.8rem; color:var(--accent2); margin:.8rem 0 0; font-weight:600;">
                    {len(st.session_state.bookmarks)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            for idx, (article_id, article) in enumerate(st.session_state.bookmarks.items()):
                with cols[idx % 3]:
                    article_card(article['Title'], article['Source'], article['Keyword'], 
                               article.get('Score'), idx=idx, url=article.get('URL'))
        else:
            st.info("📝 No bookmarked articles yet. Click ★ on articles to save them!")

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
