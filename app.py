import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="NLP News Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== DOWNLOAD NLTK DATA ==============
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    for package in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
        try:
            nltk.download(package, quiet=True)
        except:
            pass

download_nltk_data()

# ============== GLOBAL RESOURCES ==============
@st.cache_resource
def get_preprocessing_resources():
    """Get stopwords and lemmatizer (cached globally)"""
    stop_words = set(stopwords.words('english'))
    extra_stopwords = {
        "new", "said", "say", "year", "world",
        "could", "one", "make", "day", "watch", "wa", "ha", "may"
    }
    stop_words = stop_words.union(extra_stopwords)
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

STOP_WORDS, LEMMATIZER = get_preprocessing_resources()

# ============== CONFIGURATION ==============
try:
    API_KEY = st.secrets.get("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY_HERE")
except:
    API_KEY = "YOUR_NEWSAPI_KEY_HERE"

KEYWORDS = ["ai", "climate", "economy", "healthcare", "election"]

# ============== TEXT CLEANING ==============
def clean_text(text):
    """Remove non-printable characters and extra whitespace"""
    if isinstance(text, str):
        text = re.sub(r'[\x00-\x1F\x7F-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return text

def clean_text_for_nlp(text):
    """Clean text for NLP processing"""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """Tokenize, lemmatize, and remove stopwords"""
    try:
        tokens = word_tokenize(text)
        lemmatized_tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
        filtered_tokens = [word for word in lemmatized_tokens if word not in STOP_WORDS]
        return ' '.join(filtered_tokens)
    except:
        return text

# ============== API FUNCTIONS ==============
@st.cache_data(show_spinner=False)
def fetch_news(from_date, to_date):
    """Fetch news from NewsAPI for all keywords"""
    articles_list = []
    
    for keyword in KEYWORDS:
        for page in range(1, 3):
            try:
                url = f"https://newsapi.org/v2/everything?q={keyword}&from={from_date}&to={to_date}&pageSize=100&page={page}&apiKey={API_KEY}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    if not articles:
                        break
                    
                    for article in articles:
                        articles_list.append({
                            "Title": article.get("title", ""),
                            "Description": article.get("description", ""),
                            "Source": article.get("source", {}).get("name", ""),
                            "Published Date": article.get("publishedAt", ""),
                            "Keyword": keyword
                        })
                    time.sleep(0.2)
                else:
                    break
            except:
                break
    
    return pd.DataFrame(articles_list)

def clean_dataset(df):
    """Clean dataset"""
    df["Title"] = df["Title"].apply(clean_text)
    df["Description"] = df["Description"].apply(clean_text)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Title"], inplace=True)
    df["Published Date"] = pd.to_datetime(df["Published Date"])
    df['news'] = df['Title'] + ' ' + df['Description']
    df['clean_news'] = df['news'].apply(clean_text_for_nlp)
    df['preprocessed_news'] = df['clean_news'].apply(preprocess_text)
    return df

# ============== ANALYSIS FUNCTIONS ==============
def perform_tfidf_analysis(df):
    """Perform TF-IDF analysis"""
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_news'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    word_scores = sorted(list(zip(feature_names, mean_tfidf)), key=lambda x: x[1], reverse=True)
    return {word: float(score) for word, score in word_scores[:10]}

def perform_lda_analysis(df):
    """Perform LDA analysis"""
    vectorizer = CountVectorizer(max_features=500, max_df=0.6, min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['preprocessed_news'])
    lda = LatentDirichletAllocation(n_components=3, max_iter=50, learning_method='batch', doc_topic_prior=0.1, topic_word_prior=0.1, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = {}
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-8:][::-1]]
        topics[f"Topic {i+1}"] = ", ".join(top_words)
    return topics

def perform_sentiment_analysis(df):
    """Perform sentiment analysis"""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['clean_news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment_scores'].apply(lambda s: 'Positive' if s >= 0.2 else ('Negative' if s <= -0.2 else 'Neutral'))
    sentiment_dist = df['sentiment_label'].value_counts().to_dict()
    return df, sentiment_dist

# ============== SESSION STATE ==============
if 'df' not in st.session_state:
    st.session_state.df = None
if 'tfidf_results' not in st.session_state:
    st.session_state.tfidf_results = None
if 'lda_topics' not in st.session_state:
    st.session_state.lda_topics = None
if 'sentiment_dist' not in st.session_state:
    st.session_state.sentiment_dist = None

# ============== SIDEBAR ==============
with st.sidebar:
    st.header("⚙️ Data Fetching")
    st.divider()
    
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=14)
    
    from_date = st.date_input("📅 From Date", value=default_start, max_value=today)
    to_date = st.date_input("📅 To Date", value=today, max_value=today)
    
    st.divider()
    st.info(f"🔍 Keywords: {', '.join(KEYWORDS)}")
    st.divider()
    
    # Date Validation
    if from_date > to_date:
        st.error("❌ From Date cannot be after To Date!")
    else:
        days_diff = (to_date - from_date).days
        st.success(f"✓ Date range: {days_diff} days")
        if days_diff == 0:
            st.warning("⚠️ Same day - limited results")
        elif days_diff > 30:
            st.warning("⚠️ Large range (>30 days)")
    
    st.divider()
    
    # Fetch Button
    if st.button("🚀 Fetch & Analyze", use_container_width=True, type="primary"):
        if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
            st.error("❌ Configure NewsAPI Key!")
            st.stop()
        
        if from_date > to_date:
            st.error("❌ Invalid date range!")
            st.stop()
        
        progress = st.progress(0)
        status = st.status("Processing...", expanded=True)
        
        try:
            with status:
                st.write("📥 Fetching news...")
            progress.progress(15)
            
            df_raw = fetch_news(from_date.isoformat(), to_date.isoformat())
            
            if df_raw.empty:
                st.error("❌ No articles found.")
                st.stop()
            
            with status:
                st.write(f"✓ Fetched {len(df_raw)} articles")
                st.write("🔄 Processing...")
            progress.progress(50)
            
            df = clean_dataset(df_raw)
            
            with status:
                st.write(f"✓ Cleaned to {len(df)} articles")
                st.write("📊 Analyzing...")
            progress.progress(75)
            
            tfidf_results = perform_tfidf_analysis(df)
            lda_topics = perform_lda_analysis(df)
            df, sentiment_dist = perform_sentiment_analysis(df)
            
            progress.progress(100)
            
            st.session_state.df = df
            st.session_state.tfidf_results = tfidf_results
            st.session_state.lda_topics = lda_topics
            st.session_state.sentiment_dist = sentiment_dist
            st.session_state.from_date = from_date
            st.session_state.to_date = to_date
            
            with status:
                st.write("✅ Complete!")
            
            status.update(state="complete")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Navigation Radio Buttons
    if st.session_state.df is not None:
        st.header("📂 Navigation")
        section = st.radio(
            "Select View",
            ["📊 Summary", "🔑 Keywords", "🎯 Topics", "❤️ Sentiment", "📥 Download"],
            label_visibility="collapsed"
        )
    else:
        st.info("👈 Fetch data first!")
        section = None

# ============== MAIN CONTENT ==============
st.title("📰 NLP News Analysis Pipeline")

if st.session_state.df is None:
    st.info("👈 Use sidebar to fetch and analyze news articles!")
else:
    df = st.session_state.df
    
    # ============== 📊 SUMMARY PAGE ==============
    if section == "📊 Summary":
        st.subheader("📊 Data Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📰 Total", len(df))
        with col2:
            st.metric("📊 Sources", df['Source'].nunique())
        with col3:
            st.metric("🔍 Keywords", df['Keyword'].nunique())
        with col4:
            st.metric("📅 Date Range", f"{st.session_state.from_date} to {st.session_state.to_date}")
        with col5:
            st.metric("🌐 Unique", len(df['Source'].unique()))
        
        st.divider()
        
        st.write("### 📰 Articles by Keyword")
        keyword_counts = df['Keyword'].value_counts()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(keyword_counts)
        with col2:
            st.dataframe(keyword_counts.to_frame('Count'), use_container_width=True, hide_index=False)
        
        st.divider()
        
        st.write("### 📡 Top News Sources")
        source_counts = df['Source'].value_counts().head(10)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(source_counts)
        with col2:
            st.dataframe(source_counts.to_frame('Count'), use_container_width=True, hide_index=False)
        
        st.divider()
        
        st.write("### 📋 Data Sample")
        st.dataframe(df[['Title', 'Source', 'Keyword', 'Published Date']].head(10), use_container_width=True)
    
    # ============== 🔑 KEYWORDS PAGE ==============
    elif section == "🔑 Keywords":
        st.subheader("🔑 Top 10 Keywords (TF-IDF)")
        
        st.info("TF-IDF measures keyword importance in the corpus. Higher scores = more important.")
        
        tfidf_results = st.session_state.tfidf_results
        tfidf_df = pd.DataFrame(list(tfidf_results.items()), columns=['Keyword', 'Score']).sort_values('Score', ascending=False).reset_index(drop=True)
        tfidf_df.index = tfidf_df.index + 1
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(tfidf_df.set_index('Keyword')['Score'].sort_values(ascending=True))
        with col2:
            st.dataframe(tfidf_df, use_container_width=True)
        
        st.divider()
        
        st.write("### 📈 Detailed Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Keywords", len(tfidf_df))
        with col2:
            st.metric("Max Score", f"{tfidf_df['Score'].max():.4f}")
        with col3:
            st.metric("Avg Score", f"{tfidf_df['Score'].mean():.4f}")
        with col4:
            st.metric("Min Score", f"{tfidf_df['Score'].min():.4f}")
        
        st.divider()
        
        st.write("### 📥 Export")
        csv = tfidf_df.to_csv(index=False)
        st.download_button("📥 Download Keywords CSV", data=csv, file_name="top_keywords.csv", mime="text/csv", use_container_width=True)
    
    # ============== 🎯 TOPICS PAGE ==============
    elif section == "🎯 Topics":
        st.subheader("🎯 LDA Topics Analysis")
        
        st.info("LDA discovers abstract topics from documents. Each topic shows its top 8 relevant words.")
        
        lda_topics = st.session_state.lda_topics
        
        for topic_name, words in lda_topics.items():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{topic_name}**")
                with col2:
                    st.write(words)
                st.divider()
        
        st.write("### 📊 Topic Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Topics", len(lda_topics))
        with col2:
            total_words = sum(len([w.strip() for w in words.split(',')]) for words in lda_topics.values())
            st.metric("Total Words", total_words)
        with col3:
            avg_words = total_words / len(lda_topics)
            st.metric("Avg Words/Topic", f"{avg_words:.1f}")
        
        st.divider()
        
        st.write("### 📥 Export")
        topics_data = [{'Topic': t, 'Words': w} for t, w in lda_topics.items()]
        topics_df = pd.DataFrame(topics_data)
        csv = topics_df.to_csv(index=False)
        st.download_button("📥 Download Topics CSV", data=csv, file_name="lda_topics.csv", mime="text/csv", use_container_width=True)
    
    # ============== ❤️ SENTIMENT PAGE ==============
    elif section == "❤️ Sentiment":
        st.subheader("❤️ Sentiment Analysis")
        
        st.info("VADER sentiment analysis. Classifications: Positive (≥0.2), Neutral (-0.2 to 0.2), Negative (≤-0.2)")
        
        sentiment_dist = st.session_state.sentiment_dist
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            positive = sentiment_dist.get('Positive', 0)
            st.metric("😊 Positive", positive, f"{(positive/len(df)*100):.1f}%")
        with col2:
            neutral = sentiment_dist.get('Neutral', 0)
            st.metric("😐 Neutral", neutral, f"{(neutral/len(df)*100):.1f}%")
        with col3:
            negative = sentiment_dist.get('Negative', 0)
            st.metric("😞 Negative", negative, f"{(negative/len(df)*100):.1f}%")
        with col4:
            st.metric("📊 Total", len(df))
        
        st.divider()
        
        # 🥧 PIE CHART
        st.write("### 🥧 Sentiment Distribution")
        sentiment_chart = pd.DataFrame(list(sentiment_dist.items()), columns=['Sentiment', 'Count'])
        
        fig = px.pie(
            sentiment_chart,
            values='Count',
            names='Sentiment',
            title="Sentiment Distribution",
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
        )
        fig.update_traces(textposition='inside', textinfo='label+percent')
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Sentiment by Keyword
        st.write("### 📊 Sentiment by Keyword")
        sentiment_keyword = pd.crosstab(df['Keyword'], df['sentiment_label'])
        st.dataframe(sentiment_keyword, use_container_width=True)
        
        fig_keyword = px.bar(
            sentiment_keyword.reset_index(),
            x='Keyword',
            y=['Positive', 'Negative', 'Neutral'],
            title="Sentiment by Keyword",
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'},
            barmode='group'
        )
        st.plotly_chart(fig_keyword, use_container_width=True)
        
        st.divider()
        
        # Score distribution
        st.write("### 📈 Sentiment Score Distribution")
        fig_hist = px.histogram(
            df,
            x='sentiment_scores',
            nbins=30,
            title="Distribution of Sentiment Scores",
            labels={'sentiment_scores': 'Sentiment Score', 'count': 'Number of Articles'},
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        st.write("### 📊 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df['sentiment_scores'].mean():.4f}")
        with col2:
            st.metric("Median", f"{df['sentiment_scores'].median():.4f}")
        with col3:
            st.metric("Std Dev", f"{df['sentiment_scores'].std():.4f}")
        with col4:
            st.metric("Range", f"{df['sentiment_scores'].min():.4f} to {df['sentiment_scores'].max():.4f}")
        
        st.divider()
        
        st.write("### 🔍 Most Positive & Negative")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Most Positive**")
            top_pos = df.nlargest(5, 'sentiment_scores')[['Title', 'sentiment_scores', 'Source']]
            for idx, row in top_pos.iterrows():
                st.write(f"**Score:** {row['sentiment_scores']:.4f}")
                st.write(f"*{row['Title']}*")
                st.write(f"Source: {row['Source']}")
                st.divider()
        
        with col2:
            st.write("**Top 5 Most Negative**")
            top_neg = df.nsmallest(5, 'sentiment_scores')[['Title', 'sentiment_scores', 'Source']]
            for idx, row in top_neg.iterrows():
                st.write(f"**Score:** {row['sentiment_scores']:.4f}")
                st.write(f"*{row['Title']}*")
                st.write(f"Source: {row['Source']}")
                st.divider()
        
        st.divider()
        
        st.write("### 📥 Export")
        sentiment_export = df[['Title', 'Source', 'sentiment_scores', 'sentiment_label']].copy()
        csv = sentiment_export.to_csv(index=False)
        st.download_button("📥 Download Sentiment CSV", data=csv, file_name="sentiment_analysis.csv", mime="text/csv", use_container_width=True)
    
    # ============== 📥 DOWNLOAD PAGE ==============
    elif section == "📥 Download":
        st.subheader("📥 Export & Download Results")
        
        st.info("Download your analysis results in various formats.")
        
        # Full Dataset
        st.write("### 1️⃣ Full Dataset")
        csv_full = df.to_csv(index=False)
        st.download_button("📥 Download Full CSV", data=csv_full, file_name=f"analysis_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        
        st.divider()
        
        # Sentiment Data
        st.write("### 2️⃣ Sentiment Analysis")
        sentiment_df = df[['Title', 'Source', 'Keyword', 'Published Date', 'sentiment_scores', 'sentiment_label']].copy()
        csv_sentiment = sentiment_df.to_csv(index=False)
        st.download_button("📥 Download Sentiment CSV", data=csv_sentiment, file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(sentiment_df.head(), use_container_width=True)
        
        st.divider()
        
        # Clean Text
        st.write("### 3️⃣ Cleaned Text Data")
        clean_text_df = df[['Title', 'clean_news', 'preprocessed_news', 'Keyword']].copy()
        csv_clean = clean_text_df.to_csv(index=False)
        st.download_button("📥 Download Cleaned Text CSV", data=csv_clean, file_name=f"cleaned_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        
        st.divider()
        
        # Keywords
        st.write("### 4️⃣ Top 10 Keywords")
        tfidf_results = st.session_state.tfidf_results
        tfidf_df = pd.DataFrame(list(tfidf_results.items()), columns=['Keyword', 'TF-IDF Score']).sort_values('TF-IDF Score', ascending=False)
        csv_tfidf = tfidf_df.to_csv(index=False)
        st.download_button("📥 Download Keywords CSV", data=csv_tfidf, file_name=f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(tfidf_df, use_container_width=True)
        
        st.divider()
        
        # Topics
        st.write("### 5️⃣ LDA Topics")
        lda_topics = st.session_state.lda_topics
        topics_data = [{'Topic': t, 'Words': w} for t, w in lda_topics.items()]
        topics_df = pd.DataFrame(topics_data)
        csv_topics = topics_df.to_csv(index=False)
        st.download_button("📥 Download Topics CSV", data=csv_topics, file_name=f"topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(topics_df, use_container_width=True)
        
        st.divider()
        
        # Summary Report
        st.write("### 6️⃣ Analysis Summary Report")
        summary_data = {
            'Metric': ['From Date', 'To Date', 'Total Articles', 'Unique Sources', 'Keywords', 'Positive', 'Neutral', 'Negative', 'Avg Sentiment', 'Topics'],
            'Value': [str(st.session_state.from_date), str(st.session_state.to_date), len(df), df['Source'].nunique(), df['Keyword'].nunique(), 
                     st.session_state.sentiment_dist.get('Positive', 0), st.session_state.sentiment_dist.get('Neutral', 0), st.session_state.sentiment_dist.get('Negative', 0),
                     f"{df['sentiment_scores'].mean():.4f}", len(lda_topics)]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        st.download_button("📥 Download Summary CSV", data=csv_summary, file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Import
        st.write("### 📤 Import Existing CSV")
        st.write("Upload a previously downloaded CSV to analyze without re-fetching.")
        
        uploaded_file = st.file_uploader("Choose CSV", type="csv")
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(uploaded_df)} rows")
                st.dataframe(uploaded_df.head(), use_container_width=True)
                
                if st.button("✅ Use this data", use_container_width=True):
                    st.session_state.df = uploaded_df
                    st.success("✅ Data loaded! Refresh to see in other views.")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
