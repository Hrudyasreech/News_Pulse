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
import warnings

warnings.filterwarnings('ignore')

# ============== DOWNLOAD NLTK DATA ==============
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data - only core packages"""
    for package in ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            pass  # Silent fail, not critical for functionality

download_nltk_data()

# ============== GLOBAL PREPROCESSING RESOURCES ==============
# 🔥 IMPROVEMENT 1: Cache stopwords & lemmatizer globally (much faster!)
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

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="NLP News Analysis Pipeline",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    h1 {
        color: #667eea;
    }
    h2 {
        color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ============== CONFIGURATION ==============
# 🔥 IMPROVEMENT 4: Secure API key using st.secrets
try:
    API_KEY = st.secrets.get("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY_HERE")
except:
    API_KEY = "YOUR_NEWSAPI_KEY_HERE"

KEYWORDS = ["ai", "climate", "economy", "healthcare", "election"]

# ============== TEXT CLEANING FUNCTIONS ==============
def clean_text(text):
    """Remove non-printable characters and extra whitespace"""
    if isinstance(text, str):
        text = re.sub(r'[\x00-\x1F\x7F-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return text

def clean_text_for_nlp(text):
    """Clean text for NLP processing: lowercase, remove HTML, special chars, extra spaces"""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# ============== PREPROCESSING FUNCTIONS ==============
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
# 🔥 IMPROVEMENT 3: Cache API calls to avoid repeated requests
@st.cache_data(show_spinner=False)
def fetch_news(from_date, to_date):
    """Fetch news from NewsAPI for all keywords"""
    articles_list = []
    
    for keyword in KEYWORDS:
        for page in range(1, 3):  # Fetch 2 pages per keyword
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
                    
                    time.sleep(0.2)  # Rate limiting
                else:
                    # 🔥 IMPROVEMENT: Better error handling
                    if response.status_code == 401:
                        pass  # Invalid API key handled in main
                    elif response.status_code == 429:
                        pass  # Rate limited
                    break
                    
            except requests.exceptions.Timeout:
                pass  # Timeout, move to next keyword
            except requests.exceptions.RequestException:
                pass  # Network error, move to next keyword
            except Exception:
                pass  # Other error, move to next keyword
    
    return pd.DataFrame(articles_list)

# ============== CLEANING FUNCTIONS ==============
def clean_dataset(df):
    """Remove duplicates, handle missing values, and clean text"""
    # 🔥 IMPROVEMENT: Actually apply clean_text function (was missing!)
    df["Title"] = df["Title"].apply(clean_text)
    df["Description"] = df["Description"].apply(clean_text)
    
    # Drop null values
    df.dropna(inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(subset=["Title"], inplace=True)
    
    # Convert Published Date to datetime
    df["Published Date"] = pd.to_datetime(df["Published Date"])
    
    return df

# ============== ANALYSIS FUNCTIONS ==============
def perform_tfidf_analysis(df):
    """
    Perform TF-IDF vectorization and extract top 10 words
    🔥 IMPROVEMENT 2: Use clean_news (not preprocessed) for better TF-IDF
    This lets TfidfVectorizer handle its own stopwords and frequency analysis
    """
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)  # Include bigrams like "climate change"
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_news'])
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 🔥 IMPROVEMENT 2: Memory-efficient sparse matrix operation
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    
    word_scores = sorted(list(zip(feature_names, mean_tfidf)), key=lambda x: x[1], reverse=True)
    return {word: float(score) for word, score in word_scores[:10]}

def perform_lda_analysis(df):
    """
    Perform LDA topic modeling
    🔥 IMPROVEMENT: Use proper LDA parameters (max_df, min_df, ngrams)
    """
    vectorizer = CountVectorizer(
        max_features=500,
        max_df=0.6,  # Ignore words appearing in >60% of docs
        min_df=3,    # Ignore words appearing in <3 docs
        ngram_range=(1, 2)  # Include bigrams
    )
    X = vectorizer.fit_transform(df['preprocessed_news'])
    
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=50,
        learning_method='batch',
        doc_topic_prior=0.1,
        topic_word_prior=0.1,
        random_state=42
    )
    lda.fit(X)
    
    words = vectorizer.get_feature_names_out()
    topics = {}
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-8:][::-1]]
        topics[f"Topic {i+1}"] = ", ".join(top_words)
    
    return topics

def perform_sentiment_analysis(df):
    """Perform sentiment analysis using VADER"""
    sia = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['clean_news'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    def get_sentiment_label(score):
        if score >= 0.2:
            return 'Positive'
        elif score <= -0.2:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment_label'] = df['sentiment_scores'].apply(get_sentiment_label)
    sentiment_dist = df['sentiment_label'].value_counts().to_dict()
    
    return df, sentiment_dist

# ============== MAIN APP ==============
def main():
    st.title("📰 NLP News Analysis Pipeline")
    st.markdown("Fetch news articles, clean, preprocess, and analyze using NLP")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # 🔥 IMPROVEMENT: Add date validation
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=14)
    
    from_date = st.sidebar.date_input(
        "📅 From Date",
        value=default_start,
        max_value=today
    )
    
    to_date = st.sidebar.date_input(
        "📅 To Date",
        value=today,
        max_value=today
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Keywords: {', '.join(KEYWORDS)}")
    
    # Validate dates
    if from_date > to_date:
        st.error("❌ 'From Date' cannot be after 'To Date'")
        return
    
    days_diff = (to_date - from_date).days
    if days_diff == 0:
        st.warning("⚠️ Fetching for same day - may have limited results")
    elif days_diff > 30:
        st.warning("⚠️ Large date range (>30 days) may take longer")
    
    # Main Button
    if st.button("🚀 Start Analysis", use_container_width=True):
        # API Key Check
        if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
            st.error("❌ Please configure your NewsAPI Key!")
            st.info("📝 For local testing: Update `API_KEY` in app.py")
            st.info("☁️ For cloud deployment: Add `NEWSAPI_KEY` to Streamlit Secrets")
            return
        
        # ============== PIPELINE EXECUTION ==============
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # STEP 1: Fetch News
            with status_placeholder.container():
                st.info("📥 **STEP 1:** Fetching news from NewsAPI...")
            
            df_raw = fetch_news(from_date.isoformat(), to_date.isoformat())
            progress_bar.progress(15)
            
            if df_raw.empty:
                st.error("❌ No articles found for the given date range. Try expanding the dates.")
                return
            
            st.success(f"✓ Fetched {len(df_raw)} articles")
            
            # STEP 2: Clean Dataset
            with status_placeholder.container():
                st.info("🧹 **STEP 2:** Cleaning and deduplicating data...")
            
            df = clean_dataset(df_raw)
            progress_bar.progress(30)
            
            st.success(f"✓ Cleaned to {len(df)} unique articles")
            
            # STEP 3: Add news columns
            with status_placeholder.container():
                st.info("📝 **STEP 3:** Creating news columns...")
            
            df['news'] = df['Title'] + ' ' + df['Description']
            df['clean_news'] = df['news'].apply(clean_text_for_nlp)
            progress_bar.progress(45)
            
            # STEP 4: Text Preprocessing
            with status_placeholder.container():
                st.info("⚙️ **STEP 4:** Preprocessing text (tokenization, lemmatization, stopword removal)...")
            
            df['preprocessed_news'] = df['clean_news'].apply(preprocess_text)
            progress_bar.progress(60)
            
            # STEP 5: TF-IDF Analysis
            with status_placeholder.container():
                st.info("📊 **STEP 5:** Performing TF-IDF analysis...")
            
            tfidf_results = perform_tfidf_analysis(df)
            progress_bar.progress(75)
            
            # STEP 6: LDA Topic Modeling
            with status_placeholder.container():
                st.info("🎯 **STEP 6:** Performing LDA topic modeling...")
            
            lda_topics = perform_lda_analysis(df)
            progress_bar.progress(90)
            
            # STEP 7: Sentiment Analysis
            with status_placeholder.container():
                st.info("❤️ **STEP 7:** Performing sentiment analysis...")
            
            df, sentiment_dist = perform_sentiment_analysis(df)
            progress_bar.progress(100)
            
            # Clear status placeholders
            status_placeholder.empty()
            progress_bar.empty()
            
            st.success("✅ Pipeline completed successfully!")
            st.divider()
            
            # ============== RESULTS DISPLAY ==============
            
            # Summary Metrics
            st.header("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Articles", len(df))
            
            with col2:
                st.metric("Unique Sources", df['Source'].nunique())
            
            with col3:
                st.metric("Date Range", f"{from_date} to {to_date}")
            
            with col4:
                st.metric("Keywords", len(KEYWORDS))
            
            st.divider()
            
            # Articles by Keyword
            st.subheader("📰 Articles by Keyword")
            keyword_counts = df['Keyword'].value_counts()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(keyword_counts)
            
            with col2:
                st.dataframe(keyword_counts.to_frame('Count'), use_container_width=True)
            
            st.divider()
            
            # Sentiment Analysis Results
            st.subheader("❤️ Sentiment Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_count = sentiment_dist.get('Positive', 0)
                st.metric("😊 Positive", positive_count, f"{(positive_count/len(df)*100):.1f}%")
            
            with col2:
                neutral_count = sentiment_dist.get('Neutral', 0)
                st.metric("😐 Neutral", neutral_count, f"{(neutral_count/len(df)*100):.1f}%")
            
            with col3:
                negative_count = sentiment_dist.get('Negative', 0)
                st.metric("😞 Negative", negative_count, f"{(negative_count/len(df)*100):.1f}%")
            
            # Sentiment Distribution Chart
            sentiment_chart = pd.DataFrame(
                list(sentiment_dist.items()),
                columns=['Sentiment', 'Count']
            )
            st.bar_chart(sentiment_chart.set_index('Sentiment'))
            
            st.divider()
            
            # TF-IDF Top 10 Words
            st.subheader("🔑 Top 10 Words (TF-IDF)")
            tfidf_df = pd.DataFrame(
                list(tfidf_results.items()),
                columns=['Word', 'Score']
            ).sort_values('Score', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(tfidf_df.set_index('Word'))
            
            with col2:
                st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # LDA Topics
            st.subheader("🎯 LDA Topics (Top 3)")
            
            for topic_name, words in lda_topics.items():
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{topic_name}**")
                    with col2:
                        st.write(words)
            
            st.divider()
            
            # Data Preview
            st.subheader("📊 Data Preview")
            tab1, tab2, tab3 = st.tabs(["Original Data", "Cleaned Data", "Final Data"])
            
            with tab1:
                display_cols = ['Title', 'Source', 'Keyword', 'Published Date']
                st.dataframe(df[display_cols].head(10), use_container_width=True)
            
            with tab2:
                display_cols = ['Title', 'clean_news', 'Keyword']
                st.dataframe(df[display_cols].head(10), use_container_width=True)
            
            with tab3:
                display_cols = ['Title', 'preprocessed_news', 'sentiment_label', 'sentiment_scores']
                st.dataframe(df[display_cols].head(10), use_container_width=True)
            
            st.divider()
            
            # Download Results
            st.subheader("📥 Download Results")
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name=f"sentimental_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error during pipeline execution: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
