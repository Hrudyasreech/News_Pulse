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
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

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
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #667eea;
    }
    h2 {
        color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ============== CONFIGURATION ==============
API_KEY = "YOUR_NEWSAPI_KEY_HERE"  # ⚠️ PASTE YOUR API KEY HERE
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
    """Clean text for NLP processing"""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# ============== PREPROCESSING FUNCTIONS ==============
def preprocess_text(text):
    """Tokenize, lemmatize, and remove stopwords"""
    stop_words = set(stopwords.words('english'))
    extra_stopwords = {
        "new", "said", "say", "year", "world",
        "could", "one", "make", "day", "watch", "wa", "ha", "may"
    }
    stop_words = stop_words.union(extra_stopwords)
    lemmatizer = WordNetLemmatizer()
    
    try:
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        filtered_tokens = [word for word in lemmatized_tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    except:
        return text

# ============== API FUNCTIONS ==============
def fetch_news(from_date, to_date, progress_bar=None):
    """Fetch news from NewsAPI for all keywords"""
    articles_list = []
    total_keywords = len(KEYWORDS)
    
    for idx, keyword in enumerate(KEYWORDS):
        st.info(f"🔄 Fetching articles for keyword: **{keyword}**")
        
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
                    
                    time.sleep(0.5)  # Rate limiting
                else:
                    st.warning(f"⚠️ Failed to fetch {keyword}, page {page}")
            
            except Exception as e:
                st.error(f"❌ Error fetching {keyword}: {str(e)}")
        
        if progress_bar:
            progress_bar.progress((idx + 1) / total_keywords)
    
    return pd.DataFrame(articles_list)

# ============== DATA CLEANING FUNCTIONS ==============
def clean_dataset(df):
    """Remove duplicates, handle missing values, and clean text"""
    # Remove non-printable characters
    df["Title"] = df["Title"].apply(clean_text)
    df["Description"] = df["Description"].apply(clean_text)
    
    # Drop null values
    df.dropna(inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(subset=["Title"], inplace=True)
    
    # Convert Published Date to datetime
    df["Published Date"] = pd.to_datetime(df["Published Date"])
    
    return df

# ============== FEATURE ENGINEERING ==============
def add_news_columns(df):
    """Create news and clean_news columns"""
    df['news'] = df['Title'] + ' ' + df['Description']
    df['clean_news'] = df['news'].apply(clean_text_for_nlp)
    return df

def add_preprocessing(df):
    """Add preprocessed_news column"""
    df['preprocessed_news'] = df['clean_news'].apply(preprocess_text)
    return df

# ============== ANALYSIS FUNCTIONS ==============
def perform_tfidf_analysis(df):
    """Perform TF-IDF vectorization and extract top 10 words"""
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_news'])
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    mean_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
    
    word_scores = list(zip(feature_names, mean_tfidf))
    sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    top_10_words = sorted_word_scores[:10]
    
    tfidf_results = {word: float(score) for word, score in top_10_words}
    return tfidf_results

def perform_lda_analysis(df):
    """Perform LDA topic modeling"""
    bow_vectorizer = CountVectorizer(max_features=500, max_df=0.6, min_df=3, ngram_range=(1, 2))
    X_bow = bow_vectorizer.fit_transform(df['preprocessed_news'])
    
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=50,
        learning_method='batch',
        doc_topic_prior=0.1,
        topic_word_prior=0.1,
        random_state=42
    )
    lda.fit(X_bow)
    
    feature_names = bow_vectorizer.get_feature_names_out()
    topics = {}
    
    for idx, topic in enumerate(lda.components_):
        top_words_indices = topic.argsort()[-8:][::-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics[f"Topic {idx + 1}"] = ", ".join(top_words)
    
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
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📰 NLP News Analysis Pipeline")
    
    with col2:
        st.write("")
        st.write("")
        api_status = "✅ Configured" if API_KEY != "YOUR_NEWSAPI_KEY_HERE" else "❌ NOT CONFIGURED"
        st.metric("API Status", api_status)
    
    st.markdown("""
    This application fetches news articles from NewsAPI, processes them through an NLP pipeline, 
    and performs sentiment analysis, TF-IDF keyword extraction, and LDA topic modeling.
    """)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        st.sidebar.error("❌ **API Key not configured!**\n\nPlease update the `API_KEY` variable in app.py with your NewsAPI key from https://newsapi.org")
    
    today = datetime.utcnow().date()
    fourteen_days_ago = today - timedelta(days=14)
    
    from_date = st.sidebar.date_input(
        "📅 From Date:",
        value=fourteen_days_ago,
        max_value=today
    )
    
    to_date = st.sidebar.date_input(
        "📅 To Date:",
        value=today,
        max_value=today
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Keywords")
    st.sidebar.info(f"Analyzing news for: {', '.join(KEYWORDS)}")
    
    # Main Content
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        run_button = st.button("🚀 Start Analysis", use_container_width=True, key="run_btn")
    
    if run_button:
        if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
            st.error("❌ Please configure your NewsAPI key first!")
            return
        
        if from_date > to_date:
            st.error("❌ 'From Date' cannot be after 'To Date'")
            return
        
        # Create a container for the pipeline
        pipeline_container = st.container()
        
        with pipeline_container:
            st.markdown("---")
            st.header("🔄 Pipeline Execution")
            
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                # Step 1: Fetch News
                with status_placeholder.container():
                    st.info("📥 **STEP 1:** Fetching news from NewsAPI...")
                progress_bar = progress_placeholder.progress(0)
                
                df_raw = fetch_news(from_date.isoformat(), to_date.isoformat(), progress_bar)
                
                if len(df_raw) == 0:
                    st.error("❌ No articles found for the given date range. Try expanding the dates.")
                    return
                
                progress_placeholder.progress(0.15)
                
                # Step 2: Clean Dataset
                with status_placeholder.container():
                    st.info("🧹 **STEP 2:** Cleaning dataset (removing duplicates, handling missing values)...")
                df = clean_dataset(df_raw)
                progress_placeholder.progress(0.30)
                
                # Step 3: Add news columns
                with status_placeholder.container():
                    st.info("📝 **STEP 3:** Creating news columns...")
                df = add_news_columns(df)
                progress_placeholder.progress(0.45)
                
                # Step 4: Text preprocessing
                with status_placeholder.container():
                    st.info("⚙️ **STEP 4:** Preprocessing text (tokenization, lemmatization, stopword removal)...")
                df = add_preprocessing(df)
                progress_placeholder.progress(0.60)
                
                # Step 5: TF-IDF Analysis
                with status_placeholder.container():
                    st.info("📊 **STEP 5:** Performing TF-IDF analysis...")
                tfidf_results = perform_tfidf_analysis(df)
                progress_placeholder.progress(0.75)
                
                # Step 6: LDA Topic Modeling
                with status_placeholder.container():
                    st.info("🎯 **STEP 6:** Performing LDA topic modeling...")
                lda_topics = perform_lda_analysis(df)
                progress_placeholder.progress(0.90)
                
                # Step 7: Sentiment Analysis
                with status_placeholder.container():
                    st.info("❤️ **STEP 7:** Performing sentiment analysis...")
                df, sentiment_dist = perform_sentiment_analysis(df)
                progress_placeholder.progress(1.0)
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()
                
                st.success("✅ Pipeline completed successfully!")
                st.markdown("---")
                
                # ============== RESULTS DISPLAY ==============
                # Summary Statistics
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
                
                st.markdown("---")
                
                # Articles by Keyword
                st.subheader("📰 Articles by Keyword")
                keyword_counts = df['Keyword'].value_counts()
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.bar_chart(keyword_counts)
                
                with col2:
                    st.dataframe(keyword_counts.to_frame('Count'), use_container_width=True)
                
                st.markdown("---")
                
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
                
                st.markdown("---")
                
                # TF-IDF Top 10 Words
                st.subheader("🔑 Top 10 Words (TF-IDF)")
                tfidf_df = pd.DataFrame(
                    list(tfidf_results.items()),
                    columns=['Word', 'Score']
                ).sort_values('Score', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.bar_chart(tfidf_df.set_index('Word'))
                
                with col2:
                    st.dataframe(tfidf_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # LDA Topics
                st.subheader("🎯 LDA Topics (Top 3)")
                
                for topic_name, words in lda_topics.items():
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.write(f"**{topic_name}**")
                        with col2:
                            st.write(words)
                
                st.markdown("---")
                
                # Data Preview
                st.subheader("📊 Data Preview")
                tab1, tab2, tab3 = st.tabs(["Original Data", "Cleaned Data", "Preprocessed Data"])
                
                with tab1:
                    display_cols = ['Title', 'Source', 'Keyword', 'Published Date']
                    st.dataframe(df[display_cols].head(10), use_container_width=True)
                
                with tab2:
                    display_cols = ['Title', 'clean_news', 'Keyword']
                    st.dataframe(df[display_cols].head(10), use_container_width=True)
                
                with tab3:
                    display_cols = ['preprocessed_news', 'sentiment_label', 'sentiment_scores']
                    st.dataframe(df[display_cols].head(10), use_container_width=True)
                
                st.markdown("---")
                
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
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"❌ Error during pipeline execution: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()