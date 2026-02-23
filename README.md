# 📰 NLP News Analysis Pipeline - Streamlit App

A complete end-to-end NLP pipeline for analyzing news articles fetched from NewsAPI. This Streamlit app performs data fetching, cleaning, text preprocessing, and advanced NLP analysis including sentiment analysis, TF-IDF keyword extraction, and LDA topic modeling.

## 🚀 Features

- **📥 News Fetching**: Automatically fetch news from NewsAPI for multiple keywords
- **🧹 Data Cleaning**: Remove duplicates, handle missing values, clean non-printable characters
- **📝 Text Processing**: Lowercasing, HTML tag removal, special character removal, whitespace normalization
- **⚙️ Preprocessing**: Tokenization, lemmatization, and stopword removal
- **📊 TF-IDF Analysis**: Extract top 10 keywords by TF-IDF score
- **🎯 LDA Topic Modeling**: Identify top 3 topics from articles using Latent Dirichlet Allocation
- **❤️ Sentiment Analysis**: VADER-based sentiment classification (Positive, Negative, Neutral)
- **📥 CSV Export**: Download complete results with all processing stages

## 📋 Pipeline Stages

1. **STEP 1**: Fetch news from NewsAPI (2 pages per keyword)
2. **STEP 2**: Clean dataset (remove duplicates, handle missing values)
3. **STEP 3**: Create news columns (combine title + description)
4. **STEP 4**: Text preprocessing (tokenization, lemmatization, stopword removal)
5. **STEP 5**: TF-IDF analysis (extract top 10 keywords)
6. **STEP 6**: LDA topic modeling (find top 3 topics)
7. **STEP 7**: Sentiment analysis (classify sentiment)

## 📊 Output Columns

The final CSV file contains:
- **Title**: News article title
- **Description**: Article description
- **Source**: News source name
- **Published Date**: Publication date
- **Keyword**: Search keyword used
- **news**: Combined title + description
- **clean_news**: Cleaned text (lowercase, no special chars)
- **preprocessed_news**: Preprocessed text (tokenized, lemmatized, stopwords removed)
- **sentiment_scores**: VADER compound sentiment score (-1 to 1)
- **sentiment_label**: Sentiment classification (Positive/Negative/Neutral)

## 🔧 Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd nlp-news-analysis-pipeline

# Or just download the files
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

### Get NewsAPI Key
1. Visit [https://newsapi.org](https://newsapi.org)
2. Sign up for a free account
3. Copy your API key

### Configure API Key
Open `app.py` and replace the placeholder:
```python
API_KEY = "YOUR_NEWSAPI_KEY_HERE"  # Replace with your actual key
```

Change to:
```python
API_KEY = "your_actual_newsapi_key_here"
```

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📱 How to Use

1. **Configure Dates**: 
   - Use the sidebar to select start and end dates
   - Default is last 14 days

2. **Start Analysis**:
   - Click the "🚀 Start Analysis" button
   - Watch the pipeline execute through all 7 steps

3. **View Results**:
   - **Summary Statistics**: Total articles, unique sources, date range
   - **Articles by Keyword**: Distribution across search terms
   - **Sentiment Analysis**: Positive, Negative, Neutral counts with charts
   - **Top 10 Keywords**: TF-IDF scores with visualization
   - **LDA Topics**: Top 3 topics with their keywords
   - **Data Preview**: Original, cleaned, and preprocessed data

4. **Download Results**:
   - Click "📥 Download Full Results as CSV" to save results

## 🎯 Keywords Analyzed

Default keywords are:
- `ai`
- `climate`
- `economy`
- `healthcare`
- `election`

To change keywords, edit `KEYWORDS` in `app.py`:
```python
KEYWORDS = ["ai", "climate", "economy", "healthcare", "election"]
```

## 📊 Analysis Details

### Sentiment Analysis
- **Positive**: Score ≥ 0.2
- **Neutral**: Score between -0.2 and 0.2
- **Negative**: Score ≤ -0.2

### TF-IDF
- Extracts 1000 features
- Shows top 10 words by average TF-IDF score
- Higher score = more important keyword

### LDA Topic Modeling
- 3 topics extracted
- 8 top words per topic
- Helps understand main themes in news

## 🛠️ Troubleshooting

### Issue: API Key not configured
**Solution**: Update `API_KEY` in `app.py` with your actual NewsAPI key

### Issue: No articles found
**Solution**: 
- Try expanding the date range
- Check if keywords are valid
- Ensure API key has sufficient quota

### Issue: Memory error with large datasets
**Solution**: 
- Reduce the date range
- Use fewer keywords
- Increase system RAM

### Issue: NLTK data not found
**Solution**: The app automatically downloads required NLTK data on first run

## 📦 Dependencies

- **streamlit**: Web app framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **requests**: HTTP client for API calls
- **nltk**: Natural language processing
- **scikit-learn**: Machine learning & text vectorization

## 📈 Performance Tips

- First run will download NLTK data (~500MB)
- Processing 400+ articles takes ~2-3 minutes
- For faster results, reduce date range or number of keywords

## 🔐 API Usage

- Free plan: 100 requests/day
- Paid plans available at [newsapi.org](https://newsapi.org)
- Each keyword query = 1 request

## 📝 Example Output

```
Total Articles: 418
Unique Sources: 85

Top 10 Words (TF-IDF):
1. ai: 0.0349
2. trump: 0.0304
3. election: 0.0255
...

LDA Topics:
Topic 1: ai, healthcare, economy, people, health...
Topic 2: ai, climate, trump, change, job...
Topic 3: election, trump, president, party...

Sentiment Distribution:
😊 Positive: 197 (47.1%)
😐 Neutral: 91 (21.8%)
😞 Negative: 130 (31.1%)
```

## 🤝 Contributing

Feel free to modify and extend the pipeline!

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

NLP Analysis Pipeline - News Article Analysis System

## ⚠️ Disclaimer

This tool is for educational and research purposes. Ensure you comply with NewsAPI's terms of service and respect copyright laws when using the fetched data.

---

**Need Help?**
- Check the error messages in the app
- Review the console output for detailed logs
- Ensure all dependencies are installed: `pip install -r requirements.txt`
