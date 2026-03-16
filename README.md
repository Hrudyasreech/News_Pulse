# 📰 NewsPulse 

# NLP Powered News Analytics Platform

NewsPulse is an interactive Streamlit-based news intelligence platform
that fetches real-time news articles and performs advanced Natural
Language Processing (NLP) analysis to uncover trends, topics, and
sentiment.

The application processes news articles using machine learning
techniques such as TF-IDF keyword extraction, LDA topic modeling, and
sentiment analysis, and visualizes insights using interactive
dashboards.

------------------------------------------------------------------------

## 🚀 Features

### 🔐 User Authentication

Secure login system with role-based access.

Demo credentials:

  Role    Username   Password
  ------- ---------- ----------
  Admin   admin      admin123
  User    user       user123

------------------------------------------------------------------------

## 📥 Data Collection

News articles are fetched dynamically using the NewsAPI.

Users can: - Enter up to 5 keywords - Select a date range - Fetch
multiple articles per keyword

Example keywords: AI, climate, economy, healthcare, elections

------------------------------------------------------------------------

## 🧠 NLP Processing Pipeline

### 1️⃣ Data Cleaning

-   Remove duplicates
-   Remove invalid titles
-   Handle missing values
-   Convert publication timestamps

### 2️⃣ Text Preprocessing

Includes: - Lowercasing - Removing HTML tags - Removing special
characters - Tokenization - Stopword removal - Lemmatization

Libraries used: - NLTK - Scikit-learn

------------------------------------------------------------------------

### 3️⃣ TF-IDF Keyword Extraction

Important keywords are extracted using TF-IDF vectorization.

Example:

  Keyword    Score
  ---------- -------
  ai         0.034
  election   0.028
  economy    0.025

------------------------------------------------------------------------

### 4️⃣ Topic Modeling (LDA)

The system discovers hidden topics in the news dataset using Latent
Dirichlet Allocation (LDA).

Example:

Topic: AI & Healthcare\
Keywords: ai, healthcare, technology, data, innovation

------------------------------------------------------------------------

### 5️⃣ Sentiment Analysis

Sentiment is calculated using the VADER sentiment analyzer.

  Score         Label
  ------------- ----------
  ≥ 0.2         Positive
  -0.2 -- 0.2   Neutral
  ≤ -0.2        Negative

------------------------------------------------------------------------

## 📊 Dashboards

### Summary Dashboard

Displays: - Total articles - Unique sources - Keyword distribution - Top
sources - Recent articles

### Trend Analysis

Visualizations include: - Keyword frequency over time - Sentiment
trends - TF-IDF keyword importance

Built using Plotly interactive charts.

### Topic Discovery

Displays LDA-discovered topics and keywords.

### Sentiment Dashboard

Includes: - Sentiment distribution - Sentiment by keyword - Sentiment
score histogram - Most positive and negative articles

### ⭐ Reading List

Users can bookmark articles by clicking the star icon. Bookmarked
articles appear in the Reading List section.

------------------------------------------------------------------------

## 👑 Admin Dashboard

Admin users have access to system analytics including:

-   Dataset overview
-   Keyword distribution
-   Source distribution
-   Data quality checks
-   API usage metrics
-   Raw data preview

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Category           Tools
  ------------------ ---------------
  Frontend           Streamlit
  NLP                NLTK
  Machine Learning   Scikit-learn
  Visualization      Plotly
  Data Processing    Pandas, NumPy
  API                NewsAPI

------------------------------------------------------------------------

## 📦 Installation

### 1️⃣ Clone the Repository

git clone https://github.com/yourusername/newspulse.git\
cd newspulse

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Setup NewsAPI Key

Create a file:

.streamlit/secrets.toml

Add:

NEWSAPI_KEY = "your_newsapi_key_here"

Get a key from https://newsapi.org

### 4️⃣ Run the Application

streamlit run app.py

Open browser: http://localhost:8501

------------------------------------------------------------------------

## 📁 Project Structure

NewsPulse\
│\
├── app.py\
├── requirements.txt\
├── README.md\
└── .streamlit/secrets.toml

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

## 👩‍💻 Author

NewsPulse --- NLP News Analytics Platform
