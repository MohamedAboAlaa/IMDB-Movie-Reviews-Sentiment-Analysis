# IMDB Movie Reviews Sentiment Analysis

A comprehensive sentiment analysis project that classifies movie reviews from the IMDB dataset as either **positive** or **negative** using machine learning techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Saved Models](#saved-models)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete NLP pipeline for sentiment analysis on movie reviews. It includes data preprocessing, feature extraction, model training, evaluation, and visualization. Two different machine learning algorithms are compared: **Logistic Regression** and **Naive Bayes**.

## ✨ Features

- **Data Preprocessing**: Text cleaning, HTML tag removal, stopwords filtering
- **Feature Engineering**: TF-IDF and Count Vectorization
- **Model Comparison**: Logistic Regression vs Naive Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualization**: Word clouds for positive and negative reviews
- **Model Persistence**: Save trained models for future use

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
nltk>=3.6.0
matplotlib>=3.3.0
wordcloud>=1.8.0
streamlit>=1.28.0
```

## 🚀 Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd sentiment-analysis-imdb
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib wordcloud streamlit
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 📊 Dataset

- **Source**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Format**: CSV file named `IMDB Dataset.csv`
- **Size**: 50,000 movie reviews
- **Labels**: Binary classification (positive/negative)
- **Balance**: Equally distributed (25,000 positive, 25,000 negative)

### Dataset Structure:
| Column | Description |
|--------|-------------|
| review | Raw movie review text |
| sentiment | Label (positive/negative) |

## 🎮 Usage

### Option 1: Training Models with Jupyter Notebook

#### Step 1: Prepare Your Environment
```bash
pip install -r requirements.txt
```

#### Step 2: Download the Dataset
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Download `IMDB Dataset.csv`
3. Place it in your project directory

#### Step 3: Run the Jupyter Notebook
```bash
jupyter notebook sentiment_analysis.ipynb
```

#### Step 4: Execute Cells in Order
The notebook is structured in sequential steps:

1. **Load and Preview Dataset** - Import data and explore structure
2. **Explore the Dataset** - Check dimensions, missing values, class distribution
3. **Text Cleaning & Preprocessing** - Clean and prepare text data
4. **Text Vectorization** - Convert text to numerical features
5. **Encode Target Labels** - Convert sentiment labels to numeric format
6. **Split Dataset** - Create train/test splits
7. **Train Logistic Regression** - Train and evaluate first model
8. **Install WordCloud** - Set up visualization library
9. **Visualize Word Clouds** - Generate word frequency visualizations
10. **Train Naive Bayes** - Train and evaluate second model
11. **Model Comparison** - Compare both algorithms

### Option 2: Running the Streamlit Web App

Once you have trained your models and saved them as pickle files, you can use the interactive web application.

#### Step 1: Install Streamlit
```bash
pip install streamlit
```

#### Step 2: Run the Streamlit App
```bash
streamlit run app.py
```

#### Step 3: Use the Web Interface
1. Open your browser to `http://localhost:8501`
2. Enter a movie review in the text area
3. Click "Analyze Review" to get the sentiment prediction
4. The app will classify the review as either **Positive 😃** or **Negative 😞**

### Example Usage:
- **Positive Review**: "This movie was absolutely amazing! Great acting and fantastic plot."
- **Negative Review**: "Terrible movie. Boring plot and bad acting."

## 🏗️ Project Structure

```
sentiment-analysis-imdb/
│
├── sentiment_analysis.ipynb    # Main Jupyter notebook
├── app.py                     # Streamlit web application
├── IMDB Dataset.csv           # Dataset file (download separately)
├── model.pkl                  # Saved Logistic Regression model
├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 📈 Model Performance

### Logistic Regression (TF-IDF Features)
- **Accuracy**: 88.68%
- **Precision**: 0.88-0.90
- **Recall**: 0.87-0.90
- **F1-Score**: 0.88-0.89

### Naive Bayes (Count Vectorization)
- **Accuracy**: 84.48%
- **Precision**: 0.84-0.85
- **Recall**: 0.84
- **F1-Score**: 0.84

### Key Insights:
- Logistic Regression outperforms Naive Bayes on this dataset
- Both models achieve good performance for sentiment classification
- Logistic Regression provides more balanced precision and recall

## 💾 Saved Models

The notebook automatically saves the trained models for future use:

```python
import pickle

# Save the Logistic Regression model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
```

### Loading Saved Models:
```python
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

# Make predictions on new text
new_review = ["This movie was amazing!"]
vectorized_review = loaded_vectorizer.transform(new_review)
prediction = loaded_model.predict(vectorized_review)
```

## 🌐 Streamlit Web Application

The project includes a user-friendly web interface built with Streamlit (`app.py`):

### Features:
- **Interactive text input** for movie reviews
- **Real-time sentiment prediction**
- **Clean, responsive UI** with emojis
- **GitHub repository link**

### App.py Code Structure:
```python
import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter your review, and the app will predict whether it is positive or negative:")

# Input text
user_input = st.text_area("Review:")

if st.button("Analyze Review"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vect_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vect_input)[0]
        sentiment = "Positive 😃" if prediction == 1 else "Negative 😞"
        st.success(f"Classification: {sentiment}")
    else:
        st.error("Please enter a review")
```

### Running the Streamlit App:
1. **Prerequisites**: Ensure you have `model.pkl` and `vectorizer.pkl` files
2. **Install Streamlit**: `pip install streamlit`  
3. **Run the app**: `streamlit run app.py`
4. **Access the app**: Open `http://localhost:8501` in your browser

## 📊 Visualization

The project includes word cloud visualizations that show:
- **Most frequent words in positive reviews**
- **Most frequent words in negative reviews**

These visualizations help understand what words are most associated with each sentiment class.

## 🔧 Customization Options

### Modify Feature Extraction:
```python
# Adjust TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increase feature count
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,            # Minimum document frequency
    max_df=0.95          # Maximum document frequency
)
```

### Try Different Models:
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
```

## 🛠️ Troubleshooting

### Common Issues:

1. **NLTK Download Error**:
   ```python
   import nltk
   nltk.download('stopwords', quiet=True)
   ```

2. **Memory Issues with Large Datasets**:
   - Reduce `max_features` in vectorizers
   - Use smaller data samples for initial testing

3. **Missing Dataset**:
   - Ensure `IMDB Dataset.csv` is in the correct directory
   - Check file permissions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [WordCloud Documentation](https://amueller.github.io/word_cloud/)

---

**Note**: Make sure to download the IMDB dataset from Kaggle before running the notebook. The dataset is required for the sentiment analysis to work properly.
