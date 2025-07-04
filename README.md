# 📰 Fake vs Real News Detection

This project is a machine learning and natural language processing (NLP) pipeline to classify news articles as **Real** or **Fake**. It uses the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and is implemented in a Jupyter Notebook.


## 🧠 Objective

To build a classifier that can determine whether a given news article is **fake** or **real**, based on its title and text content.

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK / spaCy (optional for preprocessing)
- TfidfVectorizer / CountVectorizer
- Logistic Regression / Naive Bayes / Random Forest / XGBoost
- Jupyter Notebook

## 📊 Workflow

1. **Data Loading & Merging**
   - Load `Fake.csv` and `True.csv`
   - Add labels and concatenate into one DataFrame

2. **Data Preprocessing**
   - Lowercasing, removing punctuation and stopwords
   - Tokenization
   - Vectorization using TF-IDF

3. **Model Training**
   - Split into training and testing sets
   - Train multiple models (e.g., Logistic Regression, Naive Bayes)
   - Evaluate using accuracy, precision, recall, and F1-score

4. **Model Evaluation**
   - Confusion matrix
   - ROC-AUC score
   - Cross-validation

## ✅ Results

- The best-performing model achieves an accuracy of **XX%**.
- ROC-AUC score: **YY**

## 🔮 Future Improvements

- Use advanced NLP models like BERT
- Deploy the model as a Flask app or using Streamlit
- Build an API for real-time predictions

## 📌 Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
