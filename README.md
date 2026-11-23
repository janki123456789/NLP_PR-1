<div align="center">

# ✈️ Airline Tweet Sentiment Analysis
### NLP • Text Preprocessing • Machine Learning • Streamlit

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Text%20Analytics-8A2BE2?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

**Turning airline customer tweets into meaningful sentiment insights.**

<p>
  <a href="PASTE_YOUR_STREAMLIT_APP_URL_HERE">
    <img src="https://img.shields.io/badge/🚀_Open_Live_Streamlit_App-Click_Here-FF4B4B?style=for-the-badge" alt="Open Streamlit App">
  </a>
</p>

</div>

---

## 🌟 Project Overview

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to airline-related tweets. It processes customer text and identifies the sentiment expressed in each tweet:

- 🟢 **Positive** — customer expresses satisfaction.
- 🟡 **Neutral** — customer expresses a neutral or informational opinion.
- 🔴 **Negative** — customer expresses dissatisfaction or a complaint.

The project follows an end-to-end NLP workflow, from exploring raw tweets and cleaning text to extracting features, training classification models, evaluating results, and using a trained model in a Streamlit web application.

## 🎯 Objectives

- Explore and understand real-world airline customer tweets.
- Clean text using Regular Expressions (RegEx).
- Perform tokenisation and stop-word handling while preserving negation words.
- Explore Bag of Words (BoW), TF-IDF, and Word2Vec representations.
- Compare heuristic, VADER, and machine-learning sentiment approaches.
- Train and evaluate text-classification models.
- Perform Part-of-Speech (POS) tagging.
- Build an interactive Streamlit application for sentiment prediction.

## 🧰 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas, NumPy | Data handling and numerical operations |
| Matplotlib | Data visualisation |
| Regular Expressions (`re`) | Text cleaning |
| NLTK | Tokenisation, stop words, VADER and POS tagging |
| Scikit-learn | Feature extraction, model training and evaluation |
| Gensim | Word2Vec embeddings |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |


## 🔗 Important Links

| Resource | Link |
|---|---|
| 📊 Dataset (Kaggle – Twitter US Airline Sentiment) | https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment |
| 🚀 Deployed Streamlit link | STREAMLIT_APP_URL |

## 🔄 NLP Workflow

```text
Raw Airline Tweets
        │
        ▼
Data Loading & Exploration
        │
        ▼
Text Cleaning with RegEx
        │
        ▼
Tokenisation & Stop-word Handling
        │
        ▼
Feature Extraction
  ┌─────┼───────────┐
  ▼     ▼           ▼
 BoW  TF-IDF     Word2Vec
  └─────┼───────────┘
        ▼
Sentiment Approaches
  ┌─────┼───────────┐
  ▼     ▼           ▼
Heuristic VADER  ML Models
        │
        ▼
Model Evaluation
        │
        ▼
Saved Sentiment Pipeline
        │
        ▼
Streamlit Web Application
        │
        ▼
Positive / Neutral / Negative
```

## 📘 Practical Tasks Covered

| Task | Description |
|---|---|
| Task 1 | NLP fundamentals, pipeline and dataset exploration |
| Task 2 | Text cleaning with RegEx and tokenisation |
| Task 3 | Heuristic and VADER sentiment approaches |
| Task 4 | Bag of Words and TF-IDF feature extraction |
| Task 5 | Word2Vec embeddings and document vectors |
| Task 6 | Text classification pipeline, model comparison, evaluation and model saving |
| Task 7 | Part-of-Speech (POS) tagging and POS-based features |
| Task 8 | Streamlit application development and deployment |


## 📁 Project Structure

```text
Airline-Tweet-Sentiment-Analysis/
│
├── PR-1.ipynb                 # NLP practical notebook (Tasks 1–7)
├── Tweets.csv                 # Airline tweet dataset
├── app.py                     # Streamlit application
├── nlp_utils.py               # Text cleaning and helper functions
├── requirements.txt           # Project dependencies
├── sentiment_model.joblib     # Trained ML pipeline (generated from notebook)
└── README.md                  # Project documentation
```

*Use the actual filenames in your GitHub repository if they differ from the names shown above.*

## 📊 Model Evaluation

The notebook compares multiple approaches using classification metrics, including:

- Accuracy
- Macro F1-score
- Classification report
- Confusion matrix
- Cross-validation
- Error analysis

The best model should be selected based on the evaluation results produced by running the notebook. The README intentionally does not claim a specific accuracy or best-performing model because those values depend on the actual notebook execution.


## ⚠️ Limitations

- Sarcasm, irony and mixed opinions can be difficult to classify.
- Informal language, spelling mistakes and short tweets may affect predictions.
- VADER and simple word-based rules may not fully understand airline-specific context.
- Model predictions depend on the quality and balance of the training data.

## 🔮 Future Improvements

- Experiment with advanced models such as LSTM, BERT or other transformer-based models.
- Improve handling of sarcasm, emojis and informal language.
- Add more airline and customer-review datasets.
- Improve model monitoring and prediction explanations.
- Add multilingual sentiment analysis.

## 🎥 Video

(VIDEO_URL) 

## 👩‍💻 Author

**Janki Dholariya**  
AI-ML / Data Science Student

---