import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Airline Sentiment Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "sentiment_model.joblib"
DATA_PATH = Path(__file__).parent / "Tweets.csv"

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(circle at 8% 0%, rgba(99,102,241,.12), transparent 27%),
          radial-gradient(circle at 95% 5%, rgba(14,165,233,.10), transparent 25%),
          #f6f8fc;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #17233d; }
    .hero {
        padding: 30px 32px;
        border-radius: 24px;
        color: white;
        background: linear-gradient(120deg, #172554 0%, #3730a3 52%, #0369a1 100%);
        box-shadow: 0 18px 45px rgba(30, 41, 99, .18);
        margin-bottom: 24px;
    }
    .hero h1 { color: white !important; font-size: 2.35rem; margin: 0 0 8px 0; }
    .hero p { color: #dbeafe; font-size: 1.02rem; margin: 0; }
    .eyebrow {
        color: #bfdbfe; text-transform: uppercase; letter-spacing: .16em;
        font-size: .72rem; font-weight: 700; margin-bottom: 10px;
    }
    .panel {
        background: rgba(255,255,255,.92);
        border: 1px solid #e6eaf2;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 8px 24px rgba(15,23,42,.045);
    }
    .muted { color: #64748b; font-size: .92rem; }
    .sentiment {
        display: inline-block; padding: 7px 13px; border-radius: 999px;
        font-size: .82rem; font-weight: 700; letter-spacing: .04em;
    }
    .positive { background: #dcfce7; color: #166534; }
    .neutral { background: #e2e8f0; color: #334155; }
    .negative { background: #fee2e2; color: #991b1b; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf2; padding: 16px;
        border-radius: 16px; box-shadow: 0 6px 18px rgba(15,23,42,.04);
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 12px; font-weight: 700; min-height: 2.8rem;
    }
    section[data-testid="stSidebar"] { background: #eef2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Text cleaning: keeps the app independent from NLTK downloads
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Simple fallback so the page can still open if the model file is missing.
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "thank",
    "thanks", "helpful", "friendly", "wonderful", "best", "nice", "happy",
    "appreciate", "smooth", "comfortable", "quick", "easy", "perfect"
}
NEGATIVE_WORDS = {
    "bad", "worst", "poor", "late", "delay", "delayed", "cancel", "cancelled",
    "lost", "rude", "angry", "hate", "terrible", "awful", "broken", "problem",
    "issue", "missed", "disappointed", "unhelpful"
}

def fallback_predict(text):
    words = clean_text(text).split()
    pos = sum(word in POSITIVE_WORDS for word in words)
    neg = sum(word in NEGATIVE_WORDS for word in words)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def predict_sentiment(text, model):
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", cleaned, "The text is empty after cleaning."
    if model is not None:
        try:
            prediction = str(model.predict([cleaned])[0]).lower()
            if prediction not in {"positive", "neutral", "negative"}:
                prediction = prediction.strip()
            return prediction, cleaned, "TF-IDF + Logistic Regression model"
        except Exception as exc:
            return fallback_predict(text), cleaned, f"Fallback used because model prediction failed: {exc}"
    return fallback_predict(text), cleaned, "Simple word-list fallback (model file not found)"


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ Sentiment Studio")
    st.caption("Airline Tweet Analysis")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Sentiment Analyzer", "Dataset Insights", "About Project"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Model status**")
    try:
        model = load_model()
        if model is not None:
            st.success("Trained model loaded")
        else:
            st.warning("Model file not found — fallback mode")
    except Exception as model_error:
        model = None
        st.error("Model could not be loaded")
        st.caption(str(model_error)[:220])
    st.caption("Expected file: sentiment_model.joblib")
    st.caption("Built with Python · NLP · Streamlit")

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NATURAL LANGUAGE PROCESSING · PROJECT PR-1</div>
      <h1>Airline Sentiment Studio</h1>
      <p>Turn airline customer tweets into clear sentiment insights — positive, neutral, or negative.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Page 1: Analyzer
# ---------------------------------------------------------
if page == "Sentiment Analyzer":
    st.markdown("### Explore a customer tweet")
    st.markdown('<p class="muted">Enter a tweet below and let the NLP model classify its sentiment.</p>', unsafe_allow_html=True)

    col_input, col_examples = st.columns([1.55, 1], gap="large")
    with col_input:
        tweet = st.text_area(
            "Your tweet",
            placeholder="Example: My flight was delayed and the customer service was terrible...",
            height=155,
            max_chars=5000,
        )
        analyze = st.button("✨  Analyze sentiment", type="primary", use_container_width=True)
    with col_examples:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Try an example**")
        examples = {
            "😊 Positive": "The crew was amazing and the flight was very comfortable. Thank you!",
            "😐 Neutral": "My flight from Delhi to Mumbai is scheduled for 6 PM.",
            "😞 Negative": "My flight was delayed for hours and the staff was very rude.",
        }
        for label, example in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state["example_tweet"] = example
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if "example_tweet" in st.session_state and not tweet:
        tweet = st.session_state["example_tweet"]

    if analyze:
        if not tweet.strip():
            st.warning("Please enter a tweet before analyzing.")
        else:
            with st.spinner("Analyzing tweet..."):
                label, cleaned, method = predict_sentiment(tweet, model)
            st.markdown("---")
            st.markdown("### Analysis result")
            css_label = label if label in {"positive", "neutral", "negative"} else "neutral"
            emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(css_label, "🔎")
            result_col, clean_col = st.columns([1, 1], gap="large")
            with result_col:
                st.markdown(
                    f'<div class="panel"><div class="muted">PREDICTED SENTIMENT</div>'
                    f'<h2 style="margin:10px 0 14px 0;">{emoji} {label.title()}</h2>'
                    f'<span class="sentiment {css_label}">{label.upper()}</span>'
                    f'<p class="muted" style="margin-top:16px;">Method: {method}</p></div>',
                    unsafe_allow_html=True,
                )
            with clean_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Cleaned text**")
                st.write(cleaned if cleaned else "No words left after cleaning.")
                st.markdown("</div>", unsafe_allow_html=True)
            st.info("This is a text classification prediction, not a guarantee of the customer's actual intention.")

    st.markdown("---")
    st.markdown("### How it works")
    flow_cols = st.columns(4)
    steps = [
        ("01", "Input", "Customer tweet"),
        ("02", "Cleaning", "URLs, mentions, symbols"),
        ("03", "TF-IDF", "Text to numeric features"),
        ("04", "Prediction", "3 sentiment classes"),
    ]
    for col, (number, title, desc) in zip(flow_cols, steps):
        with col:
            st.markdown(
                f'<div class="panel"><div class="muted">{number}</div>'
                f'<h3 style="margin:6px 0;">{title}</h3><div class="muted">{desc}</div></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# Page 2: Dataset insights
# ---------------------------------------------------------
elif page == "Dataset Insights":
    st.markdown("### Dataset overview")
    st.markdown(
        '<p class="muted">Optional visual exploration of Tweets.csv. Add the dataset beside app.py to enable these charts.</p>',
        unsafe_allow_html=True,
    )
    if DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)
            required = {"text", "airline_sentiment"}
            if not required.issubset(data.columns):
                st.error("Tweets.csv must contain text and airline_sentiment columns.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total tweets", f"{len(data):,}")
                c2.metric("Airlines", str(data["airline"].nunique()) if "airline" in data else "N/A")
                c3.metric("Sentiment classes", str(data["airline_sentiment"].nunique()))
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown("**Sentiment distribution**")
                    counts = data["airline_sentiment"].value_counts().rename_axis("Sentiment").to_frame("Tweets")
                    st.bar_chart(counts)
                with right:
                    if "airline" in data.columns:
                        st.markdown("**Tweets by airline**")
                        airline_counts = data["airline"].value_counts().to_frame("Tweets")
                        st.bar_chart(airline_counts)
                st.markdown("**Sample records**")
                st.dataframe(data[["airline_sentiment", "airline", "text"]].head(10), use_container_width=True)
        except Exception as data_error:
            st.error(f"Could not read Tweets.csv: {data_error}")
    else:
        st.info("Dataset file is optional. Upload Tweets.csv to the same GitHub folder as app.py to display dataset charts.")
        st.markdown("**Required columns:** `text`, `airline_sentiment` (and optionally `airline`).")

# ---------------------------------------------------------
# Page 3: About
# ---------------------------------------------------------
else:
    st.markdown("### About this project")
    st.write(
        "This project analyzes US airline customer tweets using Natural Language Processing. "
        "The goal is to classify each tweet into one of three sentiment categories: positive, neutral, or negative."
    )
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**NLP workflow**")
        st.markdown(
            "- Dataset exploration and visualization\n"
            "- Text cleaning with regular expressions\n"
            "- Tokenization and stop-word handling\n"
            "- Bag-of-Words and TF-IDF features\n"
            "- Word2Vec embeddings\n"
            "- TF-IDF + Logistic Regression classification\n"
            "- POS tagging and analysis"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Technology stack**")
        st.markdown(
            "- Python\n"
            "- Pandas\n"
            "- Scikit-learn\n"
            "- NLTK (not required by this app interface)\n"
            "- Joblib\n"
            "- Streamlit"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Model file: sentiment_model.joblib — exported from Task 6.7 of the project notebook.")

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:.82rem;">'
    'AIRLINE SENTIMENT STUDIO · NLP PR-1 · Built with Streamlit</div>',
    unsafe_allow_html=True,
)
