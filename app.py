import streamlit as st
import pandas as pd
import nltk
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from nlp_utils import clean_text, heuristic_sentiment

st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #f7f9fc 0%, #eef3fb 100%); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
.hero {
    padding: 30px 34px; border-radius: 24px;
    background: linear-gradient(115deg, #111827 0%, #23395d 58%, #315b83 100%);
    color: white; margin-bottom: 22px;
    box-shadow: 0 14px 35px rgba(24, 43, 71, .16);
}
.hero h1 { font-family: 'Manrope', sans-serif; font-size: 2.25rem; margin: 5px 0 8px; color: white; }
.hero p { color: #d8e5f5; font-size: 1rem; margin: 0; }
.eyebrow { color: #9ec5f5; font-size: .76rem; font-weight: 700; letter-spacing: 2px; }
.metric-card {
    background: white; border: 1px solid #e4eaf2; border-radius: 18px;
    padding: 17px 19px; box-shadow: 0 5px 18px rgba(35, 57, 93, .05);
}
.metric-label { color: #718096; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; }
.metric-value { color: #172b4d; font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 800; margin-top: 5px; }
.section-title { color: #172b4d; font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800; margin: 12px 0; }
div.stButton > button {
    border-radius: 12px; border: 1px solid #d9e2ef; background: white;
    color: #23395d; font-weight: 700; min-height: 42px;
}
div.stButton > button:hover { border-color: #4778ad; color: #1d4f85; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #245b91, #397db5);
    border: 0; color: white; border-radius: 13px; min-height: 48px;
}
div[data-testid="stExpander"] { background: white; border: 1px solid #e4eaf2; border-radius: 14px; }
div[data-testid="stTextArea"] textarea { border-radius: 14px; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- NLTK resources ----------
@st.cache_resource
def setup_nltk():
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords")
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load("sentiment_model.joblib")
    except Exception:
        return None

sia = setup_nltk()
model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ✈️ Project Guide")
    st.caption("NLP Practical Report 1")
    st.markdown("---")
    st.markdown("**About this project**")
    st.write("An airline tweet sentiment analyzer that classifies a message as positive, negative or neutral.")
    st.markdown("**Dataset**")
    st.write("Twitter US Airline Sentiment")
    st.markdown("[Open dataset on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)")
    st.markdown("---")
    st.markdown("**Methods included**")
    st.write("• Heuristic rules\n\n• NLTK VADER\n\n• Machine-learning pipeline")
    st.caption("Built with Python + Streamlit")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">NATURAL LANGUAGE PROCESSING · PR 1</div>
  <h1>Airline Tweet Sentiment Analyzer ✈️</h1>
  <p>Understand how passengers feel. Enter a tweet, choose an NLP approach, and explore the prediction.</p>
</div>
""", unsafe_allow_html=True)

# Summary cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Task</div><div class="metric-value">Sentiment Analysis</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Output classes</div><div class="metric-value">3 sentiments</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">NLP approaches</div><div class="metric-value">3 methods</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="section-title">💬 Enter a passenger tweet</div>', unsafe_allow_html=True)

    examples = {
        "😊 Positive example": "The crew was amazing and the flight was comfortable. Thank you!",
        "😡 Negative example": "My flight was delayed again and the customer service was terrible.",
        "😐 Negation example": "I am not happy with the delay, but the staff was helpful."
    }
    bcols = st.columns(3)
    for col, (label, example) in zip(bcols, examples.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["tweet_input"] = example

    tweet = st.text_area(
        "Tweet text",
        key="tweet_input",
        height=155,
        placeholder="Example: My flight was delayed and the service was poor..."
    )

    approach = st.selectbox(
        "Choose an NLP approach",
        ["Heuristic", "VADER", "ML Model"],
        help="Compare a simple rule-based method, a ready-made sentiment analyzer, and your trained model."
    )

    predict = st.button("✨  Analyze sentiment", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="section-title">🧭 How it works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
      <b>01 · Input</b><br><span style="color:#718096">Type or select a sample tweet.</span><br><br>
      <b>02 · Cleaning</b><br><span style="color:#718096">Remove URLs, mentions and extra symbols.</span><br><br>
      <b>03 · Prediction</b><br><span style="color:#718096">Run the selected sentiment approach.</span><br><br>
      <b>04 · Insights</b><br><span style="color:#718096">View the result, probability chart and POS tags.</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if model is None:
        st.info("ML model file not found. Heuristic and VADER are ready. Run the notebook's save-model cell and place sentiment_model.joblib beside app.py to enable ML Model.")
    else:
        st.success("Trained ML model loaded successfully.")

# ---------- Prediction ----------
if predict:
    if not tweet.strip():
        st.warning("Please enter a tweet before clicking Analyze.")
    else:
        original = tweet.strip()
        if len(original) > 500:
            original = original[:500]
            st.warning("Your tweet was longer than 500 characters, so only the first 500 characters were analyzed.")

        cleaned = clean_text(original)
        if not cleaned:
            st.warning("The tweet became empty after cleaning. Please enter words rather than only emojis, symbols or a URL.")
        else:
            label = None
            probabilities = None
            explanation = ""

            if approach == "Heuristic":
                label, pos_score, neg_score = heuristic_sentiment(original)
                total = pos_score + neg_score
                if total:
                    probabilities = {
                        "negative": neg_score / total,
                        "neutral": 0.0,
                        "positive": pos_score / total
                    }
                explanation = f"Positive word score: {pos_score} · Negative word score: {neg_score}"

            elif approach == "VADER":
                if sia is None:
                    st.error("VADER could not be loaded. Please check your internet connection and restart the app.")
                else:
                    score = sia.polarity_scores(original)["compound"]
                    if score >= 0.05:
                        label = "positive"
                    elif score <= -0.05:
                        label = "negative"
                    else:
                        label = "neutral"
                    explanation = f"VADER compound score: {score:.3f}"
                    probabilities = {
                        "negative": max(0.0, -score),
                        "neutral": 1.0 - abs(score),
                        "positive": max(0.0, score)
                    }

            else:
                if model is None:
                    st.error("ML model is not available. Save sentiment_model.joblib from your notebook and place it in this project folder.")
                else:
                    try:
                        label = str(model.predict([cleaned])[0]).lower()
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba([cleaned])[0]
                            probabilities = {str(cls): float(p) for cls, p in zip(model.classes_, probs)}
                        explanation = "Prediction from your saved scikit-learn pipeline."
                    except Exception as e:
                        st.error(f"Could not predict with the saved model: {e}")

            if label:
                st.markdown("---")
                st.markdown('<div class="section-title">📊 Analysis result</div>', unsafe_allow_html=True)
                result_col, detail_col = st.columns([1, 1.2])
                with result_col:
                    if label == "positive":
                        st.success("😊  POSITIVE")
                    elif label == "negative":
                        st.error("😟  NEGATIVE")
                    else:
                        st.warning("😐  NEUTRAL")
                    st.caption(explanation)
                with detail_col:
                    st.markdown("**Cleaned text**")
                    st.write(cleaned)

                if probabilities:
                    chart_data = pd.DataFrame({
                        "Sentiment": list(probabilities.keys()),
                        "Score": list(probabilities.values())
                    }).set_index("Sentiment")
                    st.markdown("**Sentiment score / class probabilities**")
                    st.bar_chart(chart_data, height=210)

                with st.expander("🔎 Explore Part-of-Speech (POS) tags"):
                    try:
                        tokens = nltk.word_tokenize(cleaned)
                        tags = nltk.pos_tag(tokens)
                        pos_df = pd.DataFrame(tags, columns=["Word", "POS Tag"])
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                    except LookupError:
                        st.warning("NLTK POS tagger data is missing. Restart the app after the resource download.")
                    except Exception as e:
                        st.warning(f"POS tagging is not available: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#718096;font-size:.82rem'>"
    "NLP Practical Report 1 · Airline Tweet Sentiment Analyzer<br>"
    "Made with Python, NLTK, scikit-learn and Streamlit"
    "</div>",
    unsafe_allow_html=True
)
