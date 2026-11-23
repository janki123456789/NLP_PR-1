import re

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "comfortable", "excellent", "fantastic",
    "friendly", "good", "great", "helpful", "love", "nice", "on time",
    "quick", "smooth", "thank", "thanks", "wonderful", "appreciate",
    "professional", "easy", "perfect", "pleasant", "happy", "upgrade",
    "clean", "polite", "resolved", "impressed", "efficient"
}

NEGATIVE_WORDS = {
    "angry", "awful", "bad", "cancel", "cancelled", "canceled", "delay",
    "delayed", "disappointed", "disgusting", "hate", "horrible", "late",
    "lost", "missing", "never", "poor", "problem", "rude", "slow",
    "stuck", "terrible", "unhelpful", "unhappy", "wait", "waiting",
    "worst", "broken", "damaged", "refund", "unacceptable"
}

NEGATIONS = {"no", "not", "nor", "never", "n't", "cannot", "can't"}

def clean_text(text):
    """Clean a tweet while keeping hashtag words and negation words."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def heuristic_sentiment(text):
    """Small rule-based sentiment baseline for the app."""
    words = clean_text(text).split()
    pos_score = 0
    neg_score = 0

    for i, word in enumerate(words):
        negated = any(w in NEGATIONS for w in words[max(0, i - 2):i])
        if word in POSITIVE_WORDS:
            if negated:
                neg_score += 1
            else:
                pos_score += 1
        elif word in NEGATIVE_WORDS:
            if negated:
                pos_score += 1
            else:
                neg_score += 1

    if pos_score > neg_score:
        label = "positive"
    elif neg_score > pos_score:
        label = "negative"
    else:
        label = "neutral"

    return label, pos_score, neg_score
