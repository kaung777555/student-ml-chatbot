# ==========================================================
# Streamlit ML Chatbot (Single File)
# TF-IDF + Logistic Regression + CSV Dataset Loader
# ==========================================================

import os
import string
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# NLTK setup (stopwords)
# -------------------------
def ensure_stopwords():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

ensure_stopwords()
STOP_WORDS = set(stopwords.words("english"))

# -------------------------
# Preprocess
# -------------------------
def preprocess(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# -------------------------
# CSV dataset (auto-create if missing)
# -------------------------
DEFAULT_CSV = """text,intent
When does admission start?,admission
Admission opening date?,admission
How can I enroll?,admission
How to apply for admission?,admission
What documents are required for admission?,admission
Is admission open now?,admission
When is the deadline for admission?,admission

How much is the fee?,fee
Tell me about fee structure,fee
What is the tuition fee?,fee
Do you have installment plan?,fee
Any scholarship available?,fee
Is there any discount?,fee
What is the registration fee?,fee

What courses are available?,course
List of courses,course
Available programs?,course
Do you offer IT program?,course
Do you have engineering program?,course
Do you offer business management?,course
Is computer science available?,course

Where is the university located?,location
University location?,location
Campus address?,location
How can I reach the campus?,location
Is it near city center?,location
Which township is the university in?,location

Is there a hostel?,facility
Do you have hostel facility?,facility
Do you provide dormitory?,facility
Is library available?,facility
Do you have sports facilities?,facility
Is there a canteen?,facility
"""

CSV_PATH = "dataset.csv"

def load_dataset(csv_path: str) -> pd.DataFrame:
    # Auto-create dataset.csv if not found
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CSV)

    df = pd.read_csv(csv_path)

    # Clean: drop empty rows, ensure required columns exist
    df = df.dropna()
    if not {"text", "intent"}.issubset(df.columns):
        raise ValueError("CSV must have columns: text,intent")

    # Ensure string type
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)

    return df

df = load_dataset(CSV_PATH)
df["clean_text"] = df["text"].apply(preprocess)

# -------------------------
# Train model (cache so it trains once)
# -------------------------
@st.cache_resource
def train_model(dataframe: pd.DataFrame):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(dataframe["clean_text"])
    y = dataframe["intent"]

    # If dataset is small, stratify can fail; handle safely
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return vectorizer, model, acc

vectorizer, model, acc = train_model(df)

# -------------------------
# Responses
# -------------------------
RESPONSES = {
    "admission": "Admission usually starts in June. You can apply online or at the admission office.",
    "fee": "The annual tuition fee is $1500. Scholarships/instalments depend on policy.",
    "course": "We offer programs in IT, Business, and Engineering. Which one do you want?",
    "location": "The university is located in the city center. You can reach by bus/taxi.",
    "facility": "Hostel and campus facilities are available (subject to availability).",
    "major": "Machine Learning / Ai / Cypher security"
}
FALLBACK = "Sorry — I’m not sure yet. Try asking about admission, fees, courses, location, or hostel."

def chatbot_response(user_input: str) -> str:
    cleaned = preprocess(user_input)
    if not cleaned.strip():
        return "Please type a question 😊"
    vec = vectorizer.transform([cleaned])
    intent = model.predict(vec)[0]
    return RESPONSES.get(intent, FALLBACK)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ML Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Student Support ML Chatbot")
st.caption("NLP (TF-IDF) + Logistic Regression — Advanced academic mini project")

with st.expander("Model info / Evaluation"):
    st.write(f"Quick split accuracy: **{acc*100:.2f}%** (demo dataset)")
    st.write("Tip: Edit dataset.csv to add more examples and improve accuracy.")
    st.write(f"Dataset file: **{CSV_PATH}** (auto-created if missing)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about admission, fees, courses, location, or hostel."}
    ]

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
user_text = st.chat_input("Type your message...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    bot_text = chatbot_response(user_text)
    st.session_state.messages.append({"role": "assistant", "content": bot_text})
    with st.chat_message("assistant"):
        st.write(bot_text)

# Sidebar customization
st.sidebar.header("Customize")
st.sidebar.write("Edit responses or add training data in dataset.csv for better results.")
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown(
    "- When does admission start?\n"
    "- How much is the fee?\n"
    "- What courses are available?\n"
    "- Where is the campus?\n"
    "- Do you have hostel?"
)