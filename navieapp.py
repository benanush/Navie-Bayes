import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Spam Classification using Naive Bayes")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\benan\Documents\Data_Scientist\Streamlit\sms_spam_small.csv")

try:
    df = load_data()
    st.success("Dataset loaded successfully!")
except:
    st.error("Dataset not found! Place sms_spam_small.csv in the same folder.")
    st.stop()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")
st.write(df.head())

# -----------------------------
# Spam vs Ham Distribution
# -----------------------------
st.subheader("📈 Spam vs Ham Distribution")

label_counts = df['label'].value_counts()

fig1 = plt.figure()
plt.bar(label_counts.index, label_counts.values)
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.title("Distribution of Spam and Ham Messages")
st.pyplot(fig1)

# -----------------------------
# Encode Labels
# -----------------------------
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# -----------------------------
# Message Length Analysis
# -----------------------------
df['message_length'] = df['message'].apply(len)

st.subheader("📊 Message Length Distribution")

fig2 = plt.figure()
plt.hist(df[df['label'] == 0]['message_length'], bins=20, alpha=0.7, label="Ham")
plt.hist(df[df['label'] == 1]['message_length'], bins=20, alpha=0.7, label="Spam")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.title("Message Length Distribution")
plt.legend()
st.pyplot(fig2)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: **{accuracy:.2f}**")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig3 = plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
st.pyplot(fig3)

# -----------------------------
# User Prediction
# -----------------------------
st.subheader("🔮 Spam Detection")

user_message = st.text_area("Enter a message")

if st.button("Predict"):
    if user_message.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vector = vectorizer.transform([user_message])
        prediction = model.predict(msg_vector)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM")
        else:
            st.success("✅ This message is NOT SPAM (HAM)")



