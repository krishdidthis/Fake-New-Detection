import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('stopwords')

# ---- Data Load ----
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[["text", "label"]]

# ---- Text Cleaning ----
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df["clean_text"] = df["text"].apply(clean_text)

# ---- Train/Test Split ----
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- TF-IDF Vectorization ----
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- Model Training ----
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ---- Evaluation ----
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- Live Prediction Test ----
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = max(prob) * 100
    return label, confidence

# Test karte hain

sample_news =  "The Indian government announced new measures to improve digital infrastructure across rural areas."
label, confidence = predict_news(sample_news)
print(f"\nNews: {sample_news}")
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")

'''sample_news_2 = "Reuters reported that the US Senate voted on the new tax reform bill on Monday."
label2, confidence2 = predict_news(sample_news_2)
print(f"\nNews: {sample_news_2}")
print(f"Prediction: {label2} (Confidence: {confidence2:.2f}%)")

sample_news_3 = "You won't believe what Obama secretly did behind everyone's back, shocking video leaked!"
label3, confidence3 = predict_news(sample_news_3)
print(f"\nNews: {sample_news_3}")
print(f"Prediction: {label3} (Confidence: {confidence3:.2f}%)")
'''

#"A secret government document claims that the Moon will disappear from the sky for three days next month."
#"A viral post claims that placing a smartphone under a pillow can charge its battery without electricity."
#"The Indian government announced new measures to improve digital infrastructure across rural areas."
#"The Reserve Bank of India kept its key interest rate unchanged during its latest monetary policy meeting."
#"NASA successfully launched a research satellite to study changes in Earth's atmosphere."


import pickle
import os

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("Model saved successfully!")