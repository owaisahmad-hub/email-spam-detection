# email_spam_detection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# 2. Keep only the relevant columns
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# 3. Map labels to binary (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 5. Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train model (Linear SVM)
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 7. Predictions
y_pred = model.predict(X_test_vec)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("\nModel and vectorizer saved as spam_model.joblib & vectorizer.joblib")

# 10. Function for predicting new messages
def predict_message(message):
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"

# Example predictions
print("\nExamples:")
print(predict_message("Congratulations! You've won a free ticket!"))
print(predict_message("Hey, are we meeting tomorrow?"))
 