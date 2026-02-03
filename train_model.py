import pandas as pd
import re
import nltk
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Quick NLTK setup
for pkg in ['punkt', 'wordnet', 'stopwords', 'omw-1.4', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

print("Loading data...")
train = pd.read_csv("Data/processed/train.csv")
test = pd.read_csv("Data/processed/test.csv")

train.dropna(inplace=True)
test.dropna(inplace=True)

# Combine text columns
train['text'] = train['headlines'] + ' ' + train['description'] + ' ' + train['content']
test['text'] = test['headlines'] + ' ' + test['description'] + ' ' + test['content']

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("Cleaning text...")
train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])
y_train = train['category']
y_test = test['category']

# Train
print("Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}\n")
print(classification_report(y_test, y_pred))

# Save
os.makedirs('pickled_files', exist_ok=True)
with open('pickled_files/model_and_vectorizer.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("\nModel saved to pickled_files/")
