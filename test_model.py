import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

print("Testing model...")

with open('pickled_files/model_and_vectorizer.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
vectorizer = data['vectorizer']

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

# Test samples
samples = [
    ("Apple announces new iPhone with advanced AI features", "technology"),
    ("Stock market reaches all-time high today", "business"),
    ("Local team wins championship in thrilling match", "sports"),
    ("New blockbuster movie breaks box office records", "entertainment"),
    ("University launches new online learning platform", "education")
]

print("\nTesting predictions:\n")
correct = 0
for text, expected in samples:
    processed = clean_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = max(proba) * 100
    
    is_correct = prediction == expected
    correct += is_correct
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {text[:50]}...")
    print(f"  Predicted: {prediction.upper()} ({confidence:.1f}%)")
    print(f"  Expected: {expected.upper()}\n")

print(f"Results: {correct}/{len(samples)} correct\n")
