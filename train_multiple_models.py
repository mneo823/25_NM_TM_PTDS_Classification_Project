import pandas as pd
import re
import nltk
import os
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Training multiple models with hyperparameter tuning...\n")

# Setup NLTK
for pkg in ['punkt', 'wordnet', 'stopwords', 'omw-1.4', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

# Load and clean data
train = pd.read_csv("Data/processed/train.csv")
test = pd.read_csv("Data/processed/test.csv")
train.dropna(inplace=True)
test.dropna(inplace=True)

train['text'] = train['headlines'] + ' ' + train['description'] + ' ' + train['content']
test['text'] = test['headlines'] + ' ' + test['description'] + ' ' + test['content']

# Text processing
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

print("Processing text...")
train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])
y_train = train['category']
y_test = test['category']

# Define models and parameters for tuning
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'C': [0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs', 'saga']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': [100, 200], 'max_depth': [20, 30, None], 'min_samples_split': [2, 5]}
    },
    'Support Vector Machine': {
        'model': LinearSVC(random_state=42, max_iter=2000),
        'params': {'C': [0.1, 1.0, 10.0], 'loss': ['hinge', 'squared_hinge']}
    },
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {'alpha': [0.1, 0.5, 1.0]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    }
}

# Train each model
results = {}
trained_models = {}

for name, config in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    
    grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    trained_models[name] = best_model
    
    y_pred = best_model.predict(X_test)
    
    results[name] = {
        'best_params': grid.best_params_,
        'train_accuracy': best_model.score(X_train, y_train),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'training_time': time.time() - start,
        'cv_score': grid.best_score_
    }
    
    print(f"Best params: {grid.best_params_}")
    print(f"Test accuracy: {results[name]['test_accuracy']:.4f}")
    print(f"Training time: {results[name]['training_time']:.2f}s")

# Show results
print("\n" + "="*80)
print("Model Comparison Results")
print("="*80)

comparison = pd.DataFrame(results).T.sort_values('test_accuracy', ascending=False)
print("\n", comparison[['test_accuracy', 'f1_score', 'training_time']].to_string())

best_name = comparison['test_accuracy'].idxmax()
print(f"\nBest model: {best_name} ({results[best_name]['test_accuracy']:.4f})")

# Save models
os.makedirs('pickled_files', exist_ok=True)

with open('pickled_files/best_model_and_vectorizer.pkl', 'wb') as f:
    pickle.dump({'model': trained_models[best_name], 'vectorizer': vectorizer, 'model_name': best_name}, f)

with open('pickled_files/all_models.pkl', 'wb') as f:
    pickle.dump({'models': trained_models, 'vectorizer': vectorizer, 'results': results}, f)

comparison.to_csv('model_comparison_results.csv')

print("\nModels saved to pickled_files/")
print("Results saved to model_comparison_results.csv")
