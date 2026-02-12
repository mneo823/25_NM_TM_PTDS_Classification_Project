import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Generating model comparison visualizations...\n")

# Load models
with open('pickled_files/all_models.pkl', 'rb') as f:
    data = pickle.load(f)

models = data['models']
vectorizer = data['vectorizer']
results = data['results']

# Load test data
test = pd.read_csv("Data/processed/test.csv")
test.dropna(inplace=True)
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

test['text'] = test['text'].apply(clean_text)
X_test = vectorizer.transform(test['text'])
y_test = test['category']

# Create comparison visualizations
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy comparison
ax1 = axes[0, 0]
comparison = pd.DataFrame(results).T
train_acc = comparison['train_accuracy']
test_acc = comparison['test_accuracy']

x = range(len(comparison))
width = 0.35
ax1.bar([i - width/2 for i in x], train_acc, width, label='Train', alpha=0.8)
ax1.bar([i + width/2 for i in x], test_acc, width, label='Test', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(comparison.index, rotation=45, ha='right')
ax1.set_ylabel('Accuracy')
ax1.set_title('Train vs Test Accuracy', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: All metrics
ax2 = axes[0, 1]
metrics = comparison[['test_accuracy', 'f1_score', 'precision', 'recall']].T
metrics.plot(kind='bar', ax=ax2, alpha=0.8)
ax2.set_xticklabels(['Accuracy', 'F1', 'Precision', 'Recall'], rotation=0)
ax2.set_title('Performance Metrics', fontweight='bold')
ax2.legend(title='Models', bbox_to_anchor=(1.05, 1))
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Training time
ax3 = axes[1, 0]
times = comparison['training_time'].sort_values()
ax3.barh(range(len(times)), times, alpha=0.8, color='coral')
ax3.set_yticks(range(len(times)))
ax3.set_yticklabels(times.index)
ax3.set_xlabel('Time (seconds)')
ax3.set_title('Training Time', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Model ranking
ax4 = axes[1, 1]
ranking = comparison[['test_accuracy', 'f1_score']].sort_values('test_accuracy', ascending=True)
ranking.plot(kind='barh', ax=ax4, alpha=0.8)
ax4.set_title('Model Ranking', fontweight='bold')
ax4.legend(['Accuracy', 'F1 Score'])
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_visualizations.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_visualizations.png")

# Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

categories = sorted(y_test.unique())

for idx, (name, model) in enumerate(models.items()):
    if idx < 6:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=categories)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=categories, yticklabels=categories)
        axes[idx].set_title(f'{name}\nAcc: {results[name]["test_accuracy"]:.4f}', fontweight='bold')
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')

for idx in range(len(models), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices_all_models.png")

# Print results
print("\nModel Performance Summary:")
print("="*80)
display_df = comparison[['test_accuracy', 'f1_score', 'precision', 'recall', 'training_time']]
display_df.columns = ['Accuracy', 'F1', 'Precision', 'Recall', 'Time (s)']
print(display_df.to_string())

print("\nBest Models:")
print(f"  Highest accuracy: {comparison['test_accuracy'].idxmax()}")
print(f"  Fastest: {comparison['training_time'].idxmin()}")
print(f"  Best F1: {comparison['f1_score'].idxmax()}")
