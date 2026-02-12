import streamlit as st
import os
import sys
import re
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load models
all_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pickled_files', 'all_models.pkl')
best_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pickled_files', 'best_model_and_vectorizer.pkl')

try:
    with open(all_models_path, 'rb') as f:
        all_data = pickle.load(f)
    models_dict = all_data['models']
    vectorizer = all_data['vectorizer']
    results = all_data['results']
    model_loaded = True
    multi_model = True
except FileNotFoundError:
    try:
        with open(best_model_path, 'rb') as f:
            data = pickle.load(f)
        models_dict = {data.get('model_name', 'Model'): data['model']}
        vectorizer = data['vectorizer']
        results = None
        model_loaded = True
        multi_model = False
    except FileNotFoundError:
        model_loaded = False
        multi_model = False

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

def main():
    st.title(\"News Article Classifier\")
    st.subheader(\"Classify news articles into categories\")
    
    options = [\"Prediction\", \"Information\"]
    selection = st.sidebar.selectbox(\"Choose Option\", options)
    
    if selection == \"Information\":
        st.info(\"Project Information\")
        st.markdown(\"\"\"
        ### News Article Classification
        
        Classify news articles into:
        - Business
        - Technology
        - Sports
        - Education
        - Entertainment
        
        **Models Available:**
        \"\"\")
        
        if multi_model and results:
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Test Accuracy': f\"{result['test_accuracy']:.4f}\",
                    'F1 Score': f\"{result['f1_score']:.4f}\",
                    'Training Time': f\"{result['training_time']:.2f}s\"
                })
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            best = max(results, key=lambda x: results[x]['test_accuracy'])
            st.markdown(f\"\"\"
            **Total Models:** {len(models_dict)}  
            **Best Model:** {best}
            \"\"\")
        else:
            st.markdown(\"- Logistic Regression\")
        
        st.markdown(\"\"\"
        **Usage:**
        1. Go to Prediction page
        2. Select a model (if available)
        3. Enter news article text
        4. Click Classify
        
        **Technology:**
        - TF-IDF feature extraction
        - Text preprocessing with NLTK
        - Hyperparameter tuning via GridSearchCV
        \"\"\")
    
    if selection == \"Prediction\":
        st.info(\"Prediction with ML Models\")
        
        if not model_loaded:
            st.error(\"Models not found. Please train models first.\")
            st.stop()
        
        # Model selection
        if multi_model and len(models_dict) > 1:
            st.subheader(\"Select Model\")
            model_names = list(models_dict.keys())
            
            model_options = []
            for name in model_names:
                if results and name in results:
                    acc = results[name]['test_accuracy']
                    model_options.append(f\"{name} (Acc: {acc:.4f})\")
                else:
                    model_options.append(name)
            
            selected = st.selectbox(\"Model:\", model_options)
            selected_name = model_names[model_options.index(selected)]
            model = models_dict[selected_name]
            
            if results and selected_name in results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(\"Accuracy\", f\"{results[selected_name]['test_accuracy']:.4f}\")
                with col2:
                    st.metric(\"F1 Score\", f\"{results[selected_name]['f1_score']:.4f}\")
                with col3:
                    st.metric(\"Time\", f\"{results[selected_name]['training_time']:.2f}s\")
        else:
            model = list(models_dict.values())[0]
            selected_name = list(models_dict.keys())[0]
        
        st.subheader(\"Enter News Article\")
        news_text = st.text_area(\"Paste article text:\", 
                                 \"Type or paste news article here...\",
                                 height=200)
        
        if st.button(\"Classify\", type=\"primary\"):
            if news_text and news_text.strip() != \"Type or paste news article here...\":
                with st.spinner('Analyzing...'):
                    processed = clean_text(news_text)
                    vectorized = vectorizer.transform([processed])
                    prediction = model.predict(vectorized)
                    proba = model.predict_proba(vectorized)
                    
                    st.success(f\"**Category: {prediction[0].upper()}**\")
                    
                    st.subheader(\"Confidence Scores:\")
                    classes = model.classes_
                    probabilities = proba[0]
                    
                    results_df = pd.DataFrame({
                        'Category': classes,
                        'Confidence': probabilities
                    }).sort_values('Confidence', ascending=False)
                    
                    st.bar_chart(results_df.set_index('Category'))
                    
                    st.dataframe(
                        results_df.style.format({'Confidence': '{:.2%}'})
                        .background_gradient(cmap='Blues', subset=['Confidence']),
                        use_container_width=True
                    )
                    
                    if multi_model:
                        st.info(f\"Prediction by: **{selected_name}**\")
            else:
                st.warning(\"Please enter text to classify\")

if __name__ == '__main__':
    main()
