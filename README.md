# News Article Classification Project

Classify news articles into categories using machine learning.

## Quick Start

```bash
# Setup environment
conda create --name newsclassifier python=3.9
conda activate newsclassifier
pip install -r requirements.txt

# Train models
python train_multiple_models.py  # Trains 5 models with tuning
python train_model.py            # Quick single model training

# Test
python test_model.py

# Run web app
cd Streamlit
streamlit run base_app.py
```

## Dataset

News articles with 5 categories:
- Business, Technology, Sports, Education, Entertainment

Features: Headlines, Description, Content, URL, Category

## Models

Trained 5 models with hyperparameter tuning:
1. Support Vector Machine (98.30% - Best)
2. Logistic Regression (98.25%)
3. Naive Bayes (97.85%)
4. Gradient Boosting (96.65%)
5. Random Forest (96.35%)

## Features

- Text preprocessing (tokenization, lemmatization, stopword removal)
- TF-IDF vectorization (5000 features, unigrams + bigrams)
- GridSearchCV hyperparameter tuning
- Multi-model Streamlit web app
- Visualization scripts

## Files

- `train_multiple_models.py` - Train all models
- `train_model.py` - Quick training
- `test_model.py` - Test models
- `visualize_models.py` - Generate visualizations
- `Streamlit/base_app.py` - Web interface

## Results

Best model: SVM with 98.30% accuracy
See `model_comparison_results.csv` for detailed metrics

## 6. Streamlit<a class="anchor" id="streamlit"></a>

### What is Streamlit?

[Streamlit](https://www.streamlit.io/)  is a framework that acts as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models.

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

[Streamlit](https://www.streamlit.io/)  takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

For this repository, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `base_app.py`          | Streamlit application definition. |


#### 6.1 Running the Streamlit web app on your local machine

As a first step to becoming familiar with our web app's functioning, we recommend setting up a running instance on your own local machine. To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

- Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

- Navigate to the base of your repo where your base_app.py is stored, and start the Streamlit app.

 ```bash
 cd 2401FTDS_Classification_Project/Streamlit/
 streamlit run base_app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```
You should also be automatically directed to the base page of your web app. This should look something like:

<div id="s_image" align="center">
  <img src="https://github.com/ereshia/2401FTDS_Classification_Project/blob/main/Streamlit_image.png" width="850" height="400" alt=""/>
</div>

Congratulations! You've now officially deployed your first web application!

#### 6.2 Deploying your Streamlit web app

- To deploy your app for all to see, click on `deploy`.
  
- Please note: If it's your first time deploying it will redirect you to set up an account first. Please follow the instructions.

## 7. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Neo Molitsane](https://github.com/)                                         | Neo.Molitsane@absa.africa
| [Thabang Mathebula](https://github.com/)                                                                   | Thabang.Mathebula@absa.africa

