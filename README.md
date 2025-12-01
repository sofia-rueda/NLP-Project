#  IMDb Sentiment Analysis — NLP Project

**Classical Machine Learning with TF-IDF, Word2Vec, and GloVe Embeddings**
This repository contains a Natural Language Processing (NLP) project that classifies movie reviews from the IMDB 50K dataset into positive or negative sentiment categories.
Dataset Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
---

##  Overview

This project applies **Natural Language Processing (NLP)** techniques to classify IMDb movie reviews as **positive** or **negative**.
We compare multiple text-representation methods—**TF-IDF**, **Word2Vec**, and **GloVe**—combined with two classical machine-learning models: **Logistic Regression** and **Linear SVM**.

Our goal is to evaluate how different feature extraction approaches affect sentiment classification performance under **balanced and imbalanced** conditions.

---

##  Definition of the Task

The task is **binary sentiment classification**.
Given a movie review, the model determines whether its sentiment is:

* **0 — Negative**
* **1 — Positive**

This is a common NLP application used in customer feedback systems, recommendation engines, and automated content moderation.

---

##  My Approach

We followed a standard NLP + ML workflow:

1. **Load IMDb dataset** (50k reviews)
2. **Explore class balance**
3. **Apply preprocessing**:

   * lowercasing
   * punctuation removal
   * stopword removal
   * tokenization
   * lemmatization
4. **Generate three types of text embeddings**:

   * **TF-IDF (sparse features)**
   * **Word2Vec (semantic dense vectors)**
   * **GloVe (pretrained embeddings)**
5. **Train two classification models**:

   * Logistic Regression
   * Linear SVC
6. **Test models under both balanced and imbalanced class settings**
7. **Evaluate performance** using accuracy, precision, recall, macro F1-score
8. **Run prediction on a custom review**
9. **Create visualizations**: Word Cloud, confusion matrices

---

##  Summary of Performance Achieved

### **Best Overall Model**

* **TF-IDF + Logistic Regression (Balanced Classes)**

  * **Accuracy:** ~89%
  * **Macro F1:** 0.89
  * **Most consistent and strongest performance across all metrics**

### **Best Word2Vec Model**

* **Word2Vec + Logistic Regression (Balanced)**

  * **Accuracy:** ~86%
  * **Macro F1:** 0.86
  * Word2Vec captured semantic meaning well.

### **Best GloVe Model**

* **GloVe + Linear SVC (Balanced)**

  * **Accuracy:** ~80%
  * **Macro F1:** ~0.79
  * Strongest Glove performance, but lower than TF-IDF and Word2Vec.

### **Handling Imbalance**

We evaluated models with artificially imbalanced splits (60/40).

* TF-IDF remained the most robust under imbalance.
* Dense embeddings (Word2Vec/GloVe) dropped more in accuracy.

---

##  Summary of Work Done

This project includes:

✔ Loading & analyzing the IMDb dataset
✔ Full NLP preprocessing pipeline
✔ Word Cloud visualization
✔ TF-IDF, Word2Vec, and GloVe implementation
✔ Balanced vs. imbalanced model comparisons
✔ Logistic Regression & Linear SVC training
✔ Saved predictions for custom input
✔ Confusion matrices
✔ Performance reporting with classification reports

---

##  Data Loading & Initial Look

We used the IMDb dataset (50k reviews: 25k positive, 25k negative).
We verified class distribution, lengths, and common words before preprocessing.

---

##  Data Preprocessing

Preprocessing steps included:

* Lowercasing
* Removing punctuation
* Removing stopwords
* Tokenizing
* Lemmatizing using SpaCy
* Joining cleaned tokens back into text

This reduces noise and improves model performance.

---

## 🤖 Machine Learning

We trained and evaluated:

### **Models**

* Logistic Regression
* Linear SVC

### **Feature Extraction Methods**

* **TF-IDF vectorizer**
* **Word2Vec (self-trained)**
* **GloVe (pretrained 100-dim vectors)**

We compared results across:

1. Balanced training
2. Imbalanced training

---

##  How to Reproduce Results
Here is a clean, polished **README setup** for your GitHub repository, following the exact structure and tone you liked—BUT tailored specifically to your **IMDB Sentiment Analysis project (TF-IDF, Word2Vec, GloVe, LR, Linear SVC, Balanced vs Imbalanced classes)**.

---

#  Required Packages

To run this project, you will need the following Python packages:

### **pandas**

Used for loading, cleaning, and manipulating the IMDB movie reviews dataset.

### **numpy**

Provides numerical operations used for vectorized text embeddings such as Word2Vec and GloVe.

### **matplotlib**

A plotting library used to create visualizations (accuracy charts, word clouds, confusion matrices).

### **seaborn**

Built on top of matplotlib and used for more polished plots such as heatmaps for confusion matrices.

### **scikit-learn**

Core machine learning library used for:

* TF-IDF vectorization
* Logistic Regression
* Linear SVC
* Train/test split
* Model evaluation (accuracy, F1-score, classification report)

### **NLTK**

Used for preprocessing steps including:

* tokenization
* stopword removal
* lemmatization

### **gensim**

Provides Word2Vec and functions needed to compute sentence embeddings.

### **wordcloud**

Used to generate the sentiment word cloud for EDA.

---

#  Installation Instructions

### **1. Create a Virtual Environment** *(recommended)*

This keeps the project dependencies clean and isolated.

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### **2. Install Required Packages**

If your repository includes a `requirements.txt`, simply run:

```bash
pip install -r requirements.txt
```

Otherwise install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk gensim wordcloud
```

### **3. Verify Installation**

You can confirm packages installed successfully by running:

```python
import sklearn
import nltk
import gensim
```

### **4. (Optional) Install Jupyter Notebook**

If you're working with notebooks:

```bash
pip install notebook
```

---

#  Using Google Colab (Optional)

If you prefer not to install anything locally:

1. Go to Google Colab
2. Upload the notebook file (`.ipynb`)
3. Upload the dataset
4. Run all cells—most required packages are already installed

---
