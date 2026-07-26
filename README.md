<div align="center">

# 📰 AI-Powered Fake News Detection

### Supervised NLP text classification — distinguishing real news from fabricated news using classical feature engineering and five comparative ML models

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20BoW-9146FF?style=flat-square)](#-feature-engineering)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-5%20Models%20Compared-2ea44f?style=flat-square)](#-machine-learning-models)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](#-results)

</div>

---

> **TL;DR** — Two raw CSV datasets (real + fake news) are cleaned, merged, and vectorized with TF-IDF, then used to train and benchmark five classifiers. **Random Forest wins at 99.51% accuracy.** Every model, vectorizer, and chart is persisted to disk for reuse.

<p align="center">
  <img src="images/model_accuracy.png" alt="Model Accuracy Comparison" width="700"/>
  <br/>
  <em>Model accuracy comparison — see <a href="#-model-comparison">Section 12</a> for full breakdown</em>
</p>

---

## 📑 Table of Contents

| | | |
|---|---|---|
| [🧭 1. Project Overview](#-1-project-overview) | [📊 8. Feature Engineering](#-8-feature-engineering) | [🗂️ 15. Project Outputs](#-15-project-outputs) |
| [🎯 2. Project Objectives](#-2-project-objectives) | [✂️ 9. Train-Test Split](#-9-train-test-split) | [⚙️ 16. Installation Guide](#-16-installation-guide) |
| [🗃️ 3. Dataset Information](#-3-dataset-information) | [🤖 10. Machine Learning Models](#-10-machine-learning-models) | [🚀 17. Usage Guide](#-17-usage-guide) |
| [🛠️ 4. Technologies Used](#-4-technologies-used) | [📈 11. Model Evaluation](#-11-model-evaluation) | [🔭 18. Future Scope](#-18-future-scope) |
| [🔄 5. Project Workflow](#-5-project-workflow) | [🏆 12. Model Comparison](#-12-model-comparison) | [⚠️ 19. Limitations](#-19-limitations) |
| [🧹 6. Data Preprocessing](#-6-data-preprocessing) | [✅ 13. Results](#-13-results) | [🏁 20. Conclusion](#-20-conclusion) |
| [🔍 7. Exploratory Data Analysis](#-7-exploratory-data-analysis) | [📁 14. Project Folder Structure](#-14-project-folder-structure) | [📚 21. References](#-21-references) • [👤 22. Author](#-22-author) |

---

## 🧭 1. Project Overview

Misinformation spreads faster than verified reporting, and manually fact-checking every article is no longer feasible at scale. This project builds a complete, end-to-end ML pipeline that classifies a news article as **real** or **fake** using only its textual content — no external fact-checking APIs, no publisher metadata.

**Core idea:** real and fake articles differ in vocabulary, tone, and structure. Once text is converted into numeric features, a classifier can learn those differences.

- 🔤 **Primary feature representation:** TF-IDF (Term Frequency–Inverse Document Frequency)
- 🔤 **Secondary representation (comparison only):** Bag-of-Words
- 🤖 **Five classifiers trained independently** on the same split: Logistic Regression, Naive Bayes, Random Forest, KNN, and a Multi-Layer Perceptron — spanning parametric, probabilistic, ensemble, non-parametric, and deep learning paradigms
- 🧪 **Evaluated identically** with accuracy, precision, recall, F1-score, and confusion matrices
- 💾 **Fully reproducible** — every trained model and vectorizer is serialized with `joblib`

**Pipeline at a glance:** load two raw CSVs → clean & deduplicate → merge & shuffle → EDA → text preprocessing → TF-IDF / BoW vectorization → 80/20 split → train 5 models → evaluate → compare → run inference on unseen sentences.

> This README documents exactly what is implemented in `Fake_News_Detection.ipynb` — no additional preprocessing, feature engineering, or modeling steps have been added beyond what the notebook contains.

---

## 🎯 2. Project Objectives

- ✅ Build a from-scratch ML pipeline to classify news as **real** or **fake**, per the *Summer Internship Program in AI & ML 2026* brief (IICT)
- ✅ Load, inspect, and clean two independent raw datasets — handling missing values and duplicates before merging
- ✅ Perform EDA to understand class balance, subject distribution, and article length
- ✅ Engineer numerical text features using both **Bag-of-Words** and **TF-IDF**
- ✅ Train and compare 5 algorithms across parametric, probabilistic, ensemble, non-parametric, and neural-network paradigms
- ✅ Evaluate every model using accuracy, precision, recall, F1-score, and confusion matrices
- ✅ Benchmark all models and identify the best performer
- ✅ Persist every trained model and vectorizer with `joblib` for reuse without retraining
- ✅ Demonstrate real-world inference on hand-written, unseen sentences
- ✅ Produce documentation detailed enough to double as an IEEE-format project report source

---

## 🗃️ 3. Dataset Information

### 3.1 Dataset Files

| File | Description |
|---|---|
| `True.csv` | Verified / real news articles |
| `Fake.csv` | Fabricated / fake news articles |

### 3.2 Raw Dataset Shape

| Dataset | Rows | Columns (as loaded) |
|---|---|---|
| True News | 21,417 | 4 |
| Fake News | 23,502 | 172 ⚠️ |

> **⚠️ Data Quality Note:** `Fake.csv` loaded with 172 columns instead of 4 — a known artifact of unescaped commas inside article bodies generating extra "unnamed" columns. Resolved by explicitly retaining the intended schema:

<details>
<summary>📄 Show fix code</summary>

```python
fake = fake[['title', 'text', 'subject', 'date']]
```

</details>

Both datasets share an identical 4-column schema after this step.

### 3.3 Dataset Columns

| Column | Description |
|---|---|
| `title` | Headline of the article |
| `text` | Full body text |
| `subject` | Category/topic tag |
| `date` | Publication date |

**Engineered columns:**

| Column | Description |
|---|---|
| `text_length` | Character length of body text (`text.apply(len)`) |
| `label` | Target — `1` = real, `0` = fake |
| `content` | `title + " " + text`, the actual model input |

### 3.4 Target Labels

| Label | Meaning | Source File |
|---|---|---|
| `1` | Real News | `True.csv` |
| `0` | Fake News | `Fake.csv` |

### 3.5 Data Quality Checks

| Check | True News | Fake News |
|---|---|---|
| Missing `title` / `text` | 0 | 0 |
| Missing `subject` | 0 | 21 |
| Missing `date` | 0 | 21 |
| Duplicate rows | 206 | 23 |

### 3.6 Subject Category Breakdown

<table>
<tr>
<td valign="top">

**True News**

| Subject | Count |
|---|---|
| politicsNews | 11,272 |
| worldnews | 10,145 |

</td>
<td valign="top">

**Fake News (top categories)**

| Subject | Count |
|---|---|
| News | 9,050 |
| politics | 6,838 |
| left-news | 4,457 |
| Government News | 1,570 |
| US_News | 775 |
| Middle-east | 770 |

</td>
</tr>
</table>

> Additional low-count, malformed subject values appeared in raw `Fake.csv` (same comma-parsing artifact as above). Visualization was restricted to the six valid categories via an explicit `valid_subjects` filter.

### 3.7 Dataset Preparation Summary

1. Load both files independently via `pandas.read_csv`
2. Trim `Fake.csv` to 4 columns
3. Check missing values & duplicates on each dataset
4. Concatenate → inspect duplicates (44,919 → 44,690 rows after `drop_duplicates()`)
5. Assign labels: `1` = real, `0` = fake
6. Concatenate **again** — this final merge (44,919 rows) is what's carried into modeling
7. Shuffle via `sample(frac=1, random_state=42)`, reset index
8. Merge `title` + `text` → `content`; drop originals
9. Export to `data/processed/merged_news.csv`

---

## 🛠️ 4. Technologies Used

### Language & Environment

| Component | Detail |
|---|---|
| Language | Python |
| Environment | Jupyter Notebook (`Fake_News_Detection.ipynb`) |

### Core Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib.pyplot` | Visualization |
| `re` | Regex-based text cleaning |
| `nltk` | English stopword corpus |
| `scikit-learn` | Feature extraction, models, metrics |
| `joblib` | Model/vectorizer serialization |

### Feature Engineering

| Technique | Implementation |
|---|---|
| Bag-of-Words | `CountVectorizer(stop_words="english", max_features=5000)` |
| TF-IDF | `TfidfVectorizer(stop_words="english", max_features=5000)` |

### Algorithms

| Algorithm | scikit-learn Class |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000)` |
| Naive Bayes | `MultinomialNB()` |
| Random Forest | `RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)` |
| KNN | `KNeighborsClassifier(n_neighbors=5)` |
| Neural Network (MLP) | `MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)` |

### Evaluation Metrics

| Metric | Function |
|---|---|
| Accuracy | `accuracy_score` |
| Precision / Recall / F1 | `classification_report` |
| Confusion Matrix | `confusion_matrix`, `ConfusionMatrixDisplay` |

---

## 🔄 5. Project Workflow

<p align="center">
  <img src="images/workflow_diagram.png" alt="Project Workflow Diagram" width="600"/>
  <br/>
  <sub><i>📌 Placeholder — add an exported workflow diagram image here</i></sub>
</p>

<details>
<summary>🧩 View ASCII pipeline diagram</summary>

```
 Load True.csv & Fake.csv
            │
            ▼
      Data Cleaning  (missing values, duplicates)
            │
            ▼
      Exploratory Data Analysis (EDA)
            │
            ▼
      Label Assignment  (True = 1, Fake = 0)
            │
            ▼
      Dataset Merging & Shuffling
            │
            ▼
      Text Preprocessing  (cleaning function, stopwords)
            │
            ▼
      Feature Engineering  (Bag-of-Words, TF-IDF)
            │
            ▼
      Train-Test Split  (80/20, seed = 42)
            │
            ▼
      Model Training  (5 algorithms)
            │
            ▼
      Model Evaluation  (Accuracy, Precision, Recall, F1, Confusion Matrix)
            │
            ▼
      Model Comparison & Best Model Selection
            │
            ▼
      Custom Prediction on Unseen Sentences
```

</details>

---

## 🧹 6. Data Preprocessing

| Step | Purpose | Implementation | Impact |
|---|---|---|---|
| **6.1 Column Trimming** | Remove noise columns from malformed CSV rows | `fake = fake[['title','text','subject','date']]` | Clean, schema-consistent concatenation |
| **6.2 Missing Value Handling** | Identify gaps in `subject`/`date` | `isnull().sum()` | 21 missing values each, reported |
| **6.3 Duplicate Removal** | Remove exact repeated rows | `duplicated().sum()` → `drop_duplicates()` | 44,919 → 44,690 rows |
| **6.4 Label Encoding** | Numeric target for sklearn | `true["label"]=1`, `fake["label"]=0` | Enables standard classifiers |
| **6.5 Merge & Shuffle** | Remove ordering bias | `pd.concat` → `sample(frac=1, random_state=42)` → `reset_index(drop=True)` | Reproducible random order |
| **6.6 Feature Consolidation** | Unify headline + body | `content = title + " " + text`; drop originals | Single text field for vectorization |
| **6.7 Text Cleaning Function** | Normalize text | See below | Reduces noise & vocabulary size |

<details>
<summary>🐍 Show <code>clean_text()</code> implementation</summary>

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)
    return text
```

</details>

**Persisted intermediate datasets:**

| Stage | Output File |
|---|---|
| Merged (title + text) | `data/processed/merged_news.csv` |
| Post text-cleaning | `data/cleaned/cleaned_news.csv` |

---

## 🔍 7. Exploratory Data Analysis

<p align="center">
  <img src="images/news_distribution.png" width="320"/>
  <img src="images/true_subject_distribution.png" width="320"/>
  <img src="images/article_length_distribution.png" width="320"/>
</p>

| Chart | Purpose | Key Observation | Insight |
|---|---|---|---|
| **Real vs. Fake distribution**<br>`images/news_distribution.png` | Check class balance | 21,417 real vs. 23,502 fake | Reasonably balanced — no imbalance correction needed |
| **True News subject distribution**<br>`images/true_subject_distribution.png` | Topical composition of real news | `politicsNews` (11,272), `worldnews` (10,145) | Real news drawn from only 2 subject tags |
| **Fake News subject distribution**<br>*(displayed inline, not saved)* | Topical composition of fake news, filtered to 6 valid categories | `News` (9,050), `politics` (6,838), + 4 more | Fake news spans a wider variety of subjects |
| **Article length distribution**<br>`images/article_length_distribution.png` | Compare text length across classes | Both right-skewed, overlapping | Length alone isn't strongly discriminative — reinforces need for TF-IDF/BoW features |

---

## 📊 8. Feature Engineering

### Bag-of-Words *(comparison only)*

```python
bow_vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_bow = bow_vectorizer.fit_transform(news["content"])
```

| | |
|---|---|
| **Shape** | `(44919, 5000)` |
| **How it works** | Raw word counts across top 5,000 vocabulary terms |
| **Role** | Implemented *"for comparison only"* per the notebook; saved to `models/bow_vectorizer.pkl` — **not used to train any of the 5 models** |

### TF-IDF *(used for model training)*

```python
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(news["content"])
y = news["label"]
```

| | |
|---|---|
| **Shape** | `(44919, 5000)` |
| **How it works** | Weights terms by frequency-in-document vs. frequency-across-corpus |
| **Why chosen** | Down-weights common uninformative words, surfaces discriminative terms — this is the representation used for all 5 models |
| **Persistence** | `models/tfidf_vectorizer.pkl` |

---

## ✂️ 9. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Parameter | Value | | Split | Shape |
|---|---|---|---|---|
| Test size | 20% | | `X_train` | (35,935, 5,000) |
| Train size | 80% | | `X_test` | (8,984, 5,000) |
| Random state | 42 | | `y_train` / `y_test` | (35,935,) / (8,984,) |

> `random_state=42` guarantees the same split every run — accuracy comparisons across all 5 models are fully reproducible.

---

## 🤖 10. Machine Learning Models

Each model trained on **identical TF-IDF features** and the **identical train-test split**.

<details open>
<summary><b>10.1 Logistic Regression</b></summary>

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

| | |
|---|---|
| **Overview** | Linear parametric model estimating class probability via sigmoid over weighted TF-IDF features |
| **Why selected** | Parametric baseline per the project brief |
| **Advantages** | Fast, interpretable, strong on sparse high-dimensional text |
| **Limitations** | Assumes a linear decision boundary |

</details>

<details>
<summary><b>10.2 Naive Bayes</b></summary>

```python
nb = MultinomialNB()
nb.fit(X_train, y_train)
```

| | |
|---|---|
| **Overview** | Probabilistic classifier based on Bayes' theorem, suited to term-frequency features |
| **Why selected** | Lightweight standard text-classification baseline |
| **Advantages** | Extremely fast to train/predict |
| **Limitations** | Assumes feature (word) independence — doesn't hold in natural language |

</details>

<details>
<summary><b>10.3 Random Forest 🏆</b></summary>

```python
rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
```

| | |
|---|---|
| **Overview** | Ensemble of 20 decision trees on bootstrapped samples |
| **Why selected** | Represents ensemble-learning category; strong on sparse numerical data |
| **Advantages** | Robust to overfitting, handles high dimensionality well |
| **Limitations** | Less interpretable; larger forests are costlier (kept at 20 estimators here) |

</details>

<details>
<summary><b>10.4 K-Nearest Neighbors (KNN)</b></summary>

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

| | |
|---|---|
| **Overview** | Non-parametric, instance-based — majority vote of 5 nearest neighbors |
| **Why selected** | Represents non-parametric category per the project brief |
| **Advantages** | Simple, no distributional assumptions |
| **Limitations** | Expensive on high-dimensional sparse vectors; sensitive to `k` and dimensionality — reflected in its lower accuracy here |

</details>

<details>
<summary><b>10.5 Neural Network (MLPClassifier)</b></summary>

```python
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
```

| | |
|---|---|
| **Overview** | Feed-forward network, 1 hidden layer of 100 neurons, trained via backpropagation |
| **Why selected** | Represents the deep-learning component of the project brief |
| **Advantages** | Captures non-linear feature interactions |
| **Limitations** | Slower to train, less interpretable, sensitive to hyperparameters |

</details>

---

## 📈 11. Model Evaluation

| Metric | How It Was Used |
|---|---|
| **Accuracy** | `accuracy_score(y_test, predictions)` — all 5 models |
| **Precision / Recall / F1** | `classification_report` — Logistic Regression, Random Forest, Neural Network |
| **Confusion Matrix** | `confusion_matrix` / `ConfusionMatrixDisplay` — every model |
| **Classification Report** | Per-class + macro/weighted averages |

<details>
<summary>📋 Logistic Regression — detailed metrics</summary>

```
              precision    recall  f1-score   support

           0       0.99      0.98      0.99      4666
           1       0.98      0.99      0.99      4318

    accuracy                           0.99      8984
```

Confusion Matrix:
```
[[4589   77]
 [  41 4277]]
```

</details>

<details>
<summary>📋 Random Forest — detailed metrics</summary>

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      4666
           1       1.00      0.99      0.99      4318

    accuracy                           1.00      8984
```

</details>

<details>
<summary>📋 Neural Network — detailed metrics</summary>

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4666
           1       0.99      0.99      0.99      4318

    accuracy                           0.99      8984
```

</details>

---

## 🏆 12. Model Comparison

| Rank | Model | Accuracy |
|:---:|---|:---:|
| 🥇 | **Random Forest** | **99.51%** |
| 🥈 | Neural Network | 99.25% |
| 🥉 | Logistic Regression | 98.69% |
| 4 | Naive Bayes | 92.95% |
| 5 | KNN | 88.70% |

> **Best Model:** `Random Forest` — accuracy **0.995102** on the held-out test set.

- **Methodology:** All 5 models trained on identical TF-IDF features and the identical 80/20 split (`random_state=42`) — directly comparable results, consolidated into `outputs/model_results.csv`. Best model selected via `results_df.loc[results_df["Accuracy"].idxmax()]`.
- **Summary:** All models except KNN exceed 92% accuracy; Random Forest, Neural Network, and Logistic Regression form a top tier above 98%. KNN trails, consistent with its known weakness on high-dimensional sparse TF-IDF spaces.

<p align="center">
  <img src="images/model_accuracy.png" alt="Model Accuracy Comparison" width="650"/>
</p>

---

## ✅ 13. Results

- Five models trained and evaluated on a TF-IDF representation of **44,919** combined news articles
- 🥇 **Random Forest: 99.51%** — best performer
- 🥈 Neural Network: 99.25%
- 🥉 Logistic Regression: 98.69%
- Naive Bayes: 92.95%
- KNN: 88.70% *(lowest)*
- All models + both vectorizers persisted as `.pkl` files

**Custom prediction test** (Logistic Regression + TF-IDF):

| Input Sentence | Prediction | Result |
|---|:---:|---|
| "India launches a new AI education program for students." | `0` | ❌ Fake News |
| "NASA successfully launched a new satellite into space on Monday." | `1` | ✅ Real News |
| "The Earth is flat according to new government research." | `0` | ❌ Fake News |

---

## 📁 14. Project Folder Structure

```
AI-FAKE-NEWS-DETECTION/
├── data/
│   └── raw/
│       ├── Fake.csv
│       └── True.csv
├── images/
│   ├── article_length_distribution.png
│   ├── model_accuracy.png
│   ├── news_distribution.png
│   └── true_subject_distribution.png
├── models/
│   ├── bow_vectorizer.pkl
│   ├── knn_model.pkl
│   ├── logistic_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── neural_network_model.pkl
│   ├── random_forest_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/
│   └── Fake_News_Detection.ipynb
├── outputs/
│   └── model_results.csv
├── presentation/
├── reports/
├── rough sough/
├── scr/
├── .gitignore
├── project overview.docx
├── README.md
└── requirement.txt
```

> 📌 The notebook also writes intermediate datasets to `data/processed/merged_news.csv` and `data/cleaned/cleaned_news.csv` (referenced in code, not visible in the captured directory screenshot).

---

## 🗂️ 15. Project Outputs

| Output Type | File(s) |
|---|---|
| **Models** | `logistic_model.pkl`, `naive_bayes_model.pkl`, `random_forest_model.pkl`, `knn_model.pkl`, `neural_network_model.pkl` |
| **Vectorizers** | `bow_vectorizer.pkl`, `tfidf_vectorizer.pkl` |
| **Images** | `news_distribution.png`, `true_subject_distribution.png`, `article_length_distribution.png`, `model_accuracy.png` |
| **CSV** | `outputs/model_results.csv`, `data/processed/merged_news.csv`, `data/cleaned/cleaned_news.csv` |
| **Notebook** | `notebook/Fake_News_Detection.ipynb` |

<p align="center">
  <img src="images/screenshot_placeholder.png" alt="Project Screenshot Placeholder" width="600"/>
  <br/>
  <sub><i>📌 Placeholder — add a notebook/output screenshot here</i></sub>
</p>

---

## ⚙️ 16. Installation Guide

<details>
<summary>📋 Requirements</summary>

```
pandas
numpy
matplotlib
scikit-learn
nltk
joblib
```

</details>

```bash
# Clone the repository
git clone <repository-url>
cd AI-FAKE-NEWS-DETECTION

# Install dependencies
pip install -r requirement.txt

# Launch the notebook
jupyter notebook notebook/Fake_News_Detection.ipynb
```

Run all cells sequentially. The notebook expects `True.csv` and `Fake.csv` at `../data/raw/` relative to its location.

<details>
<summary>🐍 Run a quick prediction</summary>

```python
sample_news = ["Your custom news sentence here."]
sample_vector = vectorizer.transform(sample_news)
prediction = model.predict(sample_vector)
print("Result:", "Real News" if prediction[0] == 1 else "Fake News")
```

</details>

---

## 🚀 17. Usage Guide

1. Place `True.csv` and `Fake.csv` inside `data/raw/`
2. Run `notebook/Fake_News_Detection.ipynb` top to bottom
3. Review visualizations auto-saved to `images/`
4. Check `outputs/model_results.csv` for the full accuracy comparison
5. Load any saved model + vectorizer for inference without retraining:

```python
import joblib

model = joblib.load("models/random_forest_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

text = ["Sample news article text."]
vector = vectorizer.transform(text)
print(model.predict(vector))
```

---

## 🔭 18. Future Scope

- 🔧 Explicitly apply `clean_text()` to `content` before vectorization; compare cleaned vs. uncleaned performance
- 🧠 Add word embeddings (Word2Vec, GloVe) or transformer embeddings, per the original workflow's "embeddings" step
- 🎛️ Hyperparameter tuning (grid/randomized search) for Random Forest and the Neural Network
- 🌲 Scale up Random Forest estimators and evaluate the accuracy/runtime trade-off
- 📉 Add ROC curves and AUC alongside existing metrics
- 🌐 Deploy the best model (Random Forest) behind a web interface or API
- 📚 Test generalization on additional/more recent datasets beyond current subject categories

---

## ⚠️ 19. Limitations

- Real-news subset spans only **2 subject categories** (`politicsNews`, `worldnews`) — may limit generalization
- `clean_text()` is defined but **not shown being explicitly applied** to `content` before vectorization — both vectorizers were fit on the raw merged text
- An initial deduplication step reduced rows (44,919 → 44,690), but the dataset actually vectorized came from a **second, non-deduplicated concat** (44,919 rows) — the dedup result wasn't carried into modeling
- `Fake.csv`'s original 172-column parsing artifact may leave residual noise in `subject`/`date` for affected rows
- Evaluation used a **single train-test split**, not cross-validation
- Random Forest used a relatively small ensemble (`n_estimators=20`), not benchmarked against larger sizes

---

## 🏁 20. Conclusion

This project implemented and compared five ML algorithms — Logistic Regression, Naive Bayes, Random Forest, KNN, and an MLP neural network — for real vs. fake news classification using TF-IDF features. Starting from two raw CSV sources, the pipeline addressed genuine data-quality issues (a malformed 172-column file, missing values, duplicates) before merging, shuffling, and vectorizing 44,919 articles.

**Random Forest achieved the top accuracy at 99.51%**, closely followed by the Neural Network and Logistic Regression (both >98%). Naive Bayes and KNN trailed behind, consistent with their known theoretical limitations on high-dimensional sparse text data.

The comparative structure — spanning parametric, probabilistic, ensemble, non-parametric, and deep-learning paradigms — fulfills the project brief's discussion requirement, and this documentation is intended as the empirical foundation for a corresponding IEEE-format report.

---

## 📚 21. References

- **Libraries:** pandas, numpy, matplotlib, scikit-learn, nltk, joblib
- **Dataset:** `True.csv` / `Fake.csv` — per project guidelines, sourced from the Kaggle *Fake News Detection Dataset*
- **Project Brief:** *Project-1 Summer Internship Program in AI & ML Machine Learning 2026 — AI-Powered Fake News Detection Using Text Classification*, Indian Institute of Computing and Technology (IICT)

---

## 👤 22. Author

**Machine Learning Engineer / Project Contributor**

Developed as part of the Summer Internship Program in AI & ML (2026) — covering end-to-end pipeline design, data preprocessing, feature engineering, model training, and comparative evaluation for fake news detection.

<div align="center">

---
⭐ If you found this project useful, consider starring the repository.
</div>
