# 📧 Email Spam Classification using Logistic Regression

> An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies SMS/email messages as **Spam** or **Ham (Not Spam)** using **TF-IDF feature extraction** and **Logistic Regression**.

---

## 📌 Project Overview

Email and SMS spam messages are a common problem that can affect user productivity, security, and trust.

This project develops a machine learning-based **Spam Classification System** that automatically analyzes a text message and predicts whether it is:

* 🟢 **HAM** — legitimate message
* 🔴 **SPAM** — unwanted/promotional/fraudulent message

The project follows a complete machine learning pipeline:

```text
Raw SMS Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
NLP Text Preprocessing
      ↓
Train/Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Saved Model + Vectorizer
      ↓
User Input Prediction
```

---

# 🎯 Business Problem

Spam messages can contain:

* Unwanted advertisements
* Promotional offers
* Fraudulent messages
* Fake rewards
* Suspicious links
* Financial scams
* Phishing-style content

Manually identifying every spam message is inefficient.

The objective of this project is to build a machine learning classifier that can automatically identify suspicious messages and support automated spam filtering.

---

# 🎯 Project Objectives

The main objectives are:

1. Load and understand a real-world SMS spam dataset.
2. Perform data cleaning and exploratory data analysis.
3. Analyze the distribution of Ham and Spam messages.
4. Apply NLP preprocessing techniques.
5. Convert text into numerical features using TF-IDF.
6. Train a Logistic Regression classification model.
7. Evaluate the model using multiple classification metrics.
8. Analyze false positives and false negatives.
9. Save the trained model and TF-IDF vectorizer.
10. Accept new user messages and generate Spam/Ham predictions.
11. Store prediction results for further analysis.
12. Prepare the project for GitHub and future Streamlit deployment.

---

# 📊 Dataset

This project uses the **SMS Spam Collection Dataset**.

The dataset contains SMS messages labeled as:

* `ham`
* `spam`

Each record contains:

| Column    | Description                         |
| --------- | ----------------------------------- |
| `label`   | Message classification: ham or spam |
| `message` | Actual SMS text                     |

The dataset contains approximately **5,500+ SMS messages**.

### Dataset Source

UCI Machine Learning Repository:

**SMS Spam Collection Dataset**

Dataset page:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

# 🗂️ Project Structure

```text
Email-Spam-Classifier/
│
├── data/
│   ├── SMSSpamCollection
│   ├── spam_cleaned.csv
│   ├── spam_preprocessed.csv
│   ├── model_predictions.csv
│   └── user_predictions.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── spam_classifier.pkl
│
├── notebooks/
│   └── email_spam_classification.ipynb
│
├── screenshots/
│
├── src/
│
├── venv/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🔎 Project Workflow

## 1. Data Collection

The SMS Spam Collection dataset was downloaded and placed inside the `data/` directory.

The original dataset is tab-separated, so it was loaded using:

```python
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"]
)
```

---

# 🧹 2. Data Cleaning

The dataset was inspected for:

* Missing values
* Duplicate records
* Incorrect data types
* Class distribution
* Message length
* Duplicate messages

Duplicate records were removed using:

```python
df = df.drop_duplicates().reset_index(drop=True)
```

The cleaned dataset was saved as:

```text
data/spam_cleaned.csv
```

---

# 📊 3. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the dataset.

### Analysis included:

* Dataset shape
* Column information
* Missing-value analysis
* Duplicate analysis
* Ham vs Spam distribution
* Message length distribution
* Word-frequency analysis
* Common words in Spam messages
* Common words in Ham messages

### Class Distribution

The dataset contains significantly more legitimate messages than spam messages.

Therefore, accuracy alone is not sufficient for evaluating the classifier.

Additional metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* Specificity

were also considered.

---

# 🧠 4. NLP Text Preprocessing

Raw text cannot be directly provided to a traditional machine learning algorithm.

Therefore, the messages were processed using NLP techniques.

### Preprocessing pipeline

```text
Raw Text
   ↓
Lowercase
   ↓
Remove URLs
   ↓
Remove Email Addresses
   ↓
Remove HTML
   ↓
Remove Punctuation
   ↓
Normalize Whitespace
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Clean Text
```

The preprocessing function includes:

```python
def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)
```

The processed dataset was saved as:

```text
data/spam_preprocessed.csv
```

---

# 🔢 5. Label Encoding

The categorical labels were converted into numerical values.

```python
df["target"] = df["label"].map({
    "ham": 0,
    "spam": 1
})
```

Therefore:

```text
HAM  → 0
SPAM → 1
```

---

# ✂️ 6. Train/Test Split

The dataset was divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Configuration

| Parameter      | Value |
| -------------- | ----: |
| Test Size      |   20% |
| Training Size  |   80% |
| Random State   |    42 |
| Stratification |   Yes |

### Why Stratification?

The dataset is imbalanced between Ham and Spam messages.

Using:

```python
stratify=y
```

helps maintain a similar class distribution in both training and testing datasets.

---

# 🧮 7. TF-IDF Feature Extraction

Machine learning models require numerical input.

The cleaned text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

```python
tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

### TF-IDF Configuration

| Parameter      | Value  |
| -------------- | ------ |
| `max_features` | 5000   |
| `ngram_range`  | (1, 2) |
| `min_df`       | 2      |
| `max_df`       | 0.95   |
| `sublinear_tf` | True   |

The vectorizer was fitted **only on the training dataset**:

```python
X_train_tfidf = tfidf.fit_transform(X_train)
```

The test data was transformed using the already-fitted vectorizer:

```python
X_test_tfidf = tfidf.transform(X_test)
```

This prevents **data leakage**.

---

# 🤖 8. Logistic Regression Model

The main classification algorithm used in this project is **Logistic Regression**.

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)
```

Logistic Regression is well suited for text classification because TF-IDF produces a high-dimensional sparse feature representation.

---

# 📈 9. Model Evaluation

The model was evaluated using multiple metrics.

### Accuracy

Measures the overall proportion of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many messages predicted as Spam were actually Spam.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

### Recall

Measures how many actual Spam messages were successfully detected.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

### F1-Score

The harmonic mean of Precision and Recall.

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Specificity

Measures the proportion of legitimate Ham messages correctly identified.

```text
Specificity =
True Negatives /
(True Negatives + False Positives)
```

---

# 📊 Confusion Matrix

The confusion matrix contains four categories:

```text
                    Predicted
                  Ham       Spam

Actual Ham        TN         FP

Actual Spam       FN         TP
```

Where:

* **TN** = True Negative
* **FP** = False Positive
* **FN** = False Negative
* **TP** = True Positive

False positives and false negatives were analyzed separately to understand model errors.

---

# 🔍 10. Feature Importance

Logistic Regression provides coefficients that can be used to understand which TF-IDF features are associated with each class.

```python
feature_names = tfidf.get_feature_names_out()

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})
```

### Interpretation

Positive coefficient:

```text
Feature → more associated with Spam
```

Negative coefficient:

```text
Feature → more associated with Ham
```

This provides a degree of interpretability for the classification model.

---

# 💾 11. Model Saving

The trained Logistic Regression model was saved using `joblib`.

```python
joblib.dump(
    model,
    "models/spam_classifier.pkl"
)
```

The TF-IDF vectorizer was also saved:

```python
joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)
```

This allows the trained model to be reused without retraining it every time.

---

# 📧 12. User Input Prediction

The project supports prediction on a new message entered by the user.

The prediction pipeline is:

```text
User Message
      ↓
Text Preprocessing
      ↓
TF-IDF Transformation
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Ham / Spam
      ↓
Probability
```

Example:

```python
message = "Congratulations! You have won a free prize. Claim now!"

result = predict_spam(message)

print(result["prediction"])
```

Possible output:

```text
SPAM
```

The model also provides class probabilities:

```python
print(result["ham_probability"])
print(result["spam_probability"])
```

---

# 🧪 Example Predictions

### Example 1 — Normal Message

```text
Hey, are we meeting for lunch today?
```

Expected classification:

```text
HAM
```

---

### Example 2 — Promotional Message

```text
Congratulations! You have won a free prize. Click here to claim your reward!
```

The trained model evaluates the message and returns a Ham/Spam prediction with probabilities.

---

### Example 3 — Work Message

```text
Can you send me the project report?
```

The model processes the message through the same NLP and TF-IDF pipeline before generating its prediction.

---

# 💡 Prediction Function

The core prediction function is:

```python
def predict_spam(message):

    cleaned_message = clean_text(message)

    message_tfidf = tfidf.transform(
        [cleaned_message]
    )

    prediction = model.predict(
        message_tfidf
    )[0]

    probability = model.predict_proba(
        message_tfidf
    )[0]

    ham_probability = probability[0]
    spam_probability = probability[1]

    result = "SPAM" if prediction == 1 else "HAM"

    return {
        "message": message,
        "cleaned_message": cleaned_message,
        "prediction": result,
        "ham_probability": ham_probability,
        "spam_probability": spam_probability
    }
```

---

# 📁 Generated Files

### `spam_cleaned.csv`

Contains the cleaned dataset after duplicate removal and initial data processing.

### `spam_preprocessed.csv`

Contains the NLP-preprocessed messages.

### `model_predictions.csv`

Contains predictions generated on the test dataset.

### `user_predictions.csv`

Stores predictions generated from user-entered messages.

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF

## NLP

* NLTK
* Stopword Removal
* Tokenization
* Lemmatization

## Model Persistence

* Joblib

## Deployment

* Streamlit

## Development

* Jupyter Notebook
* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Vinayaak42/Email-Spam-Classifier.git
```

Move into the project directory:

```bash
cd Email-Spam-Classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📓 Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/email_spam_classification.ipynb
```

Run the cells sequentially.

---

# 🚀 Running Prediction

After the model and vectorizer have been trained and saved, load them:

```python
model = joblib.load(
    "models/spam_classifier.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)
```

Then provide a message:

```python
message = input(
    "Enter your email/SMS message: "
)

result = predict_spam(message)

print(result["prediction"])
```

---

# 🌐 Streamlit Deployment

The next deployment stage of the project is a Streamlit application.

The planned application will provide:

```text
📧 Email Spam Classifier
────────────────────────────

Enter your message:

[                         ]

        [ Predict ]

────────────────────────────

Prediction: SPAM

Ham Probability: 5%
Spam Probability: 95%
```

The Streamlit application will load:

```text
models/
├── spam_classifier.pkl
└── tfidf_vectorizer.pkl
```

and perform real-time predictions.

---

# 🔐 Important Machine Learning Consideration

The TF-IDF vectorizer must not be refitted on new user input.

Correct:

```python
tfidf.transform([cleaned_message])
```

Incorrect:

```python
tfidf.fit_transform([cleaned_message])
```

The vectorizer was fitted on the training data and must retain the same vocabulary and feature representation during inference.

---

# ⚠️ Limitations

This project has several limitations:

1. The model was trained on an SMS spam dataset and may not represent every type of modern email spam.
2. Spam patterns change over time.
3. Text preprocessing can remove some contextual information.
4. TF-IDF does not understand deep semantic relationships between words.
5. The model may incorrectly classify unusual messages.
6. Probability values should not be interpreted as guaranteed real-world probabilities.
7. The model does not independently inspect URLs, attachments, sender reputation, or email headers.

---

# 🚀 Future Improvements

Potential improvements include:

### Machine Learning

* Hyperparameter tuning
* Cross-validation
* Class-weight optimization
* Naive Bayes comparison
* Linear SVM comparison
* Random Forest comparison

### NLP

* Word embeddings
* Word2Vec
* GloVe
* FastText
* Transformer-based models
* BERT
* DistilBERT

### Application

* Streamlit web interface
* Batch CSV prediction
* Prediction history
* Confidence visualization
* File upload
* Email integration

### Production

* REST API using FastAPI
* Docker deployment
* Model versioning
* Monitoring
* Logging
* Automated retraining
* Model performance monitoring

---

# 📸 Screenshots

Add project screenshots inside:

```text
screenshots/
```

Recommended screenshots:

```text
screenshots/
├── eda_class_distribution.png
├── message_length_distribution.png
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
└── prediction_output.png
```

Once the Streamlit application is created, add:

```text
streamlit_app.png
```

to showcase the final application.

---

# 🎓 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Supervised Learning
* Binary Classification
* Natural Language Processing
* Text Cleaning
* Tokenization
* Stopword Removal
* Lemmatization
* TF-IDF
* N-grams
* Sparse Matrices
* Logistic Regression
* Stratified Train/Test Split
* Data Leakage Prevention
* Classification Metrics
* Confusion Matrix
* ROC-AUC
* Model Interpretability
* Probability Prediction
* Model Serialization
* Inference Pipeline

---

# 💼 Resume Project Description

**Email Spam Classification using Logistic Regression | Python, NLP, Scikit-learn**

Developed an end-to-end NLP-based spam classification system using TF-IDF feature extraction and Logistic Regression to classify SMS/email messages as Ham or Spam. Performed data cleaning, exploratory analysis, text preprocessing, lemmatization, stratified train-test splitting, model evaluation using Precision, Recall, F1-score, ROC-AUC and confusion matrix, and implemented reusable model/vectorizer serialization with Joblib for real-time user-input prediction.

---

# 🧑‍💻 Skills Demonstrated

```text
Python
Pandas
NumPy
Scikit-learn
NLTK
NLP
TF-IDF
Logistic Regression
Machine Learning
Text Classification
Data Preprocessing
EDA
Feature Engineering
Model Evaluation
Model Deployment
Streamlit
Git
GitHub
```

---

# 👨‍💻 Author

## Vinayak Kesti

Data Scientist | Data Analyst | AI & Machine Learning

📍 Bengaluru, Karnataka, India

### GitHub

https://github.com/Vinayaak42

### LinkedIn

https://www.linkedin.com/in/vinayak-kesti

---

# ⭐ Project Highlights

* 📧 End-to-end Spam Classification
* 🧹 Complete NLP preprocessing pipeline
* 📊 Exploratory Data Analysis
* 🔢 TF-IDF feature engineering
* 🤖 Logistic Regression classifier
* 📈 Multiple model evaluation metrics
* 🔍 Feature coefficient analysis
* 💾 Saved ML model and vectorizer
* 🧑‍💻 Real-time user input prediction
* 📁 Prediction history
* 🚀 Ready for Streamlit deployment

---

# 📜 License

This project is intended for educational, portfolio, and demonstration purposes.

The dataset is provided by the UCI Machine Learning Repository and is subject to its applicable dataset terms.
