# ✉️ AI-Powered Spam Email Detector

An end-to-end Machine Learning and Natural Language Processing (NLP) web application built using **Python, Scikit-Learn, and Streamlit**. This project classifies raw email input text into either **Spam** or **Ham (Not Spam)** by applying text-vectorization heuristics and comparing two distinct machine learning models on the fly.

Built with **Clean Architecture** principles in mind, the codebase explicitly decouples data preprocessing, model training, and the presentation/UI layer—making it highly maintainable, testable, and production-ready.

---

## 🚀 Live Demo
🔗 **[Click here to view the live deployed application]https://rrajesh-spam-detector.streamlit.app/**

---

## 🏗️ Clean Architecture & Project Structure

The project repository segregates functional layers to reflect real-world industry standards rather than a messy single-script notebook environment:

```text
spam_detector/
│
├── data/
│   └── mail_data.csv          # Structured training & testing datasets
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Text cleansing, label encoding, and TF-IDF pipeline
│   ├── model_trainer.py       # Algorithmic training, evaluation metrics, and serialization
│   └── predictor.py           # Clean abstraction layer providing the application inference API
│
├── models/
│   ├── logistic_regression.pkl # Serialized Logistic Regression weights
│   ├── naive_bayes.pkl         # Serialized Multinomial Naive Bayes model
│   └── vectorizer.pkl          # Saved vocabulary transformer mapping text to numbers
│
├── app.py                     # Streamlit frontend reactive interface dashboards
├── requirements.txt           # Project structural dependencies
└── .gitignore                 # Prevents pushing virtual env files (`venv`) to source control
