import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import string
import warnings
import nltk
from nltk.corpus import stopwords
from pickle import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt  # Importing only for Confusion Matrix

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Constants
DATA_PATH = "emails.csv"
MODEL_DIR = "models"
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(file_path):
    """Load dataset from CSV file."""
    dataset = pd.read_csv(file_path)
    print("\n✅ Dataset Loaded Successfully!\n")
    return dataset


def explore_data(dataset):
    """Display dataset information and statistics."""
    print("📊 Dataset Information:")
    print(dataset.info(), "\n")

    print("📌 Dataset Statistics:")
    print(dataset.describe(), "\n")

    print("🔹 First 5 Records:")
    print(dataset.head(), "\n")

    print("📌 Spam Frequency Counts:")
    print(dataset['spam'].value_counts(), "\n")


def clean_text(text):
    """Remove punctuation and stopwords from text."""
    nopunc = ''.join([char for char in text if char not in string.punctuation])
    return ' '.join([word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')])


def preprocess_data(dataset):
    """Clean and vectorize text data."""
    dataset.drop_duplicates(inplace=True)
    vectorizer = CountVectorizer(stop_words='english')  # Auto-remove stopwords
    message_vectors = vectorizer.fit_transform(dataset['text'])

    # Save vectorizer for future use
    dump(vectorizer, open(VECTORIZER_FILE, "wb"))
    print("✅ Vectorizer Saved!\n")

    return message_vectors, dataset['spam']


def train_model(X_train, y_train):
    """Train and save the Naïve Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, open(MODEL_FILE, "wb"))
    print("✅ Model Saved Successfully!\n")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    y_pred = model.predict(X_test)

    print(f"\n🎯 Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("📌 Classification Report:\n", classification_report(y_test, y_pred), "\n")

    # Display Confusion Matrix using Matplotlib
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(dpi=100)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# Main Execution
if __name__ == "__main__":
    dataset = load_data(DATA_PATH)
    explore_data(dataset)

    X, y = preprocess_data(dataset)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
