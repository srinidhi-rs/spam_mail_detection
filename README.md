📩 Spam Email Detection using Machine Learning & NLP 🚀

This project implements a Machine Learning-based Spam Detection System using Natural Language Processing (NLP) to classify emails as spam or non-spam with high accuracy.

🔹 Features

✅ Dataset Preprocessing – Data cleaning, stopword removal, and text vectorization using CountVectorizer

✅ Machine Learning Model – Trained a Multinomial Naïve Bayes classifier for spam detection

✅ Performance Evaluation – Achieved 99% accuracy, evaluated with Confusion Matrix & Classification Report

✅ Model Deployment – Saves trained model & vectorizer for future predictions

⚙️ How the Code Works

1️⃣ Load the Dataset

The dataset (emails.csv) is loaded into a Pandas DataFrame.

2️⃣ Data Exploration & Preprocessing

Removes duplicates

Cleans text by removing punctuation and stopwords

Uses CountVectorizer to convert text into numerical features

Saves the vectorizer for future use

3️⃣ Train-Test Split

Splits the dataset into training (80%) and testing (20%) sets.

4️⃣ Train the Machine Learning Model

Uses Multinomial Naïve Bayes, which is effective for text classification

Trains the model and saves it

5️⃣ Model Evaluation
Makes predictions on the test set

Prints accuracy, classification report, and confusion matrix

Plots Confusion Matrix to visualize performance

🛠️ Tech Stack

Programming Language: Python

Libraries Used: Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib, NLTK

Machine Learning Model: Multinomial Naïve Bayes

Feature Extraction: CountVectorizer


📜 Results

🎯 Accuracy: ~99%

📊 Confusion Matrix & Classification Report for performance evaluation

🚀 Future Enhancements

🔹 Implement TF-IDF Vectorization for better feature extraction

🔹 Deploy as a web API for real-time spam classification

🔹 Integrate with email clients for real-world application

📌 Contributions Welcome! Feel free to fork, open issues, and submit pull requests.
