import pandas as pd
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(data_path):

    # Load processed data
    df = pd.read_csv(data_path)
    logging.info("Processed data loaded successfully.")

    X = df["clean_text"]
    y = df["label"]

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    logging.info("Model training completed.")

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    logging.info(f"Model Accuracy: {accuracy}")

    # Save model + vectorizer
    joblib.dump((model, vectorizer), "models/sentiment_model.joblib")
    logging.info("Model saved successfully.")

    print("\nModel saved in models/sentiment_model.joblib")

if __name__ == "__main__":
    train_model("data/processed.csv")