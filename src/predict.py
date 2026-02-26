import pandas as pd
import joblib
import logging
import spacy

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded for prediction.")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    raise

def preprocess_text(text):
    try:
        doc = nlp(str(text).lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        return " ".join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ""

def batch_predict(input_path, output_path):

    try:
        # Load model + vectorizer
        model, vectorizer = joblib.load("models/sentiment_model.joblib")
        logging.info("Model loaded successfully.")

        # Load input data
        df = pd.read_csv(input_path)
        logging.info(f"Input data loaded from {input_path}")

        # Preprocess text
        df["clean_text"] = df["text"].apply(preprocess_text)

        # Transform using trained vectorizer
        X_vec = vectorizer.transform(df["clean_text"])

        # Predict
        df["prediction"] = model.predict(X_vec)

        # Save output
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

        print("Batch prediction completed successfully.")

    except Exception as e:
        logging.error(f"Error during batch prediction: {e}")
        raise

if __name__ == "__main__":
    batch_predict("data/sample_input.csv", "data/output.csv")