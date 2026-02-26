import pandas as pd
import spacy
import logging
import os

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    raise

def load_data(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format.")
        
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
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

def preprocess_file(input_path, output_path):
    df = load_data(input_path)
    df["clean_text"] = df["text"].apply(preprocess_text)
    df.to_csv(output_path, index=False)
    logging.info(f"Processed file saved to {output_path}")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_file("data/sample_input.csv", "data/processed.csv")