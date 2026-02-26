# Production-Ready Sentiment Analysis Pipeline

## Overview

This project implements a robust sentiment analysis pipeline ready for deployment.

It includes:
- CSV/JSON ingestion
- Text preprocessing using spaCy (tokenization and lemmatization)
- TF-IDF feature extraction
- Logistic Regression classifier
- Batch prediction service
- Logging and error handling
- Docker containerization

---

## Project Structure

data/
    sample_input.csv

src/
    ingest_preprocess.py
    train.py
    predict.py

Dockerfile
docker-compose.yml
requirements.txt

---

## Local Execution

### Install Dependencies

pip install -r requirements.txt  
python -m spacy download en_core_web_sm  

### Preprocess Data

python src/ingest_preprocess.py  

### Train Model

python src/train.py  

### Run Batch Prediction

python src/predict.py  

---

## Run with Docker

### Build

docker compose build  

### Run

docker compose up  

Output will be saved in:

data/output.csv  

---

## Logging

Logs are stored in:

logs/app.log  

---

## Summary

This project demonstrates an end-to-end deployable NLP pipeline including preprocessing, model training, batch inference, and containerized deployment.
