# Manufacturing Downtime Prediction API

This repository provides a FastAPI-based RESTful API to predict machine downtime based on manufacturing metrics. The project includes synthetic data generation, model training, and prediction endpoints.

## Features

- **Synthetic Data Generation:** Generate realistic manufacturing datasets for training
- **Model Training:** Train a logistic regression model with evaluation metrics
- **Prediction:** Predict machine downtime based on sensor metrics

## Table of Contents

1. [Setup Instructions](#setup-instructions)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
2. [Dataset Generation](#dataset-generation)
3. [Code Structure](#code-structure)
   - [Synthetic Data Generation](#synthetic-data-generation)
   - [FastAPI Implementation](#fastapi-implementation)
4. [Testing Locally](#testing-locally)
   - [Using Postman](#using-postman)
   - [Using cURL](#using-curl)

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Pip package manager

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Generate dataset:
```bash
python random_data_creation.py
```
4. Start server:
```bash
uvicorn main:app --reload
```

## Dataset Generation

Use `random_data_creation.py` to create synthetic data with customizable parameters:

- `n_samples`: Dataset size
- `n_features`: Number of input features
- `n_classes`: Output classes (0 or 1)
- `class_sep`: Class separation factor
- `flip_y`: Noise percentage

## Code Structure

### Synthetic Data Generation
`random_data_creation.py`:
- Creates synthetic data using scikit-learn
- Applies realistic feature scaling
- Outputs to `data.csv`

### FastAPI Implementation
`main.py` contains three endpoints:

#### 1. Upload Data
```http
POST /upload
```
Response:
```json
{
    "message": "File data.csv uploaded successfully"
}
```

#### 2. Train Model
```http
GET /train
```
Response:
```json
{
    "train_loss": 0.5623,
    "test_loss": 0.5891,
    "accuracy": 0.8475,
    "precision": 0.8353,
    "recall": 0.8745,
    "f1_score": 0.8544,
    "auc_roc": 0.9123
}
```

#### 3. Predict Downtime
```http
POST /predict
```
Request:
```json
{
    "Temperature": 75.5,
    "Run_Time": 250.0,
    "Pressure": 85.0,
    "Humidity": 60.0
}
```

Response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.89
}
```

## Testing Locally

### Using Postman

1. Upload Data
   - POST `http://127.0.0.1:8000/upload`
   - Body: form-data with CSV file

2. Train Model
   - GET `http://127.0.0.1:8000/train`

3. Predict Downtime
   - POST `http://127.0.0.1:8000/predict`
   - Body: raw JSON

### Using cURL

1. Upload Data:
```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@data.csv"
```

2. Train Model:
```bash
curl -X GET "http://127.0.0.1:8000/train"
```

3. Predict Downtime:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Temperature": 75.5,
           "Run_Time": 250.0,
           "Pressure": 85.0,
           "Humidity": 60.0
         }'
```