from fastapi import FastAPI, UploadFile, HTTPException, Request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
from pydantic import BaseModel
app = FastAPI()

class PredictRequest(BaseModel):
    Temperature: float
    Run_Time: float
    Pressure: float
    Humidity: float

df1 = None
model = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload_data(file: UploadFile):
    global df1
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_name = file.filename
    df1 = pd.read_csv(file.file)
    print(df1.head())
    return {"message": f"File {file_name} uploaded successfully"}

@app.get("/train")
def train_data():
    global df1
    global model
    if df1 is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a dataset first.")

    # Preprocessing the data
    Y = df1['Downtime_Flag']
    X = df1.drop(columns=['Downtime_Flag', 'Machine_ID'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculating metrics
    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_test_proba)

    return {
        "train_loss": round(train_loss, 4),
        "test_loss": round(test_loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc_roc, 4)
    }


@app.post("/predict")
async def predict_data(input_data: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Train the model before making predictions.")
    
    data = pd.DataFrame([input_data.dict()])
    
    prediction = model.predict(data)
    confidence = model.predict_proba(data).max(axis=1)[0]
    downtime = "Yes" if prediction[0] == 1 else "No"
    
    return {
        "Downtime": downtime,
        "Confidence": round(confidence, 2)
    }