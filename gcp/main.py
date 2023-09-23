from fastapi import FastAPI, HTTPException,Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import storage
import joblib
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific allowed origins
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

BUCKET_NAME = "fake_news_detection_new"
CLASS_NAMES = ['Fake News', 'Real News']

model = None

class PredictionRequest(BaseModel):
    name: str = None

def download_blob(bucket_name, source_blob_name, destination_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_name)

@app.post('/predict_n')
async def predict_n(item: PredictionRequest= Body(..., embed=True), status_code=201):
    global model
    if model is None:
        download_blob(BUCKET_NAME,
                      "models/pipeline.sav",
                      "/tmp/pipeline.sav")
        model = joblib.load('/tmp/pipeline.sav')

    # Access form data using request.text

    # Make predictions with the loaded model
    # predictions = model.predict([item_dict.text])

    # Map predictions to class names
    # class_names = [CLASS_NAMES[label] for label in predictions]

    # return {'predictions': class_names}
    return item

@app.post('/sms')
def sms(request: dict):
    response = MessagingResponse()

    global model
    if model is None:
        download_blob(BUCKET_NAME,
                      "models/pipeline.sav",
                      "/tmp/pipeline.sav")
        model = joblib.load('/tmp/pipeline.sav')

    predictions = model.predict([request.get('Body').lower()])

    if predictions == 0:
        response.message("This is fake news!")
    elif predictions == 1:
        response.message('This is true news')

    message_content = str(response)
    return message_content

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
