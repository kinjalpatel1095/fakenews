from flask import Flask, request, jsonify, make_response
from google.cloud import storage
import joblib
from twilio.twiml.messaging_response import MessagingResponse
import pandas as pd
from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app, origins=['https://us-central1-fakenews-398008.cloudfunctions.net/predict', 'http://34.125.135.184/'])


BUCKET_NAME = "fake_news_detection_new"
CLASS_NAMES = ['Fake News', 'Real News']

model = None

def download_blob(bucket_name, source_blob_name, destination_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_name)

@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict(request):
    global model
    if model is None:
        download_blob(BUCKET_NAME,
                      "models/pipeline.sav",
                      "/tmp/pipeline.sav")
        model = joblib.load('/tmp/pipeline.sav')

    # Access form data using request.form
    data = request.form['text']  # Assuming 'text' is the name of the form field

    # Make predictions with the loaded model
    predictions = model.predict([data])  # Assuming your model expects a list of data points

    # Map predictions to class names
    class_names = [CLASS_NAMES[label] for label in predictions]



    response = jsonify({'predictions': class_names})
    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/sms',methods=['POST'])
def sms(request):

    response = MessagingResponse()

    global model
    if model is None:
        download_blob(BUCKET_NAME,
                      "models/pipeline.sav",
                      "/tmp/pipeline.sav")
        model = joblib.load('/tmp/pipeline.sav')

    predictions = model.predict([request.values.get('Body').lower()])

    if predictions == 0 :
        response.message("This is fake news!")
    elif predictions == 1:
        response.message('This is true news')
    message_content = str(response)
    return message_content



if __name__ == '__main__':
    app.run(debug=True)