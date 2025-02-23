import uuid

from flask import Flask, request, session
import joblib
import pandas as pd


MODEL_ID = 'hist_gb_202502221059409325970600'  # switch out for desired model after training
MODEL = model = joblib.load(f'modeling/model_results/{MODEL_ID}/model/model.pkl')
app = Flask(__name__)
app.secret_key = 'my_precious'  # ideally, this is pulled from a password vault


@app.before_request
def set_session_uid():
    """
    Sets a UID for each session
    """
    uid = str(uuid.uuid4())
    session["uid"] = uid


@app.route("/", methods=["POST", "GET"])
def home():
    """
    Home route that will confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/health", methods=["POST", "GET"])
def health():
    """
    Health check endpoint that wil confirm if the app is healthy
    """
    return "app is healthy"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions
    """
    input_df = pd.DataFrame.from_dict([request.json], orient='columns')
    prediction = round(MODEL.predict_proba(input_df)[0][1], 3)

    print(session.get("uid"))
    print(input_df)
    print(prediction)
    print(MODEL_ID)
    print()

    return {
        'prediction': prediction
    }


if __name__ == "__main__":
    app.run()
