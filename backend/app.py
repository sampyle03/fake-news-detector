import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model_training.data_processor import custom_tokenizer
from model_training.model import RoundedKNeighborsRegressor
from tweet_scraper import scrape_tweet
from ensemble import predict

app = Flask(__name__) # Create a Flask app instance
CORS(app) # Enable CORS for all routes

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    tweet_url = data.get("url", "")
    #print(f"Received request to classify {tweet_url}")
    if not tweet_url:
        return jsonify({"error": "No url found"}), 400
    
    # Scrape the tweet and predict whether it is true or false
    prediction_data = predict(tweet_url)
    #print(prediction_data)
    raw_prediction, raw_individual_predictions, combined_score, shap_explanation = prediction_data

    # use combines score to calculate a confidence score
    confidence_raw = float(combined_score) - 0.45
    if confidence_raw < 0:
        confidence = abs(confidence_raw) / 0.45
    else: 
        confidence = abs(confidence_raw) / 0.55
    
    # convert confidence score to a label
    if confidence < 0.2:
        confidence = "Low"
    elif confidence < 0.4:
        confidence = "Medium"
    elif confidence < 0.6:
        confidence = "High"
    else:
        confidence = "Very High"

    # convert numpy values to floats to enable JSON serialisation
    prediction = float(raw_prediction)
    individual_predictions = [float(x) for x in raw_individual_predictions]

    # get top 3 most valuable tokens from SHAP values to explain the prediction to the user
    shap_tokens = shap_explanation.data[0]
    shap_values = shap_explanation.values.tolist()[0]
    top_3_tokens = [(0, "abc"), (0, "ijk"), (0, "xyz")]
    for token_no in range(len(shap_tokens)):
        for j in range(len(top_3_tokens)):
            if shap_values[token_no][1] < 0:
                if abs(shap_values[token_no][1])*2.66 > abs(top_3_tokens[j][0]) and len(shap_tokens[token_no]) > 4: # multiply fake tokens by 2.66 to support tokens suggesting fake news
                    top_3_tokens.insert(j, (shap_values[token_no][1]*2.66, shap_tokens[token_no]))
                    top_3_tokens.pop()
                    break
            else:
                if abs(shap_values[token_no][1]) > abs(top_3_tokens[j][0]) and len(shap_tokens[token_no]) > 4:
                    top_3_tokens.insert(j, (shap_values[token_no][1], shap_tokens[token_no]))
                    top_3_tokens.pop()
                    break

    if not prediction_data:
        return jsonify({"error": "Failed to predict"}), 500
    
    return jsonify({"prediction": prediction, "individual_predictions": individual_predictions, "confidence": confidence, "top_3_tokens": top_3_tokens})

if __name__ == "__main__":
    print("Flask app is running on port 5000")
    app.run(port=5000, debug=True)