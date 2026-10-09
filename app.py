from flask import Flask, request, render_template
import os
import sys

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        # Map HTML form data to our CustomData object
        data = CustomData(
            credit_score=int(request.form.get('credit_score')),
            geography=request.form.get('geography'),
            gender=request.form.get('gender'),
            age=int(request.form.get('age')),
            tenure=int(request.form.get('tenure')),
            balance=float(request.form.get('balance')),
            num_of_products=int(request.form.get('num_of_products')),
            has_cr_card=int(request.form.get('has_cr_card')),
            is_active_member=int(request.form.get('is_active_member')),
            estimated_salary=float(request.form.get('estimated_salary'))
        )
        
        pred_df = data.get_data_as_data_frame()
        logging.info("DataFrame created from web input:")
        logging.info(pred_df)

        predict_pipeline = PredictPipeline()
        preds, probs = predict_pipeline.predict(pred_df)

        result_text = "High Risk of Churn" if preds[0] == 1 else "Low Risk of Churn (Retained)"
        prob_text = f"{probs[0] * 100:.2f}% probability of churning"

        return render_template('index.html', results=result_text, probability=prob_text, prediction=int(preds[0]))

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('index.html', results="Error occurred during prediction. Please check logs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
