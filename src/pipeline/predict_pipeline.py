import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("models", "model.pkl")
            preprocessor_path = os.path.join("models", "preprocessor.pkl")
            
            logging.info("Loading preprocessor and model...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Predicting...")
            preds = model.predict(data_scaled)
            probs = model.predict_proba(data_scaled)[:, 1]
            return preds, probs
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 credit_score: int,
                 geography: str,
                 gender: str,
                 age: int,
                 tenure: int,
                 balance: float,
                 num_of_products: int,
                 has_cr_card: int,
                 is_active_member: int,
                 estimated_salary: float):
        
        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_of_products = num_of_products
        self.has_cr_card = has_cr_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.credit_score],
                "Geography": [self.geography],
                "Gender": [self.gender],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.num_of_products],
                "HasCrCard": [self.has_cr_card],
                "IsActiveMember": [self.is_active_member],
                "EstimatedSalary": [self.estimated_salary],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
