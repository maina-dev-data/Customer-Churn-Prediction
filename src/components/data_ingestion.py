import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data', 'processed', 'train.csv')
    test_data_path: str = os.path.join('data', 'processed', 'test.csv')
    raw_data_path: str = os.path.join('data', 'raw', 'Churn_Modelling.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion component")
        try:
            # Read the dataset
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info(f"Read the dataset successfully from {self.ingestion_config.raw_data_path}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            logging.info("Initiating train-test split")
            # We use stratified split based on target feature
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Exited'])
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
