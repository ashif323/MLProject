from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfiq


import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_confiq=data_ingestion_confiq()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)