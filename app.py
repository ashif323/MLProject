from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfiq
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.mlproject.components.model_trainer import  ModelTrainer,ModelTrainerConfig



import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_confiq=data_ingestion_confiq()
        data_ingestion=DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()
        # data_transformation_config=data_transformation_config()
        data_transformatin=DataTransformation()
        train_arr,test_arr,_ =data_transformatin.initiate_data_transormation(train_data_path,test_data_path)

        ## Model Training

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        

    except Exception as e:
        logging.info("custom exception")
        raise CustomException(e,sys)