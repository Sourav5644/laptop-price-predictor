
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import r2_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
import re
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def clean_data(self, df):
        df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
        df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
        return df

    
    def process_screen_resolution(self ,df):
        df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
        
        return df.drop(columns=['ScreenResolution', 'Inches'])


    def process_cpu(self ,df):
        df['cpu_name'] = df['Cpu'].apply(lambda x: ' '.join(x.split()[0:3]))
        def categorize_cpu(text):
            if text in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
                return text
            elif text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'
        df['cpu_name'] = df['cpu_name'].apply(categorize_cpu)
        return df.drop(columns=['Cpu'])


    def process_memory(self ,df):
        def extract_storage(mem_str, storage_type):
            parts = mem_str.split('+')
            for part in parts:
                if storage_type in part:
                    size = re.search(r'(\d+\.?\d*)', part)
                    size = float(size.group(1)) if size else 0
                    unit = 'TB' if 'TB' in part else 'GB'
                    return size * 1000 if unit == 'TB' else size
            return 0
        df['SSD'] = df['Memory'].apply(lambda x: extract_storage(x, 'SSD'))
        df['HDD'] = df['Memory'].apply(lambda x: extract_storage(x, 'HDD'))
        return df.drop(columns=['Memory'])

    
    def process_gpu(self ,df):
        df['gpu_brand'] = df['Gpu'].apply(lambda x: x.split()[0])
        return df.drop(columns=['Gpu'])

    
    def process_os(self ,df):
        def simplify_os(os):
            if os in ['macOS', 'Mac OS X']:
                return 'mac'
            elif 'Windows' in os:
                return 'windows'
            else:
                return 'other'
        df['os'] = df['OpSys'].apply(simplify_os)
        return df.drop(columns=['OpSys'])
    
    def drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if 'Unnamed: 0' in df.columns:
            df= df.drop(columns=['Unnamed: 0'])
        return df


    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self.clean_data(x)
            x = self.process_screen_resolution(x)
            x = self.process_cpu(x)
            x = self.process_gpu(x)
            x = self.process_memory(x)
            x = self.process_os(x)
            x = self.drop_id_column(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score
            logging.info(f"r2_Score for this model: {trained_model_r2_score}")

            best_model_r2_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing r2_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_r2_score = r2_score(y, y_hat_best_model)
                logging.info(f"r2_Score-Production Model: {best_model_r2_score}, r2_Score-New Trained Model: {trained_model_r2_score}")
            
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            result = EvaluateModelResponse(trained_model_r2_score=trained_model_r2_score,
                                           best_model_r2_score=best_model_r2_score,
                                           is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                                           difference=trained_model_r2_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
