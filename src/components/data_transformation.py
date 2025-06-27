import sys
import numpy as np
import pandas as pd
import re
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)



    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
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
        drop_col = self._schema_config['drop_columns']
        
        df = df.drop(drop_col, axis=1)
        return df

    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            logging.info("Transformers Initialized: StandardScaler")

            # Load schema configurations
            num_features = self._schema_config['num_columns']
            mm_columns = self._schema_config['one_hot_encoding_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("one_hot_encoder", one_hot_encoder, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating laptop data transformation process")
            
            # Process training data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            train_df = self.clean_data(train_df)
            train_df = self.process_screen_resolution(train_df)
            train_df = self.process_memory(train_df)
            train_df = self.process_cpu(train_df)
            train_df = self.process_gpu(train_df)
            train_df = self.process_os(train_df)
            train_df = self.drop_id_column(train_df)

            X_train = train_df.drop(columns=[self._schema_config['target_column']])
            y_train = train_df[self._schema_config['target_column']]
            logging.info(f"Original Feature Names in train: {list(X_train.columns)}")

            # Process test data
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            test_df = self.clean_data(test_df)
            test_df = self.process_screen_resolution(test_df)
            test_df = self.process_memory(test_df)
            test_df = self.process_cpu(test_df)
            test_df = self.process_gpu(test_df)
            test_df = self.process_os(test_df)
            test_df = self.drop_id_column(test_df)

            X_test = test_df.drop(columns=[self._schema_config['target_column']])
            y_test = test_df[self._schema_config['target_column']]
            logging.info(f"Original Feature Names in test: {list(X_test.columns)}")

            # Create and fit transformer on training data
            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)

            
            
            # Transform test data using same transformer
            X_test_transformed = preprocessor.transform(X_test)

            # Create final arrays
            train_final_array = np.c_[X_train_transformed, np.array(y_train)]
            test_final_array = np.c_[X_test_transformed, np.array(y_test)]

            # Save objects and arrays
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_final_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_final_array)

            logging.info("Laptop data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys)