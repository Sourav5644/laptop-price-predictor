import sys
from src.entity.config_entity import LaptopPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class LaptopData:
    def __init__(
        self,
        Company,
        TypeName,
        Ram,
        Weight,
        Touchscreen,
        IPS,
        cpu_name,
        gpu_brand,
        os,
        SSD,
        HDD
    ):
        """
        Laptop Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Company = Company
            self.TypeName = TypeName
            self.Ram = Ram
            self.Weight = Weight
            self.Touchscreen = Touchscreen
            self.IPS = IPS
            self.cpu_name = cpu_name
            self.gpu_brand = gpu_brand
            self.os = os
            self.SSD = SSD
            self.HDD = HDD

        except Exception as e:
            raise MyException(e, sys) from e

    def get_laptop_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame with a single row of input features
        """
        try:
            laptop_input_dict = self.get_laptop_data_as_dict()
            return DataFrame([laptop_input_dict])  # Wrap dict in a list to create a single-row DataFrame
        except Exception as e:
            raise MyException(e, sys) from e

    def get_laptop_data_as_dict(self):
        """
        This function returns a dictionary of all the input features
        """
        logging.info("Entered get_laptop_data_as_dict method of LaptopData class")

        try:
            input_data = {
                "Company": self.Company,
                "TypeName": self.TypeName,
                "Ram": self.Ram,
                "Weight": self.Weight,
                "Touchscreen": self.Touchscreen,
                "IPS": self.IPS,
                "cpu_name": self.cpu_name,
                "gpu_brand": self.gpu_brand,
                "os": self.os,
                "SSD": self.SSD,
                "HDD": self.HDD
            }

            logging.info("Created laptop data dict")
            logging.info("Exited get_laptop_data_as_dict method of LaptopData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class LaptopDataRegressor:
    def __init__(
        self,
        prediction_pipeline_config: LaptopPredictorConfig = LaptopPredictorConfig(),
    ) -> None:
        """
        Initializes the LaptopDataRegressor with model config
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        Predicts the price using the trained model
        """
        try:
            logging.info("Entered predict method of LaptopDataRegressor class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            logging.info("Prediction completed")
            return result

        except Exception as e:
            raise MyException(e, sys)
