import os
import sys

import numpy as np
import pandas as pd
from CustomerChurn.entity.config_entity import ChurnPredictorConfig
from CustomerChurn.entity.s3_estimator import ChurnEstimator
from CustomerChurn.exception import ChurnException
from CustomerChurn.logger import logging
from CustomerChurn.utils.main_utils import read_yaml_file
from pandas import DataFrame

import os
import sys
import numpy as np
import pandas as pd
from CustomerChurn.entity.config_entity import ChurnPredictorConfig
from CustomerChurn.entity.s3_estimator import ChurnEstimator
from CustomerChurn.exception import ChurnException
from CustomerChurn.logger import logging
from pandas import DataFrame


class ChurnInputData:
    def __init__(self,
                 gender,
                 SeniorCitizen,
                 Partner,
                 Dependents,
                 tenure,
                 PhoneService,
                 MultipleLines,
                 InternetService,
                 OnlineSecurity,
                 OnlineBackup,
                 DeviceProtection,
                 TechSupport,
                 StreamingTV,
                 StreamingMovies,
                 Contract,
                 PaperlessBilling,
                 PaymentMethod,
                 MonthlyCharges,
                 TotalCharges,
                 ):
        try:
            self.gender = gender
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure = tenure
            self.PhoneService = PhoneService
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges

        except Exception as e:
            raise ChurnException(e, sys) from e

    def get_input_data_frame(self) -> DataFrame:
        """
        Convert input data to a DataFrame.
        """
        try:
            input_dict = self.get_input_data_as_dict()
            return DataFrame(input_dict)
        except Exception as e:
            raise ChurnException(e, sys) from e

    def get_input_data_as_dict(self):
        """
        Return the input as a dictionary
        """
        try:
            input_data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }
            return input_data
        except Exception as e:
            raise ChurnException(e, sys) from e


class ChurnPredictor:
    def __init__(self, prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise ChurnException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        """
        Predict churn from input dataframe using trained model.
        """
        try:
            logging.info("Loading model from S3")
            model = ChurnEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            return result

        except Exception as e:
            raise ChurnException(e, sys)
