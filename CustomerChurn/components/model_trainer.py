import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf  import ModelFactory
from sklearn.preprocessing import LabelEncoder

from CustomerChurn.exception import ChurnException
from CustomerChurn.logger import logging
from CustomerChurn.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from CustomerChurn.entity.config_entity import ModelTrainerConfig
from CustomerChurn.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from CustomerChurn.entity.estimator import ChurnModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config


    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
            try:
                logging.info("Using neuro_mf to get best model object and report")
                model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
                
                # Vérifier si le fichier de configuration est chargé correctement
                logging.info(f"Loading model configuration from {self.model_trainer_config.model_config_file_path}")
                model_config = read_yaml_file(self.model_trainer_config.model_config_file_path)
                logging.info(f"Model configuration loaded: {model_config}")
                
                x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

                # Essayer d'obtenir le meilleur modèle
                best_model_detail = model_factory.get_best_model(
                    X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
                )
                
                # Vérification de la présence du modèle
                if best_model_detail is None:
                    raise ValueError("No best model found. Check the configuration or the model selection process.")

                model_obj = best_model_detail.best_model

                y_pred = model_obj.predict(x_test)
                
                # Convertir les labels 'Yes' et 'No' en 1 et 0 pour y_test et y_pred
                label_encoder = LabelEncoder()
                
                # Si y_test contient des chaînes (par exemple, 'Yes', 'No'), les transformer en 1 et 0
                y_test = label_encoder.fit_transform(y_test)  # Transforme y_test de 'Yes', 'No' en 1, 0
                
                # Si y_pred contient également des valeurs de type chaîne, transformez-les aussi
                y_pred = label_encoder.transform(y_pred)  # Utilise le même encodeur pour transformer y_pred

                # Calcul des métriques
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
                
                return best_model_detail, metric_artifact
            except Exception as e:
                logging.error(f"Error in get_model_object_and_report: {str(e)}")
                raise ChurnException(e, sys) from e


    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            Churn_model = ChurnModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logging.info("Created churn model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, Churn_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ChurnException(e, sys) from e