import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from CustomerChurn.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from CustomerChurn.entity.config_entity import DataTransformationConfig
from CustomerChurn.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from CustomerChurn.exception import ChurnException
from CustomerChurn.logger import logging
from CustomerChurn.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from CustomerChurn.entity.estimator import TargetValueMapping
from sklearn.preprocessing import LabelEncoder




class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ChurnException(e, sys)

    def transf(self, tenure):
        """
        Transforms the 'tenure' column into categories.
        """
        if tenure <= 24:
            return '0 - 24 months'
        elif tenure <= 36:
            return '24 - 36 months'
        elif tenure <= 48:
            return '36 - 48 months'
        elif tenure <= 60:
            return '48 - 60 months'
        else:
            return '> 60 months'

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ChurnException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object.
        """
        logging.info("Entered get_data_transformer_object method")

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            return preprocessor

        except Exception as e:
            raise ChurnException(e, sys) from e

   
    def initiate_data_transformation(self) -> DataTransformationArtifact:
            """
            Initiates the data transformation process.
            """
            try:
                if self.data_validation_artifact.validation_status:
                    logging.info("Starting data transformation")
                    preprocessor = self.get_data_transformer_object()

                    # Read data
                    train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                    test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                    if TARGET_COLUMN not in train_df.columns:
                        raise KeyError(f"'{TARGET_COLUMN}' column is missing from the training data.")
                    
                    logging.info("Got train features and test features of Training dataset")

                    # Nettoyage de la colonne 'TotalCharges' avant transformation
                    train_df['TotalCharges'] = train_df['TotalCharges'].replace([' ', ''], np.nan)
                    train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
                    train_df.dropna(subset=['TotalCharges'], inplace=True)
                    train_df['TotalCharges'] = train_df['TotalCharges'].astype(float)

                    logging.info(f"Remaining missing values: {train_df['TotalCharges'].isnull().sum()}")

                    # Ajouter une colonne de catégorie pour 'tenure'
                    train_df['tenure_category'] = train_df['tenure'].apply(self.transf)

                    # Suppression des colonnes inutiles
                    drop_cols = self._schema_config['drop_columns']
                    input_feature_train_df = drop_columns(df=train_df.drop(columns=[TARGET_COLUMN], axis=1), cols=drop_cols)

                    # Transformation de la cible (mapping des valeurs)
                    label_encoder = LabelEncoder()
                    target_feature_train_df = label_encoder.fit_transform(train_df[TARGET_COLUMN])

                    # Appliquer le préprocesseur
                    logging.info("Applying preprocessing object on train and test dataframes")
                    input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                    # Transformation des données de test
                    test_df['TotalCharges'] = test_df['TotalCharges'].replace([' ', ''], np.nan)
                    test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
                    test_df.dropna(subset=['TotalCharges'], inplace=True)
                    test_df['tenure_category'] = test_df['tenure'].apply(self.transf)

                    input_feature_test_df = drop_columns(df=test_df.drop(columns=[TARGET_COLUMN], axis=1), cols=drop_cols)
                    target_feature_test_df = label_encoder.transform(test_df[TARGET_COLUMN])  # Transformation des labels de test

                    input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                    # Appliquer SMOTEENN sur les jeux de données d'entraînement et de test
                    logging.info("Applying SMOTEENN on Training dataset")
                    smt = SMOTEENN(sampling_strategy="minority")
                    input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                    input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)

                    logging.info("Created train and test arrays")

                    # Sauvegarder les résultats transformés et le préprocesseur
                    save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                    save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=np.c_[input_feature_train_final, target_feature_train_final])
                    save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=np.c_[input_feature_test_final, target_feature_test_final])

                    # Création de l'objet artifact de transformation des données
                    data_transformation_artifact = DataTransformationArtifact(
                        transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                        transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                        transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                    )

                    return data_transformation_artifact

                else:
                    raise Exception(self.data_validation_artifact.message)

            except Exception as e:
                raise ChurnException(e, sys) from e