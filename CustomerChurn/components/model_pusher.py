import sys

from CustomerChurn.cloud_storage.aws_storage import SimpleStorageService
from CustomerChurn.exception import ChurnException
from CustomerChurn.logger import logging
from CustomerChurn.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from CustomerChurn.entity.config_entity import ModelPusherConfig
from CustomerChurn.entity.s3_estimator import ChurnEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.churn_estimator = ChurnEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")

            self.churn_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)


            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise ChurnException(e, sys) from e