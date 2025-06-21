from CustomerChurn.logger import logging 
from CustomerChurn.exception import ChurnException
import sys
# logging.info("Démarrage du programme.")
# logging.warning("Ceci est un avertissement.")
# logging.error("Une erreur s'est produite.")
# try:
#     a=2/0
# except Exception as e:
#     raise ChurnException(e,sys)
from CustomerChurn.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()