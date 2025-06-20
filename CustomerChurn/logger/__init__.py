import os
import logging
from datetime import datetime

# Obtenir le chemin du dossier racine du projet
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # <- monte 2 niveaux

# Créer le dossier logs à la racine
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Nom du fichier log
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configuration du logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

# Optionnel : affichage dans le terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)
