FROM python:3.8.5-slim-buster

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier le contenu de ton projet dans l’image
COPY . /app
# Copier le fichier .env
COPY .env .env
# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Lancer l'application
CMD ["python3", "app.py"]
