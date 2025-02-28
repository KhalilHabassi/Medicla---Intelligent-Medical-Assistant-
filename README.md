# Medicla - Assistant Médical Intelligent

<img src="Medical_assistant.jpeg" alt="Medicla Logo" width="300" />


## À propos

Medicla est un assistant médical intelligent basé sur un système RAG (Retrieval Augmented Generation) qui permet aux utilisateurs de poser des questions médicales en langage naturel et d'obtenir des réponses précises basées sur des données médicales fiables. Le système utilise une combinaison de techniques de récupération d'information et de génération de texte pour fournir des réponses pertinentes et contextuelles.

## Fonctionnalités

- 🔍 Interface de chat intuitive pour poser des questions médicales
- 💬 Traitement du langage naturel pour comprendre les requêtes des utilisateurs
- 📊 Génération de réponses basées sur des données médicales fiables
- 🌐 Support multilingue (Anglais, Français, Arabe, Espagnol)
- 🎥 Génération de réponses audiovisuelles
- 🧠 Raffinement des réponses avec Gemini Pro pour une meilleure qualité
- 📚 Accès à plusieurs sources de données (Présentations, MedQuAD)

## Architecture

Le projet est construit autour de plusieurs composants clés :

1. **API FastAPI** : Gère les requêtes des utilisateurs et communique avec la base de données vectorielle
2. **Interface Streamlit** : Interface utilisateur du chatbot
3. **Base de données PostgreSQL** : Stocke les embeddings vectoriels et les données médicales
4. **Modèle d'embedding** : Utilise SentenceTransformer pour générer des représentations vectorielles des questions
5. **Gemini Pro** : Raffinement des réponses générées

## Structure du projet

```
medicla/
├── api.py                    # API FastAPI pour le traitement des requêtes
├── app.py                    # Interface utilisateur Streamlit
├── ingest.py                 # Fonctions générales pour l'ingestion de données
├── ingest_base_embedding.py  # Script pour créer et ingérer la table qa_table
├── retrieve.py               # Fonctions pour récupérer des documents pertinents
├── audiovisuel.py            # Module pour générer des réponses audiovisuelles
├── config.py                 # Configuration du projet
└── .env                      # Variables d'environnement
```

## Configuration requise

- Python 3.10+
- PostgreSQL avec extension pgvector
- Clé API Gemini Pro

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/KhalilHabassi/Medicla---Intelligent-Medical-Assistant-.git
cd medicla
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement dans un fichier `.env` :
```
GEMINI_API_KEY=votre_clé_api_gemini
DB_PASSWORD=votre_mot_de_passe_db
```

4. Assurez-vous que PostgreSQL est installé et configuré avec l'extension pgvector

## Préparation de la base de données

1. Créez la base de données et les tables nécessaires :
```bash
psql -U postgres -c "CREATE DATABASE gen_ai_db;"
psql -U postgres -d gen_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

2. Ingérez les données dans la base :
```bash
python ingest_base_embedding.py
```

## Utilisation

1. Démarrez l'API :
```bash
uvicorn api:app --host 0.0.0.0 --port 8181
```

2. Lancez l'interface utilisateur Streamlit :
```bash
streamlit run app.py
```

3. Accédez à l'interface via votre navigateur à l'adresse : `http://localhost:8501`

## API Endpoints

- **POST /answer_from_table** : Récupère une réponse à une question médicale
- **GET /get_sources** : Récupère les sources pertinentes pour une question donnée

## Fonctionnement du système RAG

1. **Récupération (Retrieval)** : 
   - La question de l'utilisateur est convertie en embedding vectoriel
   - Le système recherche les documents les plus similaires dans la base de données vectorielle
   - Les documents pertinents sont récupérés en fonction de leur score de similarité

2. **Augmentation (Augmentation)** :
   - Les documents récupérés sont utilisés comme contexte pour générer une réponse
   - Gemini Pro raffine la réponse pour améliorer sa qualité et sa précision

3. **Génération (Generation)** :
   - Une réponse finale est générée et présentée à l'utilisateur
   - Une version audiovisuelle de la réponse est également créée

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

© 2025 Medicla. Tous droits réservés.