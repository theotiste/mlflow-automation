import mlflow
from mlflow.tracking import MlflowClient
from arize.pandas.logger import Client
import os

# Configurer la connexion Arize AI
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")  # Ajoute cette variable dans les secrets GitHub
SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")  # Ajoute cette variable aussi
arize_client = Client(space_key=SPACE_KEY, api_key=ARIZE_API_KEY)

# Connexion à MLflow
mlflow.set_tracking_uri("postgresql+psycopg2://postgres:mlflowpass@localhost:5432/mlflow_db")
client = MlflowClient()

# Récupérer le modèle enregistré dans MLflow
model_name = "theo_model"

# Vérifier si le modèle existe
models = client.search_registered_models()
if model_name not in [m.name for m in models]:
    print(f" Modèle {model_name} non trouvé dans MLflow !")
    exit(1)

# Récupérer la dernière version du modèle
latest_version = client.get_latest_versions(model_name, stages=["Production"])
if not latest_version:
    print(f" Aucun modèle en Production trouvé pour {model_name} !")
    exit(1)

model_version = latest_version[0].version
model_uri = f"models:/{model_name}/Production"

print(f" Envoi du modèle {model_name} (v{model_version}) à Arize AI...")

# Envoyer le modèle à Arize AI
arize_client.log_model(
    model_id=model_name,
    model_version=model_version,
    model_type="classification",  # Modifier selon le type de modèle (classification, regression, NLP)
    model_artifact_path=model_uri
)

print(f" Modèle {model_name} (v{model_version}) envoyé avec succès à Arize AI !")
