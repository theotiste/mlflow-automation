import subprocess
import sys

# Fonction pour installer un module si non installé
def install_module(package):
    try:
        __import__(package)
    except ImportError:
        print(f" Installation du module {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#  Vérification et installation des modules nécessaires
install_module("mlflow")
install_module("pandas")
install_module("sqlalchemy")
install_module("psycopg2")
install_module("ace_tools_open")
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("postgresql+psycopg2://postgres:mlflowpass@localhost:5432/mlflow_db")
client = MlflowClient()
#  Enregistrer une version du modèle
experiment_name = "recherche-model"
model_name = "theo_model"

experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)

    if runs:
        best_run = runs[0]
        run_id = best_run.info.run_id

        print(f" Enregistrement du modèle depuis le run {run_id}")

        #  Enregistrer le modèle dans le Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        print(f" Modèle {model_name} enregistré avec succès, version : {model_version.version}")

        #  Passer automatiquement en "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        print(f" Modèle {model_name} (v{model_version.version}) est maintenant en Production !")
    else:
        print(" Aucun run trouvé dans l'expérience.")
else:
    print(" L'expérience spécifiée n'existe pas dans MLflow.")
import pandas as pd
import mlflow
from sqlalchemy import create_engine

#  Connexion MLflow PostgreSQL
mlflow.set_tracking_uri("postgresql+psycopg2://postgres:mlflowpass@localhost:5432/mlflow_db")

experiment_name = "mlflow-automation"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment:
    mlflow.set_experiment(experiment_name)
    print(f" Expérience {experiment_name} trouvée, ID : {experiment.experiment_id}")
else:
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print(f" Expérience {experiment_name} créée avec ID : {experiment_id}")

#  Connexion à PostgreSQL via SQLAlchemy
engine = create_engine("postgresql+psycopg2://postgres:mlflowpass@localhost:5432/mlflow_db")
conn = engine.connect()

#  Lire les tables PostgreSQL
tables = pd.read_sql("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public';", engine)
experiments_df = pd.read_sql("SELECT * FROM experiments", engine)
registered_models_df = pd.read_sql("SELECT * FROM registered_models", engine)
model_versions_df = pd.read_sql("SELECT * FROM model_versions", engine)
metrics_df = pd.read_sql("SELECT * FROM metrics", engine)
latest_metrics_df = pd.read_sql("SELECT * FROM latest_metrics", engine)
runs_df = pd.read_sql("SELECT * FROM runs", engine)

#  Afficher les résultats dans Jupyter
import ace_tools_open as tools

tools.display_dataframe_to_user(name="Experiments Table", dataframe=experiments_df)
tools.display_dataframe_to_user(name="Registered_Models Table", dataframe=registered_models_df)
tools.display_dataframe_to_user(name="Model_Versions Table", dataframe=model_versions_df)
tools.display_dataframe_to_user(name="Metrics Table", dataframe=metrics_df)
tools.display_dataframe_to_user(name="Latest Metrics Table", dataframe=latest_metrics_df)
tools.display_dataframe_to_user(name="Runs Table", dataframe=runs_df)

# Fermer la connexion
conn.close()
