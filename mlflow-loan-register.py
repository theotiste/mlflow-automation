import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 🔹 Définir PostgreSQL comme backend MLflow
mlflow.set_tracking_uri("postgresql+psycopg2://postgres:mlflowpass@localhost:5432/mlflow_db")

# 🔹 Définir le nom du modèle dans MLflow Registry
MODEL_NAME = "my_model"

# 🔹 Initialiser le client MLflow
client = MlflowClient()

# 🔹 Récupérer le dernier run de l'expérience 'recherche-model' ou 'optimisation-model'
experiment_name = "recherche-model"  # Modifier selon le modèle à enregistrer
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id, order_by=["metrics.accuracy DESC"], max_results=1)

    if runs:
        best_run = runs[0]  # Sélectionner le meilleur run
        run_id = best_run.info.run_id

        print(f"📌 Enregistrement du modèle depuis le run {run_id}")

        # 🔹 Enregistrer le modèle dans MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, MODEL_NAME)

        print(f"✅ Modèle {MODEL_NAME} enregistré avec succès, version : {model_version.version}")

        # 🔹 Passer automatiquement à "Production"
        client.transition_model_version_stage(name=MODEL_NAME, version=model_version.version, stage="Production")

        print(f"🚀 Modèle {MODEL_NAME} (v{model_version.version}) est maintenant en Production !")
    else:
        print("⚠️ Aucun run trouvé dans l'expérience. Vérifie que l'entraînement a bien été exécuté.")
else:
    print("⚠️ L'expérience spécifiée n'existe pas dans MLflow.")

