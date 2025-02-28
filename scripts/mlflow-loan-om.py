#!/usr/bin/env python
# coding: utf-8

# In[1]:


#version de python
import sys
sys.version


# In[2]:


#version de pycaret
import pycaret
print(pycaret.__version__)


# In[3]:


#version de mlflow
import mlflow
print(mlflow.__version__)


# In[4]:


import mlflow
import time
#importation de l'outil d'expérimentation de pycaret
from pycaret.classification import ClassificationExperiment
# Configuration MLflow
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("mysql+pymysql://root:X9605man9801pap%40%21theo@localhost:3306/mlflow_db?charset=utf8mb4&useSSL=false&allowPublicKeyRetrieval=true")

mlflow.set_experiment("mlflow-automation")


# In[5]:


#chargement des données 
import pandas as pd
data = pd.read_csv(r"C:\Users\lucie\mlflow-automation\notebooks\Loan_Data.csv")
data.info()


# In[6]:


import mlflow
from pycaret.classification import ClassificationExperiment
import pandas as pd

# S'assurer que toute session MLflow précédente est fermée
if mlflow.active_run():
    mlflow.end_run()

# Session secondaire avec suivi MLflow
with mlflow.start_run(run_name="optimisation_modele"):
    
    #  Initialisation et configuration de la session secondaire
    start_time = time.time()
    session_bis = ClassificationExperiment()
    session_bis.setup(data, normalize=True, target='default', train_size=0.7,
                      data_split_stratify=True, fold=5, session_id=1,
                      log_experiment=True, experiment_name="optimisation_modele")
    duration = time.time() - start_time
    mlflow.log_param("duration_setup", duration)

    # Comparaison des modèles
    model_list = ['lr', 'nb', 'dt', 'rf', 'svm', 'lda']
    model_durations = {}

    for model in model_list:
        start_time = time.time()
        mdl = session_bis.create_model(model)
        duration = time.time() - start_time
        model_durations[model] = duration
        mlflow.log_param(f"duration_{model}", duration)

    #  Sélection du meilleur modèle (hors boucle)
    best_model = session_bis.compare_models(sort='Accuracy', include=model_list, verbose=True)
    print("Durées d'entraînement par modèle :", model_durations)

    # Vérifier le type de modèle sélectionné
    model_name = str(best_model)
    print(f"Modèle sélectionné : {model_name}")

    # Définir une grille d'hyperparamètres adaptée
    if "LogisticRegression" in model_name:
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # Paramètres valides pour LogisticRegression
    elif "DecisionTree" in model_name:
       param_grid = {'min_samples_split': [2, 10, 20], 'max_depth': [5, 10, None]}
    elif "RandomForest" in model_name:
       param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
    else:
      param_grid = {}  # Si le modèle sélectionné n'est pas connu, ne pas tuner.

    #  Lancer l'optimisation des hyperparamètres uniquement si une grille est définie
    if param_grid:
       start_time = time.time()
       tuned_mybest, essais = session_bis.tune_model(best_model, optimize="Accuracy",
                                                  choose_better=True,
                                                  custom_grid=param_grid,
                                                  search_algorithm='grid',
                                                  return_tuner=True)
       duration = time.time() - start_time
       mlflow.log_param("duration_tune_model", duration)
       print("Modèle optimisé :", tuned_mybest)
    else:
        print("Aucun tuning appliqué, modèle non pris en charge.")
        tuned_mybest = best_model
 
    #  Enregistrer les meilleurs paramètres trouvés
    best_params = tuned_mybest.get_params()
    for param, value in best_params.items():
        mlflow.log_param(f"best_{param}", value)

    #  Évaluation sur l'échantillon de test
    start_time = time.time()
    predictions = session_bis.predict_model(tuned_mybest)
    duration = time.time() - start_time
    mlflow.log_param("duration_predict_model", duration)
    print(predictions)

    # Finalisation et ré-entrainement du modèle sur l'ensemble des données
    start_time = time.time()
    modele_definitif = session_bis.finalize_model(tuned_mybest)
    duration = time.time() - start_time
    mlflow.log_param("duration_finalize_model", duration)
    print(modele_definitif)

    import os
    # Passer à MLflow 2.20.2
    os.system("conda activate theo1_env")
    #  Enregistrement du modèle "optimisation-model" dans MLflow
    import mlflow.sklearn
    if hasattr(best_model, "estimators_"):
         best_model = best_model.estimators_[0]  # Prendre le premier modèle du pipeline
         mlflow.sklearn.log_model(best_model, "recherche-model")
         # Ajouter au registre MLflow
         mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/optimisation-model", "optimisation-model")
        
         # Revenir à MLflow 1.30.1
         os.system("conda activate theo_env")
   
# Fin explicite de la session (facultatif ici car le 'with' gère la fermeture)
mlflow.end_run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




