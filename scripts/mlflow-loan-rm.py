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
mlflow.set_tracking_uri("mysql+pymysql://root:X9605man9801pap%40%21theo@mysql-host:3306/mlflow_db?allowPublicKeyRetrieval=true")
mlflow.set_experiment("mlflow-automation")


# In[5]:


#chargement des données 
import pandas as pd
data = pd.read_csv(r"C:\Users\lucie\mlflow-automation\notebooks\Loan_Data.csv")
#data = pd.read_csv('Loan_Data.csv')
data.info()


# In[6]:


import time
import mlflow
from pycaret.classification import ClassificationExperiment

# S'assurer que toute session MLflow précédente est fermée
if mlflow.active_run():
    mlflow.end_run()

# Démarrer la session primaire avec suivi MLflow
with mlflow.start_run(run_name="recherche_modele"):
    start_time = time.time()

    # Initialiser et configurer l'expérience
    session_prim = ClassificationExperiment()
    session_prim.setup(data, normalize=True, target='default', train_size=0.7,
                       data_split_stratify=True, fold=5, session_id=0,
                       log_experiment=True, experiment_name="recherche_modele")
    
    duration = time.time() - start_time
    mlflow.log_param("duration", duration)
    
    # Afficher les algorithmes disponibles
    algos = session_prim.models()
    print(algos)

    # Initialisation d'une liste vide avant la boucle
    top_models = []

    # Comparer et enregistrer la durée pour chaque modèle
    model_list = ['lr', 'nb', 'dt', 'rf', 'svm', 'lda']
    model_durations = {}

    for model in model_list:
        start_time = time.time()
        mdl = session_prim.create_model(model)
        duration = time.time() - start_time
        model_durations[model] = duration
        mlflow.log_param(f"duration_{model}", duration)
        top_models.append(mdl)  # Correction : on passe 'mdl' comme argument

        # Comparer et sélectionner les meilleurs modèles
        best_model = session_prim.compare_models(sort='Accuracy', include=model_list, verbose=True)
        print("Durées d'entraînement par modèle :", model_durations)
        # Extraire les résultats
        results = session_prim.pull()
        print(results)
        
        import os
        # Passer à MLflow 2.20.2
        os.system("conda activate theo1_env")
        #  Enregistrement du modèle "recherche-model" dans MLflow
        import mlflow.sklearn
        if hasattr(best_model, "estimators_"):
           best_model = best_model.estimators_[0]  # Prendre le premier modèle du pipeline
           mlflow.sklearn.log_model(best_model, "recherche-model")
           # Ajouter au registre MLflow
           mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/recherche-model", "recherche-model")
        
           # Revenir à MLflow 1.30.1
           os.system("conda activate theo_env")
# Fin explicite de la session (facultatif ici car le 'with' gère la fermeture)
mlflow.end_run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




