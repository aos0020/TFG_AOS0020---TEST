# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier  # <--- Cambiamos SVM por MLP
from sklearn.pipeline import Pipeline
from joblib import dump
import sys

# Ajuste de rutas
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Configuración NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

CSV_PATH = os.path.join(PROJECT_ROOT, 'Datos', 'Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv')
MODEL_FILE = os.path.join(ROOT_DIR, 'rna_model.joblib') # Nombre actualizado

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)

def ejecutar_entrenamiento():
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encuentra el archivo en {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, encoding='latin1', sep=';')
    
    print("\nPreprocesando textos...")
    X = df["symptoms_text"].apply(clean_text).values
    y = df["specialty"].values

    # 4. Configuración del Pipeline con Red Neuronal
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('ann', MLPClassifier(
            max_iter=500, 
            early_stopping=True, 
            random_state=42
        ))
    ])

    # 5. Parámetros para ANN (Capas ocultas y regularización)
    param_grid = {
        'ann__hidden_layer_sizes': [(64,), (128, 64)], # Una capa de 64 o dos capas (128 y 64)
        'ann__alpha': [0.0001, 0.001],                # Regularización L2
        'ann__activation': ['relu', 'tanh'],           # Función de activación
        'tfidf__max_features': [1000, 2000]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nIniciando búsqueda de hiperparámetros para Red Neuronal...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=skf, 
        n_jobs=-1, 
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X, y)

    print(f"\nMejor Accuracy ANN: {grid_search.best_score_:.4f}")
    print(f"Mejores Parámetros: {grid_search.best_params_}")

    # Guardar componentes
    best_pipe = grid_search.best_estimator_
    vectorizador = best_pipe.named_steps['tfidf']
    clasificador = best_pipe.named_steps['ann']

    dump((vectorizador, clasificador), MODEL_FILE)
    print(f"\nArchivo '{MODEL_FILE}' generado con éxito.")

if __name__ == "__main__":
    ejecutar_entrenamiento()