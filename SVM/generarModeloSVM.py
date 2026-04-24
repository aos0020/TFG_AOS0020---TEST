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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump
import sys

# Ajuste de ruta para importar desde la carpeta del proyecto
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 1. Configuración de NLTK (Descarga solo la primera vez)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando stopwords...")
    nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

CSV_PATH = os.path.join(PROJECT_ROOT, 'Datos', 'Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv')
MODEL_FILE = os.path.join(ROOT_DIR, 'svm_model.joblib')

def clean_text(text):
    """Función de limpieza idéntica a la de predicción"""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)

def ejecutar_entrenamiento():
    # 2. Carga de datos
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encuentra el archivo en {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, encoding='latin1', sep=';')

    
    # 3. Preparación de datos
    # Aplicamos la limpieza antes de pasar los datos al Pipeline
    print("\nPreprocesando textos...")
    X = df["symptoms_text"].apply(clean_text).values
    y = df["specialty"].values

    # 4. Configuración del Pipeline
    # Usamos un Pipeline para que el GridSearchCV maneje el flujo correctamente
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('svm', SVC(probability=True, kernel='linear', random_state=42))
    ])

    # 5. Parámetros para GridSearchCV
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'tfidf__max_features': [1000, 3000, 5000],
        'tfidf__min_df': [1, 2]
    }

    # 6. Validación Cruzada Estratificada
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nIniciando búsqueda de hiperparámetros (GridSearch)...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=skf, 
        n_jobs=-1, 
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X, y)

    # 7. Guardar el modelo final
    print(f"\nMejor Accuracy: {grid_search.best_score_:.4f}")
    print(f"Mejores Parámetros: {grid_search.best_params_}")

    # Extraemos los componentes para que coincidan con tu función cargar_modelo()
    best_pipe = grid_search.best_estimator_
    vectorizador = best_pipe.named_steps['tfidf']
    clasificador = best_pipe.named_steps['svm']

    dump((vectorizador, clasificador), MODEL_FILE)
    print(f"\nArchivo '{MODEL_FILE}' generado con éxito.")

if __name__ == "__main__":
    ejecutar_entrenamiento()