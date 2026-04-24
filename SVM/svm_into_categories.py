# -*- coding: utf-8 -*-
"""Módulo SVM para clasificación de síntomas en especialidades médicas."""

import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from joblib import load, dump

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'svm_model.joblib')

CSV_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        'Datos',
        'Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv'
    )
)


def preprocess_text(text):
    """Limpia y prepara el texto de síntomas para el modelo."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)


def entrenar_modelo():
    """Entrena el pipeline SVM a partir del CSV de datos y guarda el modelo."""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"No se encontró el CSV de entrenamiento en {CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding='latin1', sep=';', engine='python', on_bad_lines='skip')
    if 'symptoms_text' not in df.columns or 'specialty' not in df.columns:
        raise ValueError("El CSV debe contener las columnas 'symptoms_text' y 'specialty'.")

    df = df.dropna(subset=['symptoms_text', 'specialty'])
    X = df['symptoms_text'].astype(str).apply(preprocess_text)
    y = df['specialty'].astype(str)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', LinearSVC(C=1.0, max_iter=5000))
    ])
    pipeline.fit(X, y)

    try:
        dump(pipeline, MODEL_FILE)
    except Exception:
        pass

    return pipeline


def cargar_modelo():
    """Carga el modelo SVM guardado o lo entrena si no existe."""
    if os.path.exists(MODEL_FILE):
        return load(MODEL_FILE)
    return entrenar_modelo()


def svm(sintomas_usuario):
    """Predice la especialidad médica a partir del texto de síntomas."""
    pipeline = cargar_modelo()
    texto_limpio = preprocess_text(sintomas_usuario)
    prediccion = pipeline.predict([texto_limpio])[0]
    return prediccion, 0.0
