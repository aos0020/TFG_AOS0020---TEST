# -*- coding: utf-8 -*-
"""rna Classifier"""

import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from joblib import load

# Configuración de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

# Ajustamos la ruta al archivo que generamos con el script de entrenamiento de la RNA
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'rna_model.joblib')


def preprocess_text(text):
    """Limpia y prepara el texto para el modelo (Stemming + Stopwords)."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)


def cargar_modelo():
    """Carga el vectorizador TF-IDF y el clasificador de Red Neuronal."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No se encontró el modelo guardado en {MODEL_FILE}")

    # Al entrenar la RNA guardamos una tupla: (vectorizador, rna_model)
    tfidf_vectorizer, rna_model = load(MODEL_FILE)
    return tfidf_vectorizer, rna_model


def rna(sintomas_usuario):
    """Carga el modelo RNA y devuelve la predicción con su probabilidad."""
    try:
        tfidf_vectorizer, rna_model = cargar_modelo()
    except FileNotFoundError as e:
        return f"Error: {e}", 0

    # 1. Preprocesamiento (idéntico al entrenamiento)
    sintomas_usuario_limpio = preprocess_text(sintomas_usuario)
    
    # 2. Transformación a vector numérico (TF-IDF)
    vector_tfidf = tfidf_vectorizer.transform([sintomas_usuario_limpio])

    # 3. Predicción de probabilidades
    # MLPClassifier soporta predict_proba por defecto
    probs = rna_model.predict_proba(vector_tfidf)
    indice_max = np.argmax(probs)

    # 4. Obtención de resultados
    especialidad = rna_model.classes_[indice_max]
    confianza = probs[0][indice_max] * 100

    return especialidad, confianza