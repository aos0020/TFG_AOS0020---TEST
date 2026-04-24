# -*- coding: utf-8 -*-
"""SVD Classifier"""

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

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'svd_model.joblib')


def preprocess_text(text):
    """Limpia y prepara el texto para el modelo."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)


def cargar_modelo():
    """Carga el vectorizador TF-IDF, SVD y el clasificador guardados."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No se encontró el modelo guardado en {MODEL_FILE}")

    svd_vectorizer, svd_model, classifier_model = load(MODEL_FILE)
    return svd_vectorizer, svd_model, classifier_model


def svd(sintomas_usuario):
    """Carga el modelo guardado y devuelve la predicción con su probabilidad."""
    try:
        svd_vectorizer, svd_model, classifier_model = cargar_modelo()
    except FileNotFoundError as e:
        return f"Error: {e}", 0

    sintomas_usuario_limpio = preprocess_text(sintomas_usuario)
    vector_svd = svd_vectorizer.transform([sintomas_usuario_limpio])
    vector_svd = svd_model.transform(vector_svd)

    probs = classifier_model.predict_proba(vector_svd)
    indice_max = np.argmax(probs)

    especialidad = classifier_model.classes_[indice_max]
    confianza = probs[0][indice_max] * 100

    return especialidad, confianza