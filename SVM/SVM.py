# -*- coding: utf-8 -*-
"""SVM Classifier"""

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

# Cambiamos el nombre del archivo del modelo para evitar confusiones
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'svm_model.joblib')


def preprocess_text(text):
    """Limpia y prepara el texto para el modelo."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)


def cargar_modelo():
    """Carga el vectorizador TF-IDF y el clasificador SVM guardados."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No se encontró el modelo SVM en {MODEL_FILE}")

    # En SVM generalmente solo cargamos el vectorizador y el clasificador
    # (A menos que uses SVD previo a SVM, pero lo estándar es directo)
    tfidf_vectorizer, svm_model = load(MODEL_FILE)
    return tfidf_vectorizer, svm_model


def svm(sintomas_usuario):
    """Carga el modelo SVM y devuelve la predicción con su probabilidad."""
    try:
        tfidf_vectorizer, svm_model = cargar_modelo()
    except FileNotFoundError as e:
        return f"Error: {e}", 0

    # 1. Preprocesamiento
    sintomas_usuario_limpio = preprocess_text(sintomas_usuario)
    
    # 2. Transformación vectorial
    vector_tfidf = tfidf_vectorizer.transform([sintomas_usuario_limpio])

    # 3. Predicción
    # Nota: Para usar predict_proba, la SVM debe haber sido entrenada con probability=True
    try:
        probs = svm_model.predict_proba(vector_tfidf)
        indice_max = np.argmax(probs)
        especialidad = svm_model.classes_[indice_max]
        confianza = probs[0][indice_max] * 100
    except AttributeError:
        # Si la SVM no se entrenó con probabilidades, usamos predict directo
        especialidad = svm_model.predict(vector_tfidf)[0]
        confianza = 100.0  # Opcional: manejar de otra forma

    return especialidad, confianza