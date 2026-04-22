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
    nltk.download('stopwords')

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'tfidf_model.joblib')


def preprocess_text(text):
    """Limpia y prepara el texto para el modelo."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)


def cargar_modelo():
    """Carga el vectorizador TF-IDF y el modelo guardado."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"No se encontró el modelo guardado en {MODEL_FILE}")

    vectorizer, model = load(MODEL_FILE)
    return vectorizer, model


def tf_idf(sintomas_usuario):
    """Carga el modelo guardado y devuelve la predicción con su probabilidad."""
    try:
        tfidf_vectorizer, model = cargar_modelo()
    except FileNotFoundError as e:
        return f"Error: {e}", 0

    texto_limpio = preprocess_text(sintomas_usuario)
    vector_usuario = tfidf_vectorizer.transform([texto_limpio])

    probs = model.predict_proba(vector_usuario)
    indice_max = np.argmax(probs)

    especialidad = model.classes_[indice_max]
    confianza = probs[0][indice_max] * 100

    return especialidad, confianza
