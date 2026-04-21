import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import os

# Configuración de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = SnowballStemmer('spanish')
stopwords_spanish = stopwords.words('spanish')

def preprocess_text(text):
    """Limpia y prepara el texto para el modelo."""
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords_spanish]
    return ' '.join(words)

def tf_idf(sintomas_usuario):
    """Entrena el modelo y devuelve la predicción con su probabilidad."""
    # 1. Cargar datos
    # Ajusta la ruta según donde esté tu CSV en el servidor
    path_csv = 'Datos/Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv'
    
    if not os.path.exists(path_csv):
        return "Error: No se encontró el archivo de datos.", 0

    df = pd.read_csv(path_csv, encoding='latin1', engine='python', on_bad_lines='skip', sep=';')
    
    # 2. Preprocesamiento y Entrenamiento
    df['symptoms_processed'] = df['symptoms_text'].apply(preprocess_text)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['symptoms_processed'])
    y = df['specialty']
    
    X_train, _, y_train, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 3. Procesar entrada del usuario para la predicción
    texto_limpio = preprocess_text(sintomas_usuario)
    # IMPORTANTE: Usamos .transform() y pasamos una lista [texto] para evitar el error de 2D array
    vector_usuario = tfidf_vectorizer.transform([texto_limpio])
    
    # 4. Obtener Probabilidades
    probs = model.predict_proba(vector_usuario)
    indice_max = np.argmax(probs)
    
    especialidad = model.classes_[indice_max]
    confianza = probs[0][indice_max] * 100
    
    return especialidad, confianza