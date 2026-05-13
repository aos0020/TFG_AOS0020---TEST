import os
import sys
import warnings
import logging

# Suprimir todos los warnings y output de módulos
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump

# Ajuste de ruta para importar desde la carpeta del proyecto
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from TF_IDF.tf_idf import preprocess_text

# Configuración de NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

CSV_PATH = os.path.join(PROJECT_ROOT, 'Datos', 'Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv')
EXTRA_CSV_PATH = os.path.join(PROJECT_ROOT, 'Datos', 'triajes_baja_confianza_revisados.csv')
MODEL_FILE = os.path.join(ROOT_DIR, 'tfidf_model.joblib')


def _cargar_dataset_combinado():
    df = pd.read_csv(CSV_PATH, encoding='latin1', engine='python', on_bad_lines='skip', sep=';')
    if os.path.exists(EXTRA_CSV_PATH):
        try:
            df_extra = pd.read_csv(
                EXTRA_CSV_PATH, encoding='latin1', engine='python',
                on_bad_lines='skip', sep=';',
            )
            if not df_extra.empty and {'symptoms_text', 'specialty'}.issubset(df_extra.columns):
                df_extra = df_extra[['symptoms_text', 'specialty']].dropna()
                df = pd.concat([df, df_extra], ignore_index=True)
                print(f"Añadidos {len(df_extra)} registros revisados desde {os.path.basename(EXTRA_CSV_PATH)}.")
        except Exception as error:
            print(f"Aviso: no se pudieron leer los datos revisados: {error}")
    return df


def generar_modelo_tfidf():
    """Entrena y guarda el modelo TF-IDF + Logistic Regression."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    df = _cargar_dataset_combinado()
    df['symptoms_processed'] = df['symptoms_text'].apply(preprocess_text)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf_vectorizer.fit_transform(df['symptoms_processed'])
    y = df['specialty']

    X_train, _, y_train, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    dump((tfidf_vectorizer, model), MODEL_FILE)


if __name__ == '__main__':
    generar_modelo_tfidf()
