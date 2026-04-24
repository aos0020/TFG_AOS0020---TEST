import streamlit as st
from TF_IDF.tf_idf import tf_idf
from SVD.SVD import svd
from SVM.SVM import svm
from RNA.RNA import rna

# --- FUNCIONES DE LOS CLASIFICADORES (Lógica de Python) ---
# Aquí es donde integrarías tus modelos reales de Machine Learning
def clasificar_tfidf(sintoma):
    try:
        # Llamamos a la función de Python directamente
        # Es instantáneo y no abre procesos externos
        especialidad, confianza = tf_idf(sintoma)
        return f"Especialidad sugerida (TF-IDF): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e:
        return f"Error al ejecutar el módulo: {e}"

def clasificar_svd(sintoma):
    try:
        # Llamamos a la función de Python directamente
        # Es instantáneo y no abre procesos externos
        especialidad, confianza = svd(sintoma)
        return f"Especialidad sugerida (SVD): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e:
        return f"Error al ejecutar el módulo: {e}"

def clasificar_svm(sintoma):
    # Llamamos a la función SVM
    especialidad, confianza = svm(sintoma)
    return f"Especialidad sugerida (SVM): {especialidad} (Confianza: {confianza:.2f}%)"


def clasificar_rna(sintoma):
    # Llamamos a la función RNA
    especialidad, confianza = rna(sintoma)
    return f"Especialidad sugerida (RNA): {especialidad} (Confianza: {confianza:.2f}%)"

# --- INTERFAZ GRÁFICA ---

# 1) Título
st.title("Triaje de especialidad médica a partir de síntomas")

# 2) Subtítulo
st.subheader("Clasificadores: TF-IDF, SVD, SVM y RNA")

# 3) Cuerpo
# 3.1) Caja de sintoma para síntomas (3 líneas aproximadamente)
input_sintomas = st.text_area("Introduce los síntomas del paciente:", height=100, placeholder="Ej: Dolor persistente en el pecho y dificultad para respirar...")

# 3.2) Lista desplegable
opcion_clasificador = st.selectbox(
    "Clasificador:",
    ("TF-IDF", "SVD", "SVM", "RNA")
)

# 3.3) Botón y lógica de llamada
if st.button("Realizar triaje"):
    if input_sintomas.strip() == "":
        st.warning("Por favor, introduce algún síntoma antes de continuar.")
    else:
        # Llamada a un clasificador diferente según la opción
        with st.spinner('Procesando triaje médico...'):
            if opcion_clasificador == "TF-IDF":
                resultado = clasificar_tfidf(input_sintomas)
            elif opcion_clasificador == "SVD":
                resultado = clasificar_svd(input_sintomas)
            elif opcion_clasificador == "SVM":
                resultado = clasificar_svm(input_sintomas)
            else:
                resultado = clasificar_rna(input_sintomas)

        # 3.4) Caja con el resultado debajo del botón
        st.success("Resultado del análisis:")
        st.info(resultado)