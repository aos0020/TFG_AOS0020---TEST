import streamlit as st

# --- FUNCIONES DE LOS CLASIFICADORES (Lógica de Python) ---
# Aquí es donde integrarías tus modelos reales de Machine Learning
def clasificar_tfidf(texto):
    # Simulación de lógica TF-IDF
    return "Especialidad sugerida (TF-IDF): Medicina Interna"

def clasificar_svd(texto):
    # Simulación de lógica SVD
    return "Especialidad sugerida (SVD): Cardiología"

def clasificar_svm(texto):
    # Simulación de lógica SVM
    return "Especialidad sugerida (SVM): Neurología"

# --- INTERFAZ GRÁFICA ---

# 1) Título
st.title("Triaje de especialidad médica a partir de síntomas")

# 2) Subtítulo
st.subheader("Clasificadores: TF-IDF, SVD y SVM")

# 3) Cuerpo
# 3.1) Caja de texto para síntomas (3 líneas aproximadamente)
input_sintomas = st.text_area("Introduce los síntomas del paciente:", height=100, placeholder="Ej: Dolor persistente en el pecho y dificultad para respirar...")

# 3.2) Lista desplegable
opcion_clasificador = st.selectbox(
    "Clasificador:",
    ("TF-IDF", "SVD", "SVM")
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
            else:
                resultado = clasificar_svm(input_sintomas)
        
        # 3.4) Caja con el resultado debajo del botón
        st.success("Resultado del análisis:")
        st.info(resultado)