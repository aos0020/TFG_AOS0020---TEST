import streamlit as st
from streamlit_mic_recorder import speech_to_text

# Importaciones de tus modelos
from TF_IDF.tf_idf import tf_idf
from SVD.SVD import svd
from SVM.SVM import svm
from RNA.RNA import rna

# --- FUNCIONES DE LOS CLASIFICADORES ---
def clasificar_tfidf(sintoma):
    try:
        especialidad, confianza = tf_idf(sintoma)
        return f"Especialidad sugerida (TF-IDF): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e: return f"Error: {e}"

def clasificar_svd(sintoma):
    try:
        especialidad, confianza = svd(sintoma)
        return f"Especialidad sugerida (SVD): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e: return f"Error: {e}"

def clasificar_svm(sintoma):
    especialidad, confianza = svm(sintoma)
    return f"Especialidad sugerida (SVM): {especialidad} (Confianza: {confianza:.2f}%)"

def clasificar_rna(sintoma):
    especialidad, confianza = rna(sintoma)
    return f"Especialidad sugerida (RNA): {especialidad} (Confianza: {confianza:.2f}%)"

# --- INTERFAZ GRÁFICA ---

st.title("Triaje médico inteligente")
st.subheader("Clasificadores: TF-IDF, SVD, SVM y RNA")

# Lógica para mantener el texto en la caja si viene de la voz
if 'texto_voz' not in st.session_state:
    st.session_state.texto_voz = ""

# --- SECCIÓN DE ENTRADA POR VOZ ---
st.write("🎤 **¿Prefieres dictar los síntomas?**")
texto_transcrito = speech_to_text(
    language='es',
    start_prompt="Hacer clic para grabar",
    stop_prompt="Detener grabación",
    just_once=True,
    key='grabador_voz'
)

if texto_transcrito:
    st.session_state.texto_voz = texto_transcrito

# 3.1) Caja de síntomas (se actualiza con la voz si existe)
input_sintomas = st.text_area(
    "Introduce los síntomas del paciente:", 
    value=st.session_state.texto_voz,
    height=150, 
    placeholder="Ej: Dolor persistente en el pecho..."
)

# 3.2) Lista desplegable
opcion_clasificador = st.selectbox(
    "Selecciona el motor de IA:",
    ("TF-IDF", "SVD", "SVM", "RNA")
)

# 3.3) Botón y lógica de llamada
if st.button("Realizar triaje", use_container_width=True):
    if input_sintomas.strip() == "":
        st.warning("Por favor, introduce o graba algún síntoma.")
    else:
        with st.spinner('Analizando cuadro clínico...'):
            if opcion_clasificador == "TF-IDF":
                resultado = clasificar_tfidf(input_sintomas)
            elif opcion_clasificador == "SVD":
                resultado = clasificar_svd(input_sintomas)
            elif opcion_clasificador == "SVM":
                resultado = clasificar_svm(input_sintomas)
            else:
                resultado = clasificar_rna(input_sintomas)

        st.success("Resultado del análisis:")
        st.info(resultado)