import csv
import hmac
import os
import subprocess
import sys
import tomllib

import pandas as pd
import streamlit as st
from streamlit_mic_recorder import speech_to_text

# Importaciones de tus modelos
from TF_IDF.tf_idf import tf_idf
from SVD.SVD import svd
from SVM.SVM import svm
from RNA.RNA import rna


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "Datos",
    "Dataset.5000.Registros.Marz.ID.Sintomas.Enfermedad.Especialidad.csv",
)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_ENV = "TRIAJE_ADMIN_PASSWORD"
ADMIN_SECRETS_PATH = os.path.join(BASE_DIR, ".streamlit", "secrets.toml.aos006")


st.set_page_config(page_title="Triaje médico inteligente")


def obtener_password_admin():
    try:
        password = st.secrets.get("admin_password", "")
    except Exception:
        password = ""

    if password:
        return password

    if os.path.exists(ADMIN_SECRETS_PATH):
        with open(ADMIN_SECRETS_PATH, "rb") as archivo:
            datos = tomllib.load(archivo)
            password = datos.get("admin_password", "")

    return password or os.environ.get(ADMIN_PASSWORD_ENV, "")


def credenciales_admin_validas(usuario, password):
    password_admin = obtener_password_admin()
    if not password_admin:
        return False

    usuario_correcto = hmac.compare_digest(usuario.strip(), ADMIN_USERNAME)
    password_correcta = hmac.compare_digest(password, password_admin)
    return usuario_correcto and password_correcta


def cerrar_sesion_admin():
    st.session_state.admin_autenticado = False


def mostrar_login_admin():
    st.title("Acceso administrador")

    if not obtener_password_admin():
        st.error(
            f"Configura la variable de entorno {ADMIN_PASSWORD_ENV} "
            "o el valor admin_password en Streamlit secrets."
        )
        return False

    with st.form("login_admin"):
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        enviado = st.form_submit_button("Entrar", use_container_width=True)

    if enviado:
        if credenciales_admin_validas(usuario, password):
            st.session_state.admin_autenticado = True
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")

    return False


def requiere_admin():
    if "admin_autenticado" not in st.session_state:
        st.session_state.admin_autenticado = False

    if st.session_state.admin_autenticado:
        st.sidebar.button("Cerrar sesión admin", on_click=cerrar_sesion_admin)
        return True

    return mostrar_login_admin()


# --- FUNCIONES DE LOS CLASIFICADORES ---
def clasificar_tfidf(sintoma):
    try:
        especialidad, confianza = tf_idf(sintoma)
        return f"Especialidad sugerida (TF-IDF): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e:
        return f"Error: {e}"


def clasificar_svd(sintoma):
    try:
        especialidad, confianza = svd(sintoma)
        return f"Especialidad sugerida (SVD): {especialidad} (Confianza: {confianza:.2f}%)"
    except Exception as e:
        return f"Error: {e}"


def clasificar_svm(sintoma):
    especialidad, confianza = svm(sintoma)
    return f"Especialidad sugerida (SVM): {especialidad} (Confianza: {confianza:.2f}%)"


def clasificar_rna(sintoma):
    especialidad, confianza = rna(sintoma)
    return f"Especialidad sugerida (RNA): {especialidad} (Confianza: {confianza:.2f}%)"


@st.cache_data
def cargar_dataset():
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame(columns=["record_id", "symptoms_text", "disease", "specialty"])

    return pd.read_csv(
        DATASET_PATH,
        encoding="latin1",
        engine="python",
        on_bad_lines="skip",
        sep=";",
    )


def obtener_siguiente_id(df):
    if df.empty or "record_id" not in df.columns:
        return 1

    ids = pd.to_numeric(df["record_id"], errors="coerce").dropna()
    if ids.empty:
        return 1

    return int(ids.max()) + 1


def obtener_especialidades(df):
    if df.empty or "specialty" not in df.columns:
        return []

    especialidades = df["specialty"].dropna().astype(str).str.strip()
    return sorted(especialidades[especialidades != ""].unique())


def guardar_registro(sintomas, especialidad, enfermedad=""):
    df = cargar_dataset()
    nuevo_id = obtener_siguiente_id(df)

    with open(DATASET_PATH, "a", newline="", encoding="latin1", errors="replace") as archivo:
        writer = csv.writer(archivo, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([nuevo_id, sintomas.strip(), enfermedad.strip(), especialidad.strip().upper()])

    cargar_dataset.clear()
    return nuevo_id


def ejecutar_script_generador(ruta_script):
    if not os.path.exists(ruta_script):
        raise FileNotFoundError(f"No se encontró el script de regeneración: {ruta_script}")

    resultado = subprocess.run(
        [sys.executable, ruta_script],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    if resultado.returncode != 0:
        salida = resultado.stderr.strip() or resultado.stdout.strip()
        raise RuntimeError(f"Error al ejecutar {os.path.basename(ruta_script)}:\n{salida}")

    return resultado.stdout.strip()


def mostrar_pagina_triaje():
    st.title("Triaje médico inteligente")
    st.subheader("Clasificadores: TF-IDF, SVD, SVM y RNA")

    if "texto_voz" not in st.session_state:
        st.session_state.texto_voz = ""

    st.write("🎤 **¿Prefieres dictar los síntomas?**")
    texto_transcrito = speech_to_text(
        language="es",
        start_prompt="Hacer clic para grabar",
        stop_prompt="Detener grabación",
        just_once=True,
        key="grabador_voz",
    )

    if texto_transcrito:
        st.session_state.texto_voz = texto_transcrito

    input_sintomas = st.text_area(
        "Introduce los síntomas del paciente:",
        value=st.session_state.texto_voz,
        height=150,
        placeholder="Ej: Dolor persistente en el pecho...",
    )

    opcion_clasificador = st.selectbox(
        "Selecciona el motor de IA:",
        ("TF-IDF", "SVD", "SVM", "RNA"),
    )

    if st.button("Realizar triaje", use_container_width=True):
        if input_sintomas.strip() == "":
            st.warning("Por favor, introduce o graba algún síntoma.")
        else:
            with st.spinner("Analizando cuadro clínico..."):
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


def mostrar_pagina_administracion():
    st.title("Administración de datos")
    st.subheader("Añadir síntomas y especialidad")

    df = cargar_dataset()
    especialidades = obtener_especialidades(df)

    with st.form("formulario_nuevo_registro", clear_on_submit=True):
        sintomas = st.text_area(
            "Síntomas",
            height=150,
            placeholder="Ej: Dolor de cabeza intenso, náuseas y sensibilidad a la luz.",
        )

        especialidad_existente = st.selectbox(
            "Especialidad existente",
            [""] + especialidades,
            index=0,
        )
        especialidad_nueva = st.text_input(
            "Nueva especialidad",
            placeholder="Ej: NEUROLOGÍA",
        )
        enfermedad = st.text_input(
            "Enfermedad o diagnóstico asociado",
            placeholder="Opcional",
        )

        enviado = st.form_submit_button("Guardar registro", use_container_width=True)

    if enviado:
        especialidad = especialidad_nueva.strip() or especialidad_existente.strip()

        if not sintomas.strip():
            st.warning("Debes introducir los síntomas.")
        elif not especialidad:
            st.warning("Debes seleccionar o escribir una especialidad.")
        else:
            try:
                nuevo_id = guardar_registro(sintomas, especialidad, enfermedad)
                st.success(f"Registro guardado correctamente con ID {nuevo_id}.")
            except Exception as e:
                st.error(f"No se pudo guardar el registro: {e}")

    st.divider()
    st.subheader("Regenerar modelos")

    script_paths = {
        "TF-IDF": os.path.join(BASE_DIR, "TF_IDF", "generarModeloTFIDF.py"),
        "SVD": os.path.join(BASE_DIR, "SVD", "generarModeloSVD.py"),
        "SVM": os.path.join(BASE_DIR, "SVM", "generarModeloSVM.py"),
        "RNA": os.path.join(BASE_DIR, "RNA", "generarModeloRNA.py"),
    }

    col1, col2 = st.columns(2)
    acciones = [
        ("TF-IDF", script_paths["TF-IDF"], col1),
        ("SVD", script_paths["SVD"], col2),
        ("SVM", script_paths["SVM"], col1),
        ("RNA", script_paths["RNA"], col2),
    ]

    for nombre, ruta, columna in acciones:
        if columna.button(f"Regenerar {nombre}"):
            with st.spinner(f"Regenerando modelo {nombre}..."):
                try:
                    salida = ejecutar_script_generador(ruta)
                    st.success(f"Modelo {nombre} regenerado correctamente.")
                    if salida:
                        st.code(salida, language="text")
                except Exception as e:
                    st.error(f"No se pudo regenerar {nombre}: {e}")

    st.divider()
    st.subheader("Últimos registros")

    df_actualizado = cargar_dataset()
    if df_actualizado.empty:
        st.info("Todavía no hay registros en el dataset.")
    else:
        columnas = ["record_id", "symptoms_text", "disease", "specialty"]
        columnas_visibles = [col for col in columnas if col in df_actualizado.columns]
        st.dataframe(df_actualizado[columnas_visibles].tail(10), use_container_width=True)


pagina = st.sidebar.radio(
    "Navegación",
    ("Triaje", "Administración"),
)

if pagina == "Triaje":
    mostrar_pagina_triaje()
else:
    if requiere_admin():
        mostrar_pagina_administracion()
