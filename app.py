import csv
import hmac
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime

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
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FALLBACK_PATH = os.path.join(LOGS_DIR, "triaje_fallback_log.csv")
DATOS_REVISADOS_PATH = os.path.join(
    BASE_DIR, "Datos", "triajes_baja_confianza_revisados.csv"
)
ESPECIALIDAD_POR_DEFECTO = "MEDICINA GENERAL"

LOG_HEADER = [
    "timestamp",
    "clasificador",
    "rondas_realizadas",
    "umbral_confianza",
    "sintomas",
    "especialidad_sugerida_modelo",
    "confianza_modelo",
    "especialidad_elegida_paciente",
    "motivo",
]

CONFIG_POR_DEFECTO = {
    "umbral_confianza": 60.0,
    "rondas_extra": 1,
}

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_ENV = "TRIAJE_ADMIN_PASSWORD"
ADMIN_SECRETS_PATH = os.path.join(BASE_DIR, ".streamlit", "secrets.toml.aos006")


st.set_page_config(page_title="Triaje médico inteligente")


def cargar_config():
    config = dict(CONFIG_POR_DEFECTO)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as archivo:
                datos = json.load(archivo)
            if isinstance(datos, dict):
                config.update({k: datos[k] for k in datos if k in CONFIG_POR_DEFECTO})
        except (json.JSONDecodeError, OSError):
            pass

    try:
        config["umbral_confianza"] = float(config["umbral_confianza"])
    except (TypeError, ValueError):
        config["umbral_confianza"] = CONFIG_POR_DEFECTO["umbral_confianza"]

    try:
        config["rondas_extra"] = max(0, int(config["rondas_extra"]))
    except (TypeError, ValueError):
        config["rondas_extra"] = CONFIG_POR_DEFECTO["rondas_extra"]

    return config


def guardar_config(umbral_confianza, rondas_extra):
    config = {
        "umbral_confianza": float(umbral_confianza),
        "rondas_extra": max(0, int(rondas_extra)),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as archivo:
        json.dump(config, archivo, indent=2, ensure_ascii=False)
    return config


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
# Devuelven (especialidad, confianza) o (None, None) si ha habido error.
def clasificar_tfidf(sintoma):
    return tf_idf(sintoma)


def clasificar_svd(sintoma):
    return svd(sintoma)


def clasificar_svm(sintoma):
    return svm(sintoma)


def clasificar_rna(sintoma):
    return rna(sintoma)


CLASIFICADORES = {
    "TF-IDF": clasificar_tfidf,
    "SVD": clasificar_svd,
    "SVM": clasificar_svm,
    "RNA": clasificar_rna,
}

MODO_TODOS = "Todos (recomendado)"
OPCIONES_CLASIFICADOR = [MODO_TODOS] + list(CLASIFICADORES.keys())


def ejecutar_clasificador(nombre, sintomas):
    funcion = CLASIFICADORES[nombre]
    try:
        especialidad, confianza = funcion(sintomas)
        return especialidad, float(confianza), None
    except Exception as error:
        return None, None, str(error)


def ejecutar_todos_clasificadores(sintomas):
    resultados = {}
    for nombre in CLASIFICADORES:
        especialidad, confianza, error = ejecutar_clasificador(nombre, sintomas)
        resultados[nombre] = {
            "especialidad": especialidad,
            "confianza": confianza,
            "error": error,
        }
    return resultados


def mejor_resultado(resultados):
    validos = [
        (n, r) for n, r in resultados.items()
        if r["error"] is None and r["confianza"] is not None
    ]
    if not validos:
        return None, None, None
    nombre, datos = max(validos, key=lambda item: item[1]["confianza"])
    return nombre, datos["especialidad"], datos["confianza"]


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
def _asegurar_header_log():
    if not os.path.exists(LOG_FALLBACK_PATH):
        with open(LOG_FALLBACK_PATH, "w", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(LOG_HEADER)
        return

    with open(LOG_FALLBACK_PATH, "r", encoding="utf-8", newline="") as archivo:
        reader = csv.reader(archivo, delimiter=";")
        try:
            cabecera = next(reader)
        except StopIteration:
            cabecera = []
        filas = list(reader)

    if cabecera == LOG_HEADER:
        return

    if "motivo" not in cabecera:
        for fila in filas:
            while len(fila) < len(LOG_HEADER) - 1:
                fila.append("")
            fila.append("baja_confianza")

    with open(LOG_FALLBACK_PATH, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(LOG_HEADER)
        writer.writerows(filas)


def registrar_log_fallback(
    sintomas,
    clasificador,
    especialidad_modelo,
    confianza_modelo,
    especialidad_paciente,
    rondas_realizadas,
    umbral,
    motivo="baja_confianza",
):
    os.makedirs(LOGS_DIR, exist_ok=True)
    _asegurar_header_log()

    with open(LOG_FALLBACK_PATH, "a", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            clasificador,
            rondas_realizadas,
            f"{umbral:.2f}",
            sintomas.strip(),
            (especialidad_modelo or "").strip(),
            f"{confianza_modelo:.2f}" if confianza_modelo is not None else "",
            especialidad_paciente.strip().upper(),
            motivo,
        ])


def agregar_a_datos_revisados(sintomas, especialidad):
    sintomas_limpios = sintomas.strip()
    especialidad_limpia = especialidad.strip().upper()
    if not sintomas_limpios or not especialidad_limpia:
        raise ValueError("Síntomas y especialidad son obligatorios.")

    os.makedirs(os.path.dirname(DATOS_REVISADOS_PATH), exist_ok=True)
    nuevo_archivo = not os.path.exists(DATOS_REVISADOS_PATH)

    with open(DATOS_REVISADOS_PATH, "a", newline="", encoding="latin1", errors="replace") as archivo:
        writer = csv.writer(archivo, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        if nuevo_archivo:
            writer.writerow(["symptoms_text", "specialty"])
        writer.writerow([sintomas_limpios, especialidad_limpia])


def reset_estado_triaje():
    for clave in (
        "triaje_etapa",
        "triaje_sintomas",
        "triaje_clasificador",
        "triaje_resultado",
        "triaje_resultados_todos",
        "triaje_modo",
        "triaje_rondas",
        "texto_voz",
        "texto_voz_extra",
    ):
        if clave in st.session_state:
            del st.session_state[clave]


def mostrar_pagina_triaje():
    st.title("Triaje médico inteligente")
    st.subheader("Clasificadores: TF-IDF, SVD, SVM y RNA")

    config = cargar_config()
    umbral = config["umbral_confianza"]
    rondas_extra = config["rondas_extra"]

    st.caption(
        f"Umbral de confianza configurado: {umbral:.2f}% · "
        f"Rondas extra de ampliación: {rondas_extra}"
    )

    if "triaje_etapa" not in st.session_state:
        st.session_state.triaje_etapa = "inicio"
    if "triaje_sintomas" not in st.session_state:
        st.session_state.triaje_sintomas = ""
    if "triaje_clasificador" not in st.session_state:
        st.session_state.triaje_clasificador = "TF-IDF"
    if "triaje_resultado" not in st.session_state:
        st.session_state.triaje_resultado = None
    if "triaje_resultados_todos" not in st.session_state:
        st.session_state.triaje_resultados_todos = None
    if "triaje_modo" not in st.session_state:
        st.session_state.triaje_modo = MODO_TODOS
    if "triaje_rondas" not in st.session_state:
        st.session_state.triaje_rondas = 0
    if "texto_voz" not in st.session_state:
        st.session_state.texto_voz = ""

    etapa = st.session_state.triaje_etapa

    if etapa == "inicio":
        _etapa_inicio(umbral, rondas_extra)
    elif etapa == "ampliando":
        _etapa_ampliando(umbral, rondas_extra)
    elif etapa == "fallback":
        _etapa_fallback(umbral)
    elif etapa == "completado":
        _etapa_completado()
    elif etapa == "reportar_error":
        _etapa_reportar_error(umbral)


def _etapa_inicio(umbral, rondas_extra):
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
        OPCIONES_CLASIFICADOR,
        index=0,
        help=(
            "Por defecto se ejecutan los cuatro modelos y se elige el de mayor "
            "confianza. Puedes forzar un modelo concreto si lo prefieres."
        ),
    )

    if st.button("Realizar triaje", use_container_width=True):
        if input_sintomas.strip() == "":
            st.warning("Por favor, introduce o graba algún síntoma.")
            return

        if opcion_clasificador == MODO_TODOS:
            with st.spinner("Analizando cuadro clínico con todos los modelos..."):
                resultados = ejecutar_todos_clasificadores(input_sintomas)

            nombre_mejor, especialidad, confianza = mejor_resultado(resultados)
            if nombre_mejor is None:
                errores = "; ".join(
                    f"{n}: {r['error']}" for n, r in resultados.items() if r["error"]
                )
                st.error(
                    f"Ningún clasificador devolvió un resultado válido. {errores}"
                )
                return

            st.session_state.triaje_resultados_todos = resultados
            st.session_state.triaje_clasificador = nombre_mejor
        else:
            with st.spinner("Analizando cuadro clínico..."):
                especialidad, confianza, error = ejecutar_clasificador(
                    opcion_clasificador, input_sintomas
                )

            if error:
                st.error(f"Error: {error}")
                return

            st.session_state.triaje_resultados_todos = None
            st.session_state.triaje_clasificador = opcion_clasificador

        st.session_state.triaje_sintomas = input_sintomas.strip()
        st.session_state.triaje_modo = opcion_clasificador
        st.session_state.triaje_resultado = (especialidad, confianza)
        st.session_state.triaje_rondas = 0

        if confianza >= umbral:
            st.session_state.triaje_etapa = "completado"
        elif rondas_extra > 0:
            st.session_state.triaje_etapa = "ampliando"
        else:
            st.session_state.triaje_etapa = "fallback"

        st.rerun()


def _etapa_ampliando(umbral, rondas_extra):
    especialidad, confianza = st.session_state.triaje_resultado
    rondas_realizadas = st.session_state.triaje_rondas
    rondas_restantes = rondas_extra - rondas_realizadas

    st.warning(
        f"Confianza insuficiente: el modelo "
        f"**{st.session_state.triaje_clasificador}** sugirió "
        f"**{especialidad}** con **{confianza:.2f}%** "
        f"(umbral: {umbral:.2f}%)."
    )
    st.info(
        f"Por favor, añade más síntomas para afinar el diagnóstico. "
        f"Quedan **{rondas_restantes}** ronda(s) de ampliación antes de "
        "asignar especialidad manualmente."
    )

    st.markdown("**Síntomas recogidos hasta ahora:**")
    st.code(st.session_state.triaje_sintomas, language="text")

    if "texto_voz_extra" not in st.session_state:
        st.session_state.texto_voz_extra = ""

    st.write("🎤 **¿Quieres dictar los síntomas adicionales?**")
    texto_extra = speech_to_text(
        language="es",
        start_prompt="Hacer clic para grabar",
        stop_prompt="Detener grabación",
        just_once=True,
        key=f"grabador_voz_extra_{rondas_realizadas}",
    )
    if texto_extra:
        st.session_state.texto_voz_extra = texto_extra

    sintomas_adicionales = st.text_area(
        "Síntomas adicionales:",
        value=st.session_state.texto_voz_extra,
        height=120,
        placeholder="Describe nuevos síntomas o detalla los existentes...",
        key=f"sintomas_extra_{rondas_realizadas}",
    )

    col_reintentar, col_cancelar = st.columns(2)
    with col_reintentar:
        if st.button("Reintentar triaje", use_container_width=True):
            if not sintomas_adicionales.strip():
                st.warning("Añade al menos un síntoma adicional.")
                return

            sintomas_combinados = (
                f"{st.session_state.triaje_sintomas}. {sintomas_adicionales.strip()}"
            )

            with st.spinner("Analizando cuadro clínico ampliado..."):
                nueva_esp, nueva_conf, error = ejecutar_clasificador(
                    st.session_state.triaje_clasificador, sintomas_combinados
                )

            if error:
                st.error(f"Error: {error}")
                return

            st.session_state.triaje_sintomas = sintomas_combinados
            st.session_state.triaje_resultado = (nueva_esp, nueva_conf)
            st.session_state.triaje_resultados_todos = None
            st.session_state.triaje_rondas = rondas_realizadas + 1
            st.session_state.texto_voz_extra = ""

            if nueva_conf >= umbral:
                st.session_state.triaje_etapa = "completado"
            elif st.session_state.triaje_rondas >= rondas_extra:
                st.session_state.triaje_etapa = "fallback"
            # si quedan rondas, seguimos en "ampliando"
            st.rerun()

    with col_cancelar:
        if st.button("Cancelar y empezar de nuevo", use_container_width=True):
            reset_estado_triaje()
            st.rerun()


def _etapa_fallback(umbral):
    especialidad_modelo, confianza_modelo = st.session_state.triaje_resultado

    st.error(
        f"Tras {st.session_state.triaje_rondas} ronda(s) de ampliación, la "
        f"confianza sigue por debajo del umbral ({umbral:.2f}%). Último resultado "
        f"del modelo **{st.session_state.triaje_clasificador}**: "
        f"**{especialidad_modelo}** ({confianza_modelo:.2f}%)."
    )
    st.markdown("**Síntomas registrados:**")
    st.code(st.session_state.triaje_sintomas, language="text")

    st.subheader("Selección de especialidad por el paciente")

    df = cargar_dataset()
    especialidades = obtener_especialidades(df)
    if ESPECIALIDAD_POR_DEFECTO not in especialidades:
        especialidades = [ESPECIALIDAD_POR_DEFECTO] + especialidades

    especialidad_sugerida = (especialidad_modelo or ESPECIALIDAD_POR_DEFECTO).strip().upper()

    col_general, col_seleccion = st.columns(2)

    with col_general:
        st.markdown(f"**Opción 1:** aceptar la sugerencia del sistema: **{especialidad_sugerida}**.")
        if st.button(
            f"Aceptar {especialidad_sugerida}",
            use_container_width=True,
            key="btn_sugerencia_modelo",
        ):
            _finalizar_fallback(
                especialidad_sugerida,
                especialidad_modelo,
                confianza_modelo,
                umbral,
            )

    with col_seleccion:
        st.markdown("**Opción 2:** elegir manualmente otra especialidad.")
        indice_defecto = (
            especialidades.index(ESPECIALIDAD_POR_DEFECTO)
            if ESPECIALIDAD_POR_DEFECTO in especialidades
            else 0
        )
        seleccion = st.selectbox(
            "Especialidad sugerida por el paciente",
            especialidades,
            index=indice_defecto,
            key="seleccion_paciente",
        )
        if st.button(
            "Confirmar selección",
            use_container_width=True,
            key="btn_seleccion_paciente",
        ):
            _finalizar_fallback(
                seleccion,
                especialidad_modelo,
                confianza_modelo,
                umbral,
            )


def _finalizar_fallback(especialidad_paciente, especialidad_modelo, confianza_modelo, umbral):
    try:
        registrar_log_fallback(
            sintomas=st.session_state.triaje_sintomas,
            clasificador=st.session_state.triaje_clasificador,
            especialidad_modelo=especialidad_modelo,
            confianza_modelo=confianza_modelo,
            especialidad_paciente=especialidad_paciente,
            rondas_realizadas=st.session_state.triaje_rondas,
            umbral=umbral,
        )
    except OSError as error:
        st.error(f"No se pudo escribir el log: {error}")
        return

    st.session_state.triaje_resultado = (
        especialidad_paciente.upper(),
        confianza_modelo if confianza_modelo is not None else 0.0,
    )
    st.session_state.triaje_etapa = "completado"
    st.session_state.triaje_fallback = True
    st.session_state.triaje_fallback_motivo = "baja_confianza"
    st.rerun()


def _etapa_completado():
    especialidad, confianza = st.session_state.triaje_resultado
    fue_fallback = st.session_state.get("triaje_fallback", False)
    motivo_fallback = st.session_state.get("triaje_fallback_motivo", "baja_confianza")

    st.success("Resultado del análisis:")
    if fue_fallback:
        if motivo_fallback == "reporte_error":
            mensaje = (
                f"Especialidad asignada tras reporte de error: **{especialidad}**.\n\n"
                f"La sugerencia del modelo {st.session_state.triaje_clasificador} fue "
                f"marcada como incorrecta y la corrección se ha registrado en "
                f"`{os.path.relpath(LOG_FALLBACK_PATH, BASE_DIR)}`."
            )
        else:
            mensaje = (
                f"Especialidad asignada (selección del paciente): **{especialidad}**.\n\n"
                f"La sugerencia del modelo {st.session_state.triaje_clasificador} no "
                f"alcanzó el umbral de confianza. La fila se ha registrado en "
                f"`{os.path.relpath(LOG_FALLBACK_PATH, BASE_DIR)}`."
            )
        st.info(mensaje)
    else:
        st.info(
            f"Especialidad sugerida ({st.session_state.triaje_clasificador}): "
            f"**{especialidad}** (Confianza: {confianza:.2f}%)"
        )

    resultados_todos = st.session_state.get("triaje_resultados_todos")
    if resultados_todos:
        st.markdown("**Resultados de todos los modelos:**")
        ganador = st.session_state.triaje_clasificador
        filas_tabla = []
        for nombre, r in resultados_todos.items():
            if r["error"]:
                estado = "Error"
                especialidad_celda = "—"
                confianza_celda = "—"
            else:
                estado = "★ Elegido" if nombre == ganador else "OK"
                especialidad_celda = r["especialidad"]
                confianza_celda = (
                    f"{r['confianza']:.2f}" if r["confianza"] is not None else "—"
                )
            filas_tabla.append({
                "Modelo": nombre,
                "Especialidad": especialidad_celda,
                "Confianza (%)": confianza_celda,
                "Estado": estado,
            })
        df_resultados = pd.DataFrame(filas_tabla)
        st.dataframe(df_resultados, hide_index=True, use_container_width=True)

    col_nuevo, col_error = st.columns(2)
    with col_nuevo:
        if st.button("Realizar otro triaje", use_container_width=True):
            for clave in ("triaje_fallback", "triaje_fallback_motivo"):
                if clave in st.session_state:
                    del st.session_state[clave]
            reset_estado_triaje()
            st.rerun()
    with col_error:
        if not fue_fallback:
            if st.button(
                "Reportar error",
                use_container_width=True,
                key="btn_reportar_error",
                help="Indica que la especialidad sugerida no es correcta.",
            ):
                st.session_state.triaje_etapa = "reportar_error"
                st.rerun()


def _etapa_reportar_error(umbral):
    especialidad_modelo, confianza_modelo = st.session_state.triaje_resultado

    st.warning(
        f"Has indicado que el resultado del modelo "
        f"**{st.session_state.triaje_clasificador}** no es correcto: "
        f"**{especialidad_modelo}** "
        f"({confianza_modelo:.2f}% de confianza)."
    )
    st.markdown("**Síntomas registrados:**")
    st.code(st.session_state.triaje_sintomas, language="text")

    st.subheader("Indica la especialidad correcta")

    df = cargar_dataset()
    especialidades = obtener_especialidades(df)
    if ESPECIALIDAD_POR_DEFECTO not in especialidades:
        especialidades = [ESPECIALIDAD_POR_DEFECTO] + especialidades

    indice_defecto = (
        especialidades.index(ESPECIALIDAD_POR_DEFECTO)
        if ESPECIALIDAD_POR_DEFECTO in especialidades
        else 0
    )
    seleccion = st.selectbox(
        "Especialidad correcta según el paciente",
        especialidades,
        index=indice_defecto,
        key="reporte_error_especialidad",
    )

    col_enviar, col_cancelar = st.columns(2)
    with col_enviar:
        if st.button(
            "Enviar reporte de error",
            use_container_width=True,
            key="btn_enviar_reporte_error",
        ):
            try:
                registrar_log_fallback(
                    sintomas=st.session_state.triaje_sintomas,
                    clasificador=st.session_state.triaje_clasificador,
                    especialidad_modelo=especialidad_modelo,
                    confianza_modelo=confianza_modelo,
                    especialidad_paciente=seleccion,
                    rondas_realizadas=st.session_state.triaje_rondas,
                    umbral=umbral,
                    motivo="reporte_error",
                )
            except OSError as error:
                st.error(f"No se pudo escribir el log: {error}")
                return

            st.session_state.triaje_resultado = (
                seleccion.upper(),
                confianza_modelo if confianza_modelo is not None else 0.0,
            )
            st.session_state.triaje_fallback = True
            st.session_state.triaje_fallback_motivo = "reporte_error"
            st.session_state.triaje_etapa = "completado"
            st.rerun()

    with col_cancelar:
        if st.button(
            "Cancelar",
            use_container_width=True,
            key="btn_cancelar_reporte_error",
        ):
            st.session_state.triaje_etapa = "completado"
            st.rerun()


def mostrar_seccion_configuracion():
    st.subheader("Configuración del triaje")
    config = cargar_config()

    with st.form("formulario_config_triaje"):
        umbral = st.slider(
            "Umbral mínimo de confianza (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(config["umbral_confianza"]),
            step=1.0,
            help=(
                "Si la confianza devuelta por el clasificador es inferior a este "
                "valor, se pedirán más síntomas o se pasará al fallback."
            ),
        )
        rondas_extra = st.number_input(
            "Rondas extra de ampliación de síntomas",
            min_value=0,
            max_value=10,
            value=int(config["rondas_extra"]),
            step=1,
            help=(
                "Número máximo de veces que se pedirá al paciente añadir más "
                "síntomas antes de mostrar el fallback (Medicina General o "
                "selección manual)."
            ),
        )
        guardado = st.form_submit_button("Guardar configuración", use_container_width=True)

    if guardado:
        try:
            guardar_config(umbral, rondas_extra)
            st.success("Configuración actualizada.")
        except OSError as error:
            st.error(f"No se pudo guardar la configuración: {error}")


def mostrar_seccion_logs():
    st.subheader("Registro de triajes (baja confianza y reportes de error)")
    if not os.path.exists(LOG_FALLBACK_PATH):
        st.info("Todavía no hay entradas en el log.")
        return

    try:
        df_log = pd.read_csv(LOG_FALLBACK_PATH, sep=";", encoding="utf-8")
    except Exception as error:
        st.error(f"No se pudo leer el log: {error}")
        return

    if df_log.empty:
        st.info("Todavía no hay entradas en el log de fallback.")
        return

    df_dataset = cargar_dataset()
    especialidades = obtener_especialidades(df_dataset)
    if ESPECIALIDAD_POR_DEFECTO not in especialidades:
        especialidades = [ESPECIALIDAD_POR_DEFECTO] + especialidades

    st.caption(
        f"Log: `{os.path.relpath(LOG_FALLBACK_PATH, BASE_DIR)}` · "
        f"Revisados: `{os.path.relpath(DATOS_REVISADOS_PATH, BASE_DIR)}`"
    )

    for idx, fila in df_log.iloc[::-1].iterrows():
        sintomas = str(fila.get("sintomas", "")).strip()
        especialidad_registrada = str(fila.get("especialidad_elegida_paciente", "")).strip().upper()
        timestamp = str(fila.get("timestamp", ""))
        clasificador = str(fila.get("clasificador", ""))
        motivo = str(fila.get("motivo", "baja_confianza")).strip() or "baja_confianza"
        etiqueta_motivo = "reporte de error" if motivo == "reporte_error" else "baja confianza"

        opciones = list(especialidades)
        if especialidad_registrada and especialidad_registrada not in opciones:
            opciones = [especialidad_registrada] + opciones
        indice_defecto = (
            opciones.index(especialidad_registrada)
            if especialidad_registrada in opciones
            else 0
        )

        with st.container(border=True):
            st.markdown(f"**{timestamp}** · _{clasificador}_ · `{etiqueta_motivo}`")
            st.text_area(
                "Síntomas",
                value=sintomas,
                height=80,
                disabled=True,
                key=f"log_sintomas_{idx}",
                label_visibility="collapsed",
            )

            col_esp, col_btn = st.columns([3, 1])
            with col_esp:
                especialidad_elegida = st.selectbox(
                    "Especialidad",
                    opciones,
                    index=indice_defecto,
                    key=f"log_especialidad_{idx}",
                )
            with col_btn:
                st.write("")
                if st.button("Añadir a datos", key=f"log_add_{idx}", use_container_width=True):
                    try:
                        agregar_a_datos_revisados(sintomas, especialidad_elegida)
                        st.success(
                            f"Añadido a {os.path.basename(DATOS_REVISADOS_PATH)}."
                        )
                    except (OSError, ValueError) as error:
                        st.error(f"No se pudo añadir: {error}")


def mostrar_pagina_administracion():
    st.title("Administración de datos")

    mostrar_seccion_configuracion()
    st.divider()

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
    st.caption(
        f"Los modelos se entrenan combinando el dataset principal con los registros de "
        f"`{os.path.relpath(DATOS_REVISADOS_PATH, BASE_DIR)}`."
    )

    script_paths = {
        "TF-IDF": os.path.join(BASE_DIR, "TF_IDF", "generarModeloTFIDF.py"),
        "SVD": os.path.join(BASE_DIR, "SVD", "generarModeloSVD.py"),
        "SVM": os.path.join(BASE_DIR, "SVM", "generarModeloSVM.py"),
        "RNA": os.path.join(BASE_DIR, "RNA", "generarModeloRNA.py"),
    }

    if st.button("Regenerar todos los modelos", use_container_width=True, type="primary"):
        progreso = st.progress(0.0, text="Iniciando regeneración…")
        errores = []
        total = len(script_paths)
        for indice, (nombre, ruta) in enumerate(script_paths.items(), start=1):
            progreso.progress(
                (indice - 1) / total,
                text=f"Regenerando modelo {nombre} ({indice}/{total})…",
            )
            try:
                salida = ejecutar_script_generador(ruta)
                with st.expander(f"Salida de {nombre}", expanded=False):
                    st.code(salida or "(sin salida)", language="text")
            except Exception as e:
                errores.append((nombre, str(e)))
        progreso.progress(1.0, text="Regeneración completada.")

        if errores:
            for nombre, mensaje in errores:
                st.error(f"No se pudo regenerar {nombre}: {mensaje}")
        else:
            st.success("Los 4 modelos se han regenerado correctamente.")

    st.markdown("**Regenerar individualmente:**")
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
    mostrar_seccion_logs()


pagina = st.sidebar.radio(
    "Navegación",
    ("Triaje", "Administración"),
)

if pagina == "Triaje":
    mostrar_pagina_triaje()
else:
    if requiere_admin():
        mostrar_pagina_administracion()
