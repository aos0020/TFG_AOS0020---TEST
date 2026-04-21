import subprocess
import sys

def clasificar_tfidf(texto):
    try:
            # Llamamos al intérprete de python actual y al script externo
            # Pasamos el texto de los síntomas como un argumento de línea de comandos
            resultado = subprocess.run(
                [sys.executable, "TF-IDF/tf_idf.py", texto],
                text=True,
                capture_output=True,
                check=True
            )
            # Devolvemos toda la salida (stdout y stderr) para mostrar mensajes de depuración
            salida = resultado.stdout.strip()
            if resultado.stderr:
                salida += "\n[STDERR]:\n" + resultado.stderr.strip()
            return salida
    except Exception as e:
        return f"Error al ejecutar el clasificador: {e}"