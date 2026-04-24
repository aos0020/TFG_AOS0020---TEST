# -*- coding: utf-8 -*-
import speech_recognition as sr

def capturar_voz_a_texto(idioma="es-ES", segundos_maximos=10):
    """
    Función genérica para transformar audio del micrófono en texto.
    
    Argumentos:
        idioma (str): Código de lenguaje (ej: "es-ES", "en-US").
        segundos_maximos (int): Tiempo máximo que se permite hablar.
        
    Retorna:
        str: El texto reconocido o None si hubo un error.
    """
    # 1. Inicializar el reconocedor
    reconocedor = sr.Recognizer()
    
    # 2. Configurar el micrófono como fuente de entrada
    with sr.Microphone() as origen:
        print("\n>>> Micrófono activado. Puedes hablar ahora...")
        
        # Calibración para ignorar el ruido de fondo (importante para precisión)
        reconocedor.adjust_for_ambient_noise(origen, duration=1)
        
        try:
            # Escucha el audio
            # timeout: tiempo que espera a que empieces a hablar
            # phrase_time_limit: duración máxima de la frase
            audio = reconocedor.listen(origen, timeout=5, phrase_time_limit=segundos_maximos)
            
            print(">>> Procesando audio...")
            
            # 3. Llamada al servicio de reconocimiento (Google Web Speech API)
            texto_detectado = reconocedor.recognize_google(audio, language=idioma)
            
            return texto_detectado

        except sr.WaitTimeoutError:
            print("!!! Error: Se agotó el tiempo de espera (no hablaste).")
        except sr.UnknownValueError:
            print("!!! Error: No se pudo entender el audio (ruido o voz poco clara).")
        except sr.RequestError as e:
            print(f"!!! Error de conexión con el servicio: {e}")
        except Exception as e:
            print(f"!!! Error inesperado: {e}")
            
    return None

# --- EJEMPLO DE USO INDEPENDIENTE ---
if __name__ == "__main__":
    resultado = capturar_voz_a_texto()
    
    if resultado:
        print(f"\nRESULTADO FINAL: {resultado}")
        # Aquí es donde conectarías con tu modelo:
        # prediccion = mi_modelo.predict(resultado)
    else:
        print("\nNo se obtuvo ningún texto.")