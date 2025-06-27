import streamlit as stl
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from io import BytesIO

stl.set_page_config(
    page_title="Traductor de voz Español a Francés",
    page_icon=":microphone:",
    layout="centered",
)

#Titulo
stl.title("Traductor de voz Español a Francés")
#subtitulo
stl.markdown("""Esta aplicación usa un modelo de traducción de voz de español a francés cuando hables""")

#funcion para capturar el audio
def capturar_audio(duracion=5):
    stl.info(f"Escuchando..... {duracion} segundos")
    fs = 16000  # Frecuencia de muestreo
    try:
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio, fs)
            r = sr.Recognizer()
            with sr.AudioFile(tmpfile.name) as source:
                audio_data = r.record(source)
                texto = r.recognize_google(audio_data, laguage="es-ES")
                return texto
    except sr.UnknownValueError:
        stl.warning("No se pudo entender el audio. Por favor, intenta de nuevo.")
    except sr.RequestError as e:
        stl.error(f"Error al conectar con el servicio de reconocimiento de voz: {e}")
    except Exception as e:
        stl.error(f"Ocurrió un error al capturar el audio: {e}")
    return None

#funcion para la carga del modelo a cargo de la traduccion del  texto
@stl.cache_resource
def cargar_modelo():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    #aqui se encuentran los codigos del idioma
    tokenizer.lang_code_to_id = {
        #español pero latino
        "spa_Latn": tokenizer.convert_tokens_to_ids("spa_Latn"),
        "fra_Latn": tokenizer.convert_tokens_to_ids("fra_Latn"),
    }
    return tokenizer, model

#funcion que se encarga de la traduccion del texto
def traducir_texto(texto_entrada, idioma_origen="spa_Latn",idioma_destino="fra_Latn"):
    if not texto_entrada:
        return None
    try:
        tokenizer, model = cargar_modelo()
        
        tokenizer.src_lang = idioma_origen
        inputs = tokenizer(texto_entrada, return_tensors="pt",truncation=True)
        
        salida = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[idioma_destino]
        )
        
        return tokenizer.decode(salida[0], skip_special_tokens=True)
    
    except Exception as e:
        stl.error(f"Error en la traducción: {str(e)}")
        return None
    
#funcion que se encarga de reproducir el texto como un audio en frances
def reproducir_audio_frances(texto, idioma="fr"):
    try:
        tts = gTTS(text=texto, lang=idioma)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        stl.error(f"Error al generar el audio: {str(e)}")
        return None
    
def main():
    if stl.button("Presiona para hablar", type="primary"):
        
        with stl.spinner("Capturando audio..."):
            texto_es = capturar_audio()
            
            if texto_es:
                stl.success(f"Texto capturado: {texto_es}")
                
                with stl.spinnner("Traduciendo..."):
                   texto_fr = traducir_texto(texto_es)
                
                if texto_fr:
                    stl.success(f"Traducción al francés: {texto_fr}")
                    
                    buffer = reproducir_audio_frances(texto_fr)
                    if buffer:
                        stl.audio(buffer, format="audio/mp3")
                        
                        if stl.button("Repetir pronunciación"):
                            stl.audio(buffer, format="audio/mp3")
                            
main()
                   