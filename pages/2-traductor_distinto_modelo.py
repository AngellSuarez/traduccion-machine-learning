import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Traductor de voz Español → Francés",
    page_icon="🎧",
    layout="centered"
)

st.title("🎤 Traductor de voz Español → Francés")
st.markdown("Puedes **grabar tu voz** o **subir un archivo de audio** en español, y el sistema lo traducirá al francés con voz.")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    model_name = "Helsinki-NLP/opus-mt-es-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Captura desde micrófono
def capturar_audio(duracion=5):
    st.info(f"🎙️ Escuchando durante {duracion} segundos...")
    fs = 16000  # Frecuencia de muestreo
    try:
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio, fs)
            r = sr.Recognizer()
            with sr.AudioFile(tmpfile.name) as source:
                audio_data = r.record(source)
                texto = r.recognize_google(audio_data, language="es-ES")
                return texto
    except sr.UnknownValueError:
        st.warning("⚠️ No se pudo entender el audio.")
    except sr.RequestError as e:
        st.error(f"❌ Error con el servicio de reconocimiento de voz: {e}")
    except Exception as e:
        st.error(f"❌ Error al capturar el audio: {e}")
    return None

# Transcripción desde archivo subido
def transcribir_audio_subido(archivo_audio):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(archivo_audio.read())
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)
                texto = recognizer.recognize_google(audio_data, language="es-ES")
                return texto
    except sr.UnknownValueError:
        st.warning("⚠️ No se pudo entender el audio.")
    except sr.RequestError as e:
        st.error(f"❌ Error de conexión: {e}")
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
    return None

# Traducción
def traducir_texto(texto):
    try:
        tokenizer, model = cargar_modelo()
        inputs = tokenizer(texto, return_tensors="pt", truncation=True)
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Error al traducir: {str(e)}")
        return None

# Texto a voz en francés
def texto_a_voz(texto_fr):
    try:
        tts = gTTS(text=texto_fr, lang="fr")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"❌ Error al generar el audio: {str(e)}")
        return None

# Interfaz principal
def main():
    metodo = st.radio("Selecciona el método de entrada:", ["🎙️ Grabar desde micrófono", "📁 Subir archivo de audio (.wav o .mp3)"])

    texto_es = None

    if metodo == "🎙️ Grabar desde micrófono":
        if st.button("🎤 Grabar ahora"):
            texto_es = capturar_audio()

    elif metodo == "📁 Subir archivo de audio (.wav o .mp3)":
        archivo = st.file_uploader("Sube tu archivo de audio", type=["wav", "mp3"])
        if archivo is not None:
            st.audio(archivo)
            texto_es = transcribir_audio_subido(archivo)

    if texto_es:
        st.success(f"📝 Texto reconocido: {texto_es}")

        with st.spinner("🌐 Traduciendo al francés..."):
            texto_fr = traducir_texto(texto_es)

        if texto_fr:
            st.success(f"📘 Traducción: {texto_fr}")
            audio = texto_a_voz(texto_fr)
            if audio:
                st.audio(audio, format="audio/mp3")
                if st.button("🔁 Repetir audio"):
                    st.audio(audio, format="audio/mp3")

main()
