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
st.markdown("Graba o sube un audio en español (.wav o .mp3) y lo traduciremos a francés con voz.")

# Cargar modelo de traducción
@st.cache_resource
def cargar_modelo():
    model_name = "Helsinki-NLP/opus-mt-es-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Transcripción de audio a texto (español)
def transcribir_audio(archivo_audio):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(archivo_audio.read())
            tmp_path = tmp.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            texto = recognizer.recognize_google(audio_data, language="es-ES")
            return texto
    except sr.UnknownValueError:
        st.warning("⚠️ No se pudo entender el audio.")
    except sr.RequestError as e:
        st.error(f"❌ Error de conexión con el servicio de reconocimiento de voz: {e}")
    except Exception as e:
        st.error(f"❌ Error al procesar el audio: {e}")
    return None

# Grabación con micrófono
def grabar_desde_microfono(duracion=5):
    st.info(f"🎙️ Grabando durante {duracion} segundos...")
    fs = 16000
    try:
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio, fs)
            with open(tmpfile.name, "rb") as f:
                return f.read()
    except Exception as e:
        st.error(f"❌ Error al grabar: {str(e)}")
        return None

# Traducción español → francés
def traducir_texto(texto_espanol):
    if not texto_espanol:
        return None
    try:
        tokenizer, model = cargar_modelo()
        inputs = tokenizer(texto_espanol, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Error al traducir: {e}")
        return None

# Conversión de texto a voz (francés)
def texto_a_voz_frances(texto_fr):
    try:
        tts = gTTS(text=texto_fr, lang="fr")
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"❌ Error al generar audio: {str(e)}")
        return None

# Interfaz principal
def main():
    st.markdown("### 🟢 Elige una opción:")
    metodo = st.radio("Entrada de audio:", ["🎙️ Grabar con micrófono", "📁 Subir archivo .wav/.mp3"])

    if metodo == "🎙️ Grabar con micrófono":
        if st.button("🎤 Grabar 5 segundos"):
            audio_bytes = grabar_desde_microfono()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                texto_es = transcribir_audio(BytesIO(audio_bytes))
            else:
                texto_es = None

    else:
        archivo = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3"])
        if archivo:
            st.audio(archivo)
            texto_es = transcribir_audio(archivo)
        else:
            texto_es = None

    if texto_es:
        st.success(f"📝 Texto reconocido: {texto_es}")
        with st.spinner("🌐 Traduciendo..."):
            texto_fr = traducir_texto(texto_es)

        if texto_fr:
            st.success(f"📘 Traducción: {texto_fr}")
            buffer = texto_a_voz_frances(texto_fr)
            if buffer:
                st.audio(buffer, format="audio/mp3")

                if st.button("🔁 Repetir audio"):
                    st.audio(buffer, format="audio/mp3")

main()
