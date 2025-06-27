import streamlit as stl
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import os 
import time

stl.set_page_config(
    page_title="Traductor de voz Español a Francés",
    page_icon=":microphone:",
    layout="wide",
)

stl.title("Traductor de voz Español a Francés")
stl.markdown("""
            Esta aplicación usa un modelo de traducción de voz de español a francés""")