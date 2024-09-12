import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from elevenlabs import VoiceSettings,play
from elevenlabs.client import ElevenLabs
from io import BytesIO
from decouple import config 
from groq_translation import groq_translate 


groq_api_key = config("GROQ_API_KEY")
eleven_api_key = config("ELEVEN_API_KEY")
groq_client = Groq(api_key=groq_api_key)
elevenlabs_client = ElevenLabs(api_key=eleven_api_key)

def main():
    # Set page config
    st.set_page_config(page_title='Groq Translator', page_icon='🌎')
    # Set page title
    st.title('Groq Translator')

    languages = {
   "Portuguese": "pt",
   "Spanish": "es",
   "German": "de",
   "French": "fr",
   "Italian": "it",
   "Dutch": "nl",
   "Russian": "ru",
   "Japanese": "ja",
   "Chinese": "zh",
   "Korean": "ko"
   }

   # Language selection
    option = st.selectbox(
        "Language to translate to:",
        languages,
        index=None,
        placeholder="Select language...")

    # Record audio
    audio_bytes = audio_recorder()
    if audio_bytes and option:
        # Display audio player
        st.audio(audio_bytes, format="audio/wav")
        
        # Save audio to file
        audio_file_like = BytesIO(audio_bytes)
        #Perform STT
        text = speech_to_text(audio_file_like)
        st.divider() 
        with st.spinner('Translating...'): 
            translation = groq_translate(text, 'en', option) # Perform Translation
        st.subheader('Translated Text to ' + option) 
        st.write(translation.text)

def speech_to_text(audio_bytes_io): 
    transcription = groq_client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes_io.read()),
        model="distil-whisper-large-v3-en", # WhisperModel
        response_format="json",
        language="en",
        )
    return transcription.text

if __name__ == "__main__":
    main()
