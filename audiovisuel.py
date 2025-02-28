from gtts import gTTS
import sys
sys.path.append(r"C:\Users\khali\OneDrive\Bureau\gen ia\GenAI-GCP\exercices\tp_4\tp4\Lib\site-packages")
from moviepy.editor import *
import os
import re


def clean_text(text):
    
    cleaned_text = re.sub(r'[^\w\s.,?!]', '', text)
    return cleaned_text

def generate_video(response_text, language="fr", output_file="response_video.mp4"):
    """
    Génère une vidéo avec audio à partir du texte fourni dans plusieurs langues.
    :param response_text: Texte à convertir en audio/vidéo.
    :param language: Langue pour la synthèse vocale (fr, en, ar, ru, es).
    :param output_file: Nom du fichier vidéo généré.
    :return: Chemin du fichier vidéo.
    """
    try:
        response_text = clean_text(response_text)
        supported_languages = ['fr', 'en', 'ar', 'es']
        if language not in supported_languages:
            raise ValueError(f"Langue '{language}' non supportée. Choisissez parmi : {', '.join(supported_languages)}.")

        tts = gTTS(response_text, lang=language)
        audio_file = "response_audio.mp3"
        tts.save(audio_file)
        
        video_clip = ImageClip("heart.jpg", duration=10)
        audio_clip = AudioFileClip(audio_file)
        video_clip = video_clip.set_audio(audio_clip)

        video_clip.write_videofile(output_file, fps=24)

        os.remove(audio_file)

        return output_file
    except Exception as e:
        print(f"Erreur lors de la génération de la vidéo : {e}")
        return None

