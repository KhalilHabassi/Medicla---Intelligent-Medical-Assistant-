import streamlit as st
import time
import requests
import os
from PIL import Image, ImageDraw, ImageOps
from typing import List, Dict
from dotenv import load_dotenv
from audiovisuel import generate_video
import google.generativeai as genai


load_dotenv()


HOST = "http://localhost:8181"


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Clé API Gemini Pro non trouvée dans le fichier .env")

def generate_refined_response(query, original_response, language="en"):
    """Raffine la réponse originale en utilisant Gemini Pro 1.5 dans la langue spécifiée"""
    if not GEMINI_API_KEY:
        return "Erreur: Clé API Gemini Pro manquante dans le fichier .env"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        language_instructions = {
            "en": "Please provide your response in English.",
            "fr": "Veuillez fournir votre réponse en français.",
            "ar": "يرجى تقديم إجابتك باللغة العربية.",
            "es": "Por favor proporcione su respuesta en español."
        }
        
        prompt = f"""
        En tant qu'expert médical, j'aimerais que tu raffines et reformules la réponse suivante 
        à une question médicale. Améliore la clarté, la précision et la structure.
        
        Question: {query}
        
        Réponse originale: {original_response}
        
        Veuillez fournir une version raffinée et expertisée de cette réponse. 
        Conservez les informations importantes mais améliorez la qualité de l'explication médicale.
        
        {language_instructions.get(language, language_instructions["en"])}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur lors de la génération de la réponse raffinée: {str(e)}"


# Page d'accueil

def create_homepage():
    st.set_page_config(
        page_title="Medicla Chatbot",
        page_icon=":medical_symbol:",
        layout="centered"
    )

    st.markdown("""
        <style>
        /* Style général */
        .title {
            font-size: 50px !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 30px 0 !important;
            color: #0B5394 !important; /* Bleu médical */
            animation: fadeIn 1.5s ease-in;
        }
        .welcome-text {
            font-size: 24px !important;
            text-align: center !important;
            color: #073763 !important; /* Bleu foncé */
            margin-bottom: 20px !important;
        }
        .description-box {
            background-color: #E7F3FE;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            animation: slideUp 1s ease-out;
        }
        .start-button {
            text-align: center;
            padding: 20px 0;
            animation: pulse 2s infinite;
        }
        .feature-item {
            color: #0B5394 !important;
            font-size: 18px !important;
            padding: 8px 0 !important;
            text-align: center !important;
        }
        .license {
            font-size: 14px !important;
            color: #555 !important;
            text-align: center !important;
            margin-top: 40px !important;
            padding-top: 10px !important;
            border-top: 1px solid #ccc !important;
        }
        /* Style du bouton */
        .stButton button {
            background-color: #0B5394 !important;
            color: white !important;
            border: none !important;
            padding: 15px 30px !important;
            font-size: 18px !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            background-color: #073763 !important;
            transform: translateY(-2px) !important;
        }
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideUp {
            from { 
                transform: translateY(50px);
                opacity: 0;
            }
            to { 
                transform: translateY(0);
                opacity: 1;
            }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        /* Pour le logo */
        .logo {
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    img = Image.open("Flux_Dev_A_beautifully_poised_Latin_woman_embodying_the_essenc_1.jpeg").convert("RGBA")
    #img = Image.open("flux.jpeg").convert("RGBA")

    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img_cropped = img.crop((left, top, right, bottom))

    mask = Image.new("L", (min_dim, min_dim), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, min_dim, min_dim), fill=255)

    img_circular = ImageOps.fit(img_cropped, (min_dim, min_dim))
    img_circular.putalpha(mask)

    st.image(img_circular, width=250)
   
   
   
    st.markdown('<h1 class="title">Medicla </h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="description-box">', unsafe_allow_html=True)
        st.markdown('''
            <p class="welcome-text">
                Bienvenue sur Medicla 👩‍⚕️👨‍⚕️, votre assistant intelligent d'analyse de données médicales.
                Grâce à notre approche RAG (Retrieval Augmented Generation), explorez et comprenez vos données cliniques 
                et médicales de manière intuitive et sécurisée.
            </p>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🩺 Medicla votre Chatbot médical 🩺"):
        features = [
            "🔍 Interrogez vos données médicaux en quelques clics",
            "💬 Posez vos questions en langage naturel",
            "📊 Obtenez une image descriptive d'une question"
        ]
        for feature in features:
            st.markdown(f'<p class="feature-item">{feature}</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="start-button">', unsafe_allow_html=True)
    if st.button("Commencez votre exploration", type="primary", use_container_width=True):
        with st.spinner('Préparation de votre espace de travail...'):
            time.sleep(1.5)
            st.session_state.page = "chat"
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="license">', unsafe_allow_html=True)
    st.markdown('<p>© 2025 Medicla DataExplorer. Tous droits réservés.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



# Interface Chatbot


def chatbot_interface():

    st.markdown("""
    <style>
    /* Titre du chatbot */
    .chat-title {
        font-size: 42px !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #0B5394 !important; /* Bleu médical */
        margin-top: 20px;
    }
    /* Style général des messages */
    .stChatMessage p {
        font-size: 18px !important;
        line-height: 1.5;
    }
    /* Animation légère pour les messages entrants */
    .stChatMessage {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="chat-title">Medical RAG Assistant 👩‍⚕️👨‍⚕️</h2>', unsafe_allow_html=True)

    with st.sidebar:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        language = st.selectbox('Language', ['en', 'fr', 'ar', 'es'])
        data_source = st.selectbox(
            'Data Source', 
            ['Presentations (PPTX)', 'Medical Knowledge (MedQuAD)']
        )
        data_source_to_table = {
            'Presentations (PPTX)': 'kh_table',
            'Medical Knowledge (MedQuAD)': 'ae_qa_table'
        }
        if data_source == 'Medical Knowledge (MedQuAD)':
            st.info("Cette source contient des questions/réponses médicales provenant d'autorités 🩺.")
        else:
            st.info("Cette source contient des informations issues de vos slides de présentation.")
        
        audio_language = st.selectbox(
            "Audio language for the response:",
            options={"fr": "French", "en": "English", "ar": "Arabic", "es": "Spanish"}
        )
    

    for tab_name in ["standard_messages", "refined_messages"]:
        if tab_name not in st.session_state:
            st.session_state[tab_name] = [{"role": "assistant", "content": "HELLO 👋, Please write your question."}]
    

    tab1, tab2 = st.tabs(["Question/Réponse", "Réponse Raffinée (Gemini Pro)"])
    
    with tab1:
        for message in st.session_state.standard_messages:
            avatar = "🤖" if message["role"] == "assistant" else "🧑‍⚕️"
            st.chat_message(message["role"], avatar=avatar).write(message["content"])
        
        if "sources" in st.session_state and len(st.session_state.standard_messages) > 1:
            docs = st.session_state.sources
            for i, doc in enumerate(docs):
                with st.expander(f"Source {i+1}"):
                    st.write("Metadata:")
                    st.write(doc.get("source", ""), doc.get("focus_area", ""))
                    st.write("Similarity Score:", doc.get("similarity_score"))
                    st.write("Content:")
                    st.write(doc.get("content", ""))

            if len(st.session_state.standard_messages) > 1 and st.session_state.standard_messages[-1]["role"] == "assistant":
                st.write("Generating audiovisual response... 🎥")
                video_path = generate_video(st.session_state.standard_messages[-1]["content"], language=audio_language)
                if video_path:
                    st.video(video_path)
                else:
                    st.error("Failed to generate the video.")
            
    with tab2:
        for message in st.session_state.refined_messages:
            avatar = "🧞‍♂️" if message["role"] == "assistant" else "🧑‍⚕️"
            st.chat_message(message["role"], avatar=avatar).write(message["content"])
            
        if len(st.session_state.refined_messages) > 1 and st.session_state.refined_messages[-1]["role"] == "assistant":
            st.write("Generating audiovisual response... 🎥")
            video_path = generate_video(st.session_state.refined_messages[-1]["content"], language=audio_language)
            if video_path:
                st.video(video_path)
            else:
                st.error("Failed to generate the video.")
    
    if question := st.chat_input("What is your question?"):
        st.session_state.standard_messages.append({"role": "user", "content": question})
        st.session_state.refined_messages.append({"role": "user", "content": question})
        
        response = requests.post(
            f"{HOST}/answer_from_table",
            json={
                "question": question,
                "temperature": temperature,
                "lang": language
            },
            timeout=20
        )
        
        sources_response = requests.get(
            f"{HOST}/get_sources",
            params={
                "question": question,
                "temperature": temperature,
                "lang": language
            },
            timeout=20
        )
        
        if response.status_code == 200:
            standard_answer = response.json().get("answer", "No answer provided.")
            
            st.session_state.standard_messages.append({"role": "assistant", "content": standard_answer})
            
            refined_answer = generate_refined_response(question, standard_answer, language=language)
            st.session_state.refined_messages.append({"role": "assistant", "content": refined_answer})
            
            if sources_response.status_code == 200:
                st.session_state.sources = sources_response.json()
            else:
                st.error(f"Error: Unable to get sources from the API. Status: {sources_response.status_code}")
                st.error(sources_response.text)
            
            st.rerun()
        else:
            st.error(f"Error: Unable to get a response from the API. Status: {response.status_code}")
            st.error(response.text)
            return


def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        create_homepage()
    elif st.session_state.page == "chat":
        chatbot_interface()

if __name__ == "__main__":
    main()