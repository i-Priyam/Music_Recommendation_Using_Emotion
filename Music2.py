import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import json
import os

# Set page configuration
st.set_page_config(page_title="🎶 Emotion-Based Music Recommender", layout="centered")

# 🎨 Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2021/03/23/07/42/headphones-6116859_1280.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }

    h1 {
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        text-align: center;
    }

    .chatbot-container {
        background: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 15px;
        margin-top: 30px;
        color: black;
    }

    .chat-message-user {
        color: #ff6347;
        font-weight: bold;
    }

    .chat-message-bot {
        color: #2e8b57;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎧 Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)

# Load model and labels safely
if os.path.exists("emo_model.h5") and os.path.exists("labels.npy"):
    model = load_model("emo_model.h5", compile=False)
    label = np.load("labels.npy")
else:
    st.error("Model or label file not found. Please ensure 'emo_model.h5' and 'labels.npy' exist.")
    st.stop()

# Emotion quotes and lyrics
emotion_quotes_lyrics = {
    "Happy": {
        "quote": "Happiness is not by chance, but by choice. – Jim Rohn",
        "lyric": "Because I'm happy — Clap along if you feel like a room without a roof. 🎶"
    },
    "Sad": {
        "quote": "Tears come from the heart and not from the brain. – Leonardo da Vinci",
        "lyric": "Hello from the other side... 🎶"
    },
    "Angry": {
        "quote": "Speak when you are angry and you will make the best speech you will ever regret. – Ambrose Bierce",
        "lyric": "Let it go, let it go, can't hold it back anymore... 🎶"
    },
    "Surprised": {
        "quote": "Surprise is the greatest gift life can grant us. – Boris Pasternak",
        "lyric": "Is this the real life? Is this just fantasy? 🎶"
    },
    "Neutral": {
        "quote": "Life is what happens when you're busy making other plans. – John Lennon",
        "lyric": "I'm just a small town girl, living in a lonely world... 🎶"
    }
}

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Session states
if "run" not in st.session_state:
    st.session_state["run"] = "true"
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load emotion safely
if os.path.exists("emotion.npy"):
    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion = ""
else:
    np.save("emotion.npy", np.array([""]))
    emotion = ""

# Manual override
manual_emotion = st.selectbox("😌 Select Emotion Manually (Optional)", ["", "Happy", "Sad", "Angry", "Surprised", "Neutral"])
if manual_emotion:
    emotion = manual_emotion
    st.session_state["run"] = "false"

# Emotion detection processor
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            prediction = model.predict(lst)
            pred_idx = np.argmax(prediction)
            pred = label[pred_idx]
            confidence = prediction[0][pred_idx] * 100
            cv2.putText(frm, f"{pred} ({confidence:.2f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Inputs
lang = st.text_input("🎤 Preferred Language")
singer = st.text_input("🎼 Favorite Singer")
platform = st.selectbox("🎧 Choose Music Platform", [
    "Spotify", "JioSaavn", "Gaana", "Wynk", "YouTube Music", "SoundCloud", "Hungama", "Apple Music"
])

if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

if emotion:
    st.success(f"🎯 Detected Emotion: {emotion}")
    if emotion in emotion_quotes_lyrics:
        st.markdown("### 💬 Quote & 🎵 Lyric")
        st.info(f"**Quote:** {emotion_quotes_lyrics[emotion]['quote']}")
        st.success(f"**Lyric:** {emotion_quotes_lyrics[emotion]['lyric']}")

# Recommend songs
if st.button("🔍 Recommend Me Songs"):
    if not emotion:
        st.warning("Please let me capture your emotion first by allowing webcam access.")
        st.session_state["run"] = "true"
    else:
        query = f"{lang} {emotion} songs {singer}".replace(" ", "%20")
        urls = {
            "Spotify": f"https://open.spotify.com/search/{query}",
            "JioSaavn": f"https://www.jiosaavn.com/search/{query}",
            "Gaana": f"https://gaana.com/search/{query}",
            "Wynk": f"https://wynk.in/music/search/{query}",
            "YouTube Music": f"https://music.youtube.com/search?q={query}",
            "SoundCloud": f"https://soundcloud.com/search?q={query}",
            "Hungama": f"https://www.hungama.com/search/all/{query}/songs/",
            "Apple Music": f"https://music.apple.com/us/search?term={query}"
        }
        webbrowser.open(urls[platform])
        st.session_state["favorites"].append(f"{lang} - {emotion} - {singer} ({platform})")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

# Save favorites
with open("favorites.json", "w") as f:
    json.dump(st.session_state["favorites"], f)

if st.session_state["favorites"]:
    st.markdown("### ❤️ Your Saved Searches")
    for fav in reversed(st.session_state["favorites"]):
        st.markdown(f"- {fav}")

# Playlists
st.markdown("""
    <div class="frosted-glass">
        <h3>🎶 Explore Featured Global Playlists</h3>
        <div class="playlist-container">
            <iframe src="https://open.spotify.com/embed/playlist/37i9dQZF1DWXRqgorJj26U" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
            <iframe width="100%" height="80" src="https://www.youtube.com/embed/videoseries?list=PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
            <iframe width="100%" height="80" scrolling="no" frameborder="no" allow="autoplay"
                src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/playlists/209262931">
            </iframe>
        </div>
    </div>
""", unsafe_allow_html=True)

