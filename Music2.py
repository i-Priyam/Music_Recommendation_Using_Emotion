import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Set page configuration
st.set_page_config(page_title="🎶 Emotion-Based Music Recommender", layout="centered")

# 🎨 Custom CSS for better visuals
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

    .stTextInput label, .stSelectbox label {
        color: #f9f9f9;
        font-weight: bold;
    }

    .stTextInput > div > input, .stSelectbox > div > div {
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        border-radius: 5px;
    }

    .stButton > button {
        background-color: #ff6347;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 25px;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #e5533d;
        color: #ffffff;
        cursor: pointer;
    }

    .frosted-glass {
        backdrop-filter: blur(12px);
        background-color: rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }

    .playlist-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 20px;
    }

    .playlist-container iframe {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🎧 Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)

# Load model and labels
model = load_model("emo_model.h5")
label = np.load("labels.npy")

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Emotion detection class
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
            confidence = prediction[0][pred_idx] * 100  # Confidence percentage
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

btn = st.button("🔍 Recommend Me Songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first by allowing webcam access.")
        st.session_state["run"] = "true"
    else:
        search_query = f"{lang} {emotion} songs {singer}".replace(" ", "%20")
        if platform == "Spotify":
            url = f"https://open.spotify.com/search/{search_query}"
        elif platform == "JioSaavn":
            url = f"https://www.jiosaavn.com/search/{search_query}"
        elif platform == "Gaana":
            url = f"https://gaana.com/search/{search_query}"
        elif platform == "Wynk":
            url = f"https://wynk.in/music/search/{search_query}"
        elif platform == "YouTube Music":
            url = f"https://music.youtube.com/search?q={search_query}"
        elif platform == "SoundCloud":
            url = f"https://soundcloud.com/search?q={search_query}"
        elif platform == "Hungama":
            url = f"https://www.hungama.com/search/all/{search_query}/songs/"
        elif platform == "Apple Music":
            url = f"https://music.apple.com/us/search?term={search_query}"
        webbrowser.open(url)
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

# Playlist showcase
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
