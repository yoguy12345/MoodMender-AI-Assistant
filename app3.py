import streamlit as st
from openai import OpenAI
import boto3
from streamlit_mic_recorder import speech_to_text
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time

# Conditional import for DeepFace
try:
    from deepface import DeepFace
except ImportError:
    st.error("Failed to import DeepFace. Please make sure it's installed correctly.")
    DeepFace = None

# Initialize Polly client
polly = boto3.client('polly', region_name='us-east-1')  # Replace with your preferred region

# Popular AWS Polly American voices (3 Female, 3 Male)
popular_voices = [
    ('Joanna', 'Female'),
    ('Kendra', 'Female'),
    ('Ivy', 'Female'),
    ('Matthew', 'Male'),
    ('Justin', 'Male'),
    ('Joey', 'Male')
]

def synthesize_speech(text, output_filename='response.mp3', voice_id='Joanna'):
    """Synthesize speech using AWS Polly."""
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=voice_id,
        Engine='standard'
    )
    with open(output_filename, 'wb') as file:
        file.write(response['AudioStream'].read())
    return output_filename


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion = "Neutral"
        self.raw_emotion_data = {}
        self.last_detection_time = 0
        self.detection_interval = 3  # Perform emotion detection every 3 seconds

    def analyze_emotion(self, img):
        if DeepFace is None:
            return

        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            self.emotion = result[0]['dominant_emotion'].capitalize()
            self.raw_emotion_data = result[0]['emotion']
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            self.emotion = "Error"
            self.raw_emotion_data = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_interval and len(faces) > 0:
            self.analyze_emotion(img)
            self.last_detection_time = current_time

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"Emotion: {self.emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Pocket Therapist with Emotion Scores")

    if DeepFace is None:
        st.error("DeepFace is not available. Emotion detection will not work.")
        return

    # Dropdown for selecting voice
    selected_voice = st.selectbox("Select Voice", [voice[0] for voice in popular_voices])

    # Placeholder for emotion display
    emotion_placeholder = st.empty()
    raw_emotion_placeholder = st.empty()

    # Continuous video streaming and emotion detection
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    # Speech-to-text conversion
    st.write("Convert speech to text:")
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    if text:
        st.session_state.text_received = text

    if 'text_received' in st.session_state and st.session_state.text_received:
        # Display the transcribed text
        st.text(st.session_state.text_received)

        prompt = f"You are a trained psychotherapist, specializing in providing stress management strategies for people with ADHD. Give short responses for every query, less than 4 sentences. Currently, the client's emotion is estimated as {ctx.video_processor.emotion if ctx.video_processor else 'Unknown'}"

        # OpenAI API interaction
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": st.session_state.text_received}
            ]
        )
        response_text = completion.choices[0].message.content
        st.write(response_text)  # Display the response

        # Convert the response text to speech and play it
        mp3_file = synthesize_speech(response_text, voice_id=selected_voice)
        st.audio(mp3_file)

    # Update emotion display
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                emotion_placeholder.write(f"Detected Emotion: {ctx.video_processor.emotion}")
                raw_emotion_placeholder.json(ctx.video_processor.raw_emotion_data)
            time.sleep(0.1)  # Update every 100ms

if __name__ == "__main__":
    main()