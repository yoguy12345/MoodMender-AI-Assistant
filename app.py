import streamlit as st
from openai import OpenAI
import boto3
from streamlit_mic_recorder import speech_to_text
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
import os
from dotenv import load_dotenv
import face_recognition
from PIL import Image, ImageDraw

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize Polly client
polly = boto3.client('polly', region_name=AWS_REGION)

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
        self.emotion = "Neutral"
        self.raw_emotion_data = {}
        self.last_detection_time = 0
        self.detection_interval = 3  # Perform emotion detection every 3 seconds

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil_img = Image.fromarray(img)
        
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_interval:
            face_locations = face_recognition.face_locations(img)
            if face_locations:
                # Here you would typically do emotion detection
                # For now, we'll just assume a neutral emotion
                self.emotion = "Neutral"
                self.raw_emotion_data = {"Neutral": 1.0}
            self.last_detection_time = current_time

        draw = ImageDraw.Draw(pil_img)
        for (top, right, bottom, left) in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)
            draw.text((left, top - 20), f"Emotion: {self.emotion}", fill=(0, 255, 0))

        return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

def get_top_emotions(emotion_data, n=2):
    if not emotion_data:
        return ["No emotion detected"]
    sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
    return [emotion.capitalize() for emotion, _ in sorted_emotions[:n]]

def main():
    st.title("Pocket Therapist with Emotion Scores")

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

        # Get top 2 emotions
        if ctx.video_processor:
            top_emotions = get_top_emotions(ctx.video_processor.raw_emotion_data, n=2)
        else:
            top_emotions = ["No emotion detected"]
        
        emotions_str = " and ".join(top_emotions)

        prompt = f"""You are a trained psychotherapist, specializing in providing stress management strategies for people with ADHD. 
        Give short responses for every query, less than 4 sentences. 
        Currently, the client's top detected emotions are {emotions_str}. 
        Start your response with 'I detect [emotions],' where [emotions] are the top detected emotions, then continue with your advice.
        If no emotions are detected, start with 'I detect no emotion,' and provide general advice.
        """

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
            else:
                emotion_placeholder.write("No emotion detected")
                raw_emotion_placeholder.json({})
            time.sleep(0.1)  # Update every 100ms

if __name__ == "__main__":
    main()
