import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
from src.model import CNN
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('app')

def init_model():
    cnn = CNN((128, 87))
    cnn.load_model()
    return cnn

def get_spectrogram(type='mel'):
    logger.info("Extracting spectrogram")
    y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=DURATION)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logger.info("Spectrogram Extracted")
    format = '%+2.0f'
    if type == 'DB':
        ps = librosa.power_to_db(ps, ref=np.max)
        format = ''.join[format, 'DB']
        logger.info("Converted to DB scale")
    return ps, format

def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)

def main():
    title = "Guitar Chord Recognition"
    st.title(title)
    image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    st.image(image, use_column_width=True)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        cnn = init_model()
        with st.spinner("Classifying the chord"):
            chord = cnn.predict(WAVE_OUTPUT_FILE, False)
        st.success("Classification completed")
        st.write("### The recorded chord is **", chord + "**")
        if chord == 'N/A':
            st.write("Please record sound first")
        st.write("\n")

    # Add a placeholder
    if st.button('Display Spectrogram'):
        # type = st.radio("Scale of spectrogram:",
        #                 ('mel', 'DB'))
        if os.path.exists(WAVE_OUTPUT_FILE):
            spectrogram, format = get_spectrogram(type='mel')
            display(spectrogram, format)
        else:
            st.write("Please record sound first")

if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)

