#-*- coding: utf-8 -*-
import os
import pyaudio, wave, pylab
import librosa, librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.models import model_from_json
from keras import backend as K
from tkinter import *
from pygame import mixer
from PIL import Image, ImageTk

from settings import *
from src.sound import sound
from src.metrics import *

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    # os.execl(python, python, * sys.argv)
    os.execl(python, python, "-m", "src.classify", * sys.argv)

def classify():
    # Example of a Siren spectrogram
    try:
        y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr,)
        px=ps
        ps= np.array(ps.reshape(1,128,87,1))
        ps.shape
        predictions = loaded_model.predict_classes(ps)
        class_id=(predictions[0])
        print("The recorded chord is "+str(CLASSES[class_id]))
        result=canvas.create_text(250,150,text="The recorded chord is "+str(CLASSES[class_id]),font="Times 15 bold")
        display(px, y, sr)
    except:
        print("please record the sound first")

def display(ps, y, sr):
    filename = librosa.util.example_audio_file()
    y = y[:100000] # shorten audio a bit for speed
    window_size = 1024
    window = np.hanning(window_size)
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    # For plotting headlessly
    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title('Mel Spectrogram')
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig.savefig(SPECTROGRAM_FILE, dpi=50)
    image = Image.open(SPECTROGRAM_FILE)
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo # keep a reference!
    label.pack()

tk=Tk()
tk.title('Guitar Chord Classifier')
canvas_width=500
canvas_height=300

canvas=Canvas(tk,width=canvas_width,height=canvas_height)
canvas.pack()

# load json and create model
json_file = open(MODEL_JSON, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_H5)
print("Loaded model from disk")
loaded_model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy', precision, recall, fmeasure])
ox,oy=0,0

#Buttons
record=Button(canvas, text='Record',font="Times 15 bold", command=sound.record)
canvas.create_window(150, 50, window=record, height=25, width=100)

play=Button(canvas, text='Play',font="Times 15 bold", command=sound.play)
canvas.create_window(250, 50, window=play, height=25, width=50)

classify=Button(canvas, text='Classify',font="Times 15 bold", command=classify)
canvas.create_window(350, 50, window=classify, height=25, width=100)

clear=Button(canvas, text='Clear',font="Times 15 bold", command=restart_program)
canvas.create_window(250, 100, window=clear, height=25, width=100)

mainloop()
