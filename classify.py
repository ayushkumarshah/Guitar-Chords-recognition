#-*- coding: utf-8 -*-
from keras.models import model_from_json
from keras import backend as K
import librosa
import librosa.display
import numpy as np
import os
from tkinter import *
from pygame import mixer
mixer.init(44100)
from PIL import Image, ImageTk
import pandas as pd
tk=Tk()

canvas_width=500
canvas_height=300

canvas=Canvas(tk,width=canvas_width,height=canvas_height)
canvas.pack() 

import matplotlib.pyplot as plt
import pylab


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)

def record():
    import pyaudio
    import wave

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "recorded.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def classify():
    # Example of a Siren spectrogram
    y, sr = librosa.load('recorded.wav',duration=2)
    ps = librosa.feature.melspectrogram(y=y, sr=sr,)
    px=ps
    ps.shape
    ps= np.array(ps.reshape(1,128,87,1))
    ps.shape
    predictions = loaded_model.predict_classes(ps)
    class_id=(predictions[0])
    print("The recorded chord is "+str(classes[class_id]))
    result=canvas.create_text(800,270,text="The recorded chord is "+str(classes[class_id]),font="Times 15 bold")
    display(px)

def display(ps):
    filename = librosa.util.example_audio_file()
    y, sr = librosa.load('recorded.wav',duration=2)
    y = y[:100000] # shorten audio a bit for speed
    window_size = 1024
    window = np.hanning(window_size)
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    # For plotting headlessly
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig.savefig('test.png',dpi=50)
    image = Image.open("test.png")
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo # keep a reference!
    label.pack()



def play():
    mixer.Sound("recorded.wav").play()
    
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
classes=['a','am','bm','c','d','dm','e','em','f','g']
        
loaded_model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy', precision, recall,fmeasure])
          

ox,oy=0,0


#Buttons
record=Button(canvas, text='Record',font="Times 15 bold", command=record)
canvas.create_window(150, 50, window=record, height=25, width=100)

play=Button(canvas, text='Play',font="Times 15 bold", command=play)
canvas.create_window(250, 50, window=play, height=25, width=50)

classify=Button(canvas, text='Classify',font="Times 15 bold", command=classify)
canvas.create_window(350, 50, window=classify, height=25, width=100)

clear=Button(canvas, text='Clear',font="Times 15 bold", command=restart_program)
canvas.create_window(250, 100, window=clear, height=25, width=100)



mainloop()
