import pyaudio
audio = pyaudio.PyAudio()
num_devices = audio.get_device_count()

for i in range(num_devices):
    info_dict = audio.get_device_info_by_index(i)
    keys = ['name', 'index', 'maxInputChannels', 'defaultSampleRate']
    print([(key, value) for key, value in info_dict.items() if key in keys])
    print()