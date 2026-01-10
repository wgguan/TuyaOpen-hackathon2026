import pyaudio
import numpy as np
import aubio

# Test parameters
BUFFER_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32
METHOD = "default"
SAMPLE_RATE = 44100
HOP_SIZE = BUFFER_SIZE // 2

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=HOP_SIZE)

# Aubio tempo detection
o = aubio.tempo(METHOD, BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)

print("--- Listening for beats... Play some music! ---")

try:
    while True:
        data = stream.read(HOP_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)
        if o(samples):
            print("BEAT DETECTED")
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()