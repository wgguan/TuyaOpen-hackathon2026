import sounddevice as sd
import numpy as np
import librosa

# --- Settings ---
SAMPLING_RATE = 22050  # Standard for librosa
CHUNK_SIZE = 4096      # Size of audio "window" to analyze at once
CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Basic Major Chord Templates (1 = note is in chord, 0 = not)
# Example: C Major is C (1st), E (5th), and G (8th)
MAJOR_TEMPLATE = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

def get_chord_from_chroma(chroma):
    # Flatten the chroma features to a 12-element vector
    mean_chroma = np.mean(chroma, axis=1)
    
    # Check similarity against the Major template for all 12 keys
    # We "roll" the template to check C, then C#, then D...
    scores = [np.dot(np.roll(MAJOR_TEMPLATE, i), mean_chroma) for i in range(12)]
    
    return CHORD_NAMES[np.argmax(scores)]

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    # Ensure we have a clean signal
    y = indata[:, 0]
    if np.max(np.abs(y)) < 0.02:  # Silence threshold
        return

    # Compute Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLING_RATE, n_fft=CHUNK_SIZE)
    
    # Identify the chord
    chord = get_chord_from_chroma(chroma)
    print(f"Detected Chord: {chord}")

# --- Start Listening ---
print("--- Listening! Play a Major chord on a guitar or piano near the mic ---")
with sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
                    blocksize=CHUNK_SIZE, callback=audio_callback):
    while True:
        sd.sleep(1000)