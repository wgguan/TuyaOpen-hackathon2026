import sounddevice as sd
import numpy as np
import librosa
import collections
import pandas as pd
import scipy.ndimage
from pynput import keyboard

# --- CONFIGURATION ---
SAMPLING_RATE = 22050
CHUNK_SIZE = 8192  # Optimized for M1 stability
CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

TEMPLATES = {
    'Maj': np.array([1, 0, 0, -0.5, 1, 0, 0, 1, 0, 0, 0, 0]),
    'min': np.array([1, 0, 0, 1, -0.5, 0, 0, 1, 0, 0, 0, 0]),
}

# --- TRACKING ---
history = collections.deque(maxlen=8)
chord_timeline = []
running = True

def get_detailed_chord(y):
    local_n_fft = 2048 
    
    # Safety Guard: Pad if signal is too short for n_fft
    if len(y) < local_n_fft:
        y = np.pad(y, (0, local_n_fft - len(y)), mode='constant')

    # 1. Harmonic-Percussive Separation (HPSS)
    y_harmonic = librosa.effects.harmonic(y, n_fft=local_n_fft, margin=3.0)
    
    # 2. Chroma CQT
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=SAMPLING_RATE, n_fft=local_n_fft)
    
    # 3. Temporal Smoothing
    chroma = scipy.ndimage.median_filter(chroma, size=(1, 9))
    
    mean_chroma = np.mean(chroma, axis=1)
    if np.max(mean_chroma) > 0:
        mean_chroma = (mean_chroma / np.max(mean_chroma)) ** 2

    best_score, best_guess = -100, "..."
    for name, template in TEMPLATES.items():
        for root_idx in range(12):
            score = np.dot(mean_chroma, np.roll(template, root_idx))
            if score > best_score:
                best_score, best_guess = score, f"{CHORD_NAMES[root_idx]} {name}"

    history.append(best_guess)
    final_choice = collections.Counter(history).most_common(1)[0][0]
    chord_timeline.append(final_choice)
    return final_choice

def audio_callback(indata, frames, time, status):
    y = indata[:, 0]
    if np.max(np.abs(y)) < 0.02: return # Noise Gate
    detected = get_detailed_chord(y)
    print(f"\rCurrent Harmony: {detected}   ", end="", flush=True)

def on_press(key):
    global running
    try:
        if key.char == 'q':
            print("\n\n--- Stopping Session... ---")
            running = False
            return False 
    except AttributeError:
        pass

# --- MAIN EXECUTION ---
print("--- Jam Buddy v6 (Anaconda Final) ---")
print("Press 'q' to stop and generate the ranked report.")

listener = keyboard.Listener(on_press=on_press)
listener.start()

with sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
                    blocksize=CHUNK_SIZE, callback=audio_callback):
    while running:
        sd.sleep(100)

# --- RANKED PERCENTAGE TABLE ---
if chord_timeline:
    df = pd.DataFrame(chord_timeline, columns=['Chord'])
    # Rank by frequency
    report = df['Chord'].value_counts(normalize=True).mul(100).round(1).reset_index()
    report.columns = ['Chord Name', 'Percentage (%)']
    report['Occurrences'] = df['Chord'].value_counts().values
    
    print("\n" + "="*40)
    print("        RANKED CHORD ANALYSIS")
    print("="*40)
    print(report.to_string(index=False))
    print("="*40)
else:
    print("\nNo data to analyze.")