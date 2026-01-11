import sounddevice as sd
import numpy as np
import librosa
import collections

import scipy.ndimage # Required for temporal smoothing

# --- CONFIGURATION UPDATES ---
SAMPLING_RATE = 22050 
# Increase CHUNK_SIZE to ensure it's larger than the default n_fft (2048 or 4096)
CHUNK_SIZE = 8192  # Approximately 370ms of audio, which is safer for CQT
CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Templates for Chord Recognition
TEMPLATES = {
    'Maj':      [1, 0, 0, -0.5, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':      [1, 0, 0, 1, -0.5, 0, 0, 1, 0, 0, 0, 0],
    # 'HarMin':   [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], # Harmonic Minor
    # 'NatMin':   [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], # Natural Minor
}

# Buffer to smooth out results over time
history = collections.deque(maxlen=8)



def get_detailed_chord(y):
    # 1. Harmonic-Percussive Source Separation (HPSS)
    # We extract only the 'harmonic' part to ignore vocals/drums
    y_harmonic = librosa.effects.harmonic(y, margin=3.0) 
    
    # 2. Compute Chroma CQT (Better than STFT for music)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=SAMPLING_RATE)
    
    # 3. Temporal Smoothing (Median Filter)
    # This stabilizes the result by looking at neighboring bins
    chroma = scipy.ndimage.median_filter(chroma, size=(1, 9))
    
    # 4. Normalization and Squaring (your existing logic)
    mean_chroma = np.mean(chroma, axis=1)
    if np.max(mean_chroma) > 0:
        mean_chroma = (mean_chroma / np.max(mean_chroma)) ** 2

    best_score = -100 
    best_guess = "..."

    for name, template in TEMPLATES.items():
        template_np = np.array(template)
        for root_idx in range(12):
            score = np.dot(mean_chroma, np.roll(template_np, root_idx))
            if score > best_score:
                best_score = score
                best_guess = f"{CHORD_NAMES[root_idx]} {name}"

    history.append(best_guess)
    return collections.Counter(history).most_common(1)[0][0]

def audio_callback(indata, frames, time, status):
    y = indata[:, 0]
    # Simple noise gate: ignore very quiet signals
    if np.max(np.abs(y)) < 0.02: 
        return

    detected = get_detailed_chord(y)
    print(f"Current Harmony: {detected}")

# --- EXECUTION ---

import matplotlib.pyplot as plt
from collections import Counter

# --- NEW: Storage for Analysis ---
chord_timeline = []

def audio_callback(indata, frames, time, status):
    y = indata[:, 0]
    if np.max(np.abs(y)) < 0.02: return
    
    detected = get_detailed_chord(y)
    chord_timeline.append(detected) # Log for final analysis
    print(f"Current Harmony: {detected}")

# --- EXECUTION BLOCK ---
print("--- Jam Buddy Live! ---")
print("PLAY MUSIC NOW. Press CTRL+C to stop and see your report.")

try:
    with sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
                        blocksize=CHUNK_SIZE, callback=audio_callback):
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\n--- STOPPING & ANALYZING SESSION ---")

    if not chord_timeline:
        print("No audio detected.")
    else:
        # 1. GENERATE PROGRESSION (Filtered for temporal order)
        filtered_progression = []
        if len(chord_timeline) > 0:
            current_chord = chord_timeline[0]
            count = 0
            for c in chord_timeline:
                if c == current_chord:
                    count += 1
                else:
                    if count >= 3: # Outlier Filter: Must last 3 frames
                        filtered_progression.append(current_chord)
                    current_chord = c
                    count = 1
            filtered_progression.append(current_chord) # Add final chord

        # 2. CALCULATE PERCENTAGES
        total = len(chord_timeline)
        counts = Counter(chord_timeline)
        labels = list(counts.keys())
        values = [(counts[l] / total) * 100 for l in labels]

        # 3. DISPLAY RESULTS
        print("\nCHORD PROGRESSION (Cleaned):")
        print(" -> ".join(filtered_progression))

        # 4. SHOW HISTOGRAM
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color='skyblue')
        plt.ylabel('Percentage of Session (%)')
        plt.title('Harmonic Distribution of your Jam Session')
        plt.show()

# print("--- Jam Buddy Backend Live ---")
# print("Tracking every ~0.5s beat ~120 bpm.")
# with sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
#                     blocksize=CHUNK_SIZE, callback=audio_callback):
#    while True:
#         sd.sleep(1000)