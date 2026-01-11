import sounddevice as sd
import numpy as np
import librosa
import collections
import matplotlib.pyplot as plt
from collections import Counter

# --- CONFIGURATION ---
SAMPLING_RATE = 22050
# Larger blocksize for more stable analysis on Mac M1
CHUNK_SIZE = 8192 
CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

TEMPLATES = {
    'Maj': np.array([1, 0, 0, -0.5, 1, 0, 0, 1, 0, 0, 0, 0]),
    'min': np.array([1, 0, 0, 1, -0.5, 0, 0, 1, 0, 0, 0, 0]),
}

# --- TRACKING ---
history = collections.deque(maxlen=8)
chord_timeline = []

def get_detailed_chord(y):
    local_n_fft = 2048 
    
    # [FIX] Safety Check: If signal is shorter than n_fft, pad it with zeros
    # This prevents the "n_fft too large" warning
    if len(y) < local_n_fft:
        y = np.pad(y, (0, local_n_fft - len(y)), mode='constant')

    # [FIX] Noise Gate: If the signal is too weak, don't try to tune it
    # This prevents the "empty frequency set" warning
    if np.max(np.abs(y)) < 0.02: 
        return "..."

    # 1. Harmonic Filter (Cleans up vocals/drums)
    y_harmonic = librosa.effects.harmonic(y, n_fft=local_n_fft, margin=3.0)
    
    # 2. Chroma STFT with explicit window
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=SAMPLING_RATE, n_fft=local_n_fft)

    # 3. Normalization and Squaring
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
    if status: print(status)
    y = indata[:, 0]
    if np.max(np.abs(y)) < 0.02: return
    detected = get_detailed_chord(y)
    print(f"Current Harmony: {detected}")

# --- EXECUTION ---
print("--- Jam Buddy v4 Live ---")
print("Stream starting... Ready for your music!")

# Reset data before starting
chord_timeline.clear() 

# Start stream
stream = sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
                        blocksize=CHUNK_SIZE, callback=audio_callback)

with stream:
    print("\n[ACTIVE] Play now! Press CTRL+C only when you are finished.")
    try:
        # Keep program alive while recording
        while True:
            sd.sleep(500) 
    except KeyboardInterrupt:
        print("\n--- Session Ended. Generating Report... ---")

# --- POST-SESSION REPORTING ---
if chord_timeline:
    # 1. CLEAN PROGRESSION
    filtered_progression = []
    current, count = chord_timeline[0], 0
    for c in chord_timeline:
        if c == current: count += 1
        else:
            if count >= 3: filtered_progression.append(current)
            current, count = c, 1
    filtered_progression.append(current)

    print("\nCHORD PROGRESSION (Filtered):")
    print(" -> ".join(filtered_progression))

    # 2. HISTOGRAM
    counts = Counter(chord_timeline)
    labels, values = zip(*counts.items())
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='skyblue')
    plt.title('Harmonic Distribution (%)')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("No chords were detected during the session.")