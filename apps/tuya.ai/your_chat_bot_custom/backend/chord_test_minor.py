import sounddevice as sd
import numpy as np
import librosa
import collections

# --- CONFIGURATION ---
SAMPLING_RATE = 22050  # Standard for musical analysis
CHUNK_SIZE = 4096      # ~185ms window
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
    # 1. Compute Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLING_RATE, n_fft=CHUNK_SIZE)
    mean_chroma = np.mean(chroma, axis=1)
    # ... previous code to get mean_chroma ...
    
    # [FIX 1] Enhancement: Square the values to push down the noise
    mean_chroma = mean_chroma ** 2
    
    # [FIX 2] Better Normalization
    if np.max(mean_chroma) > 0:
        mean_chroma = mean_chroma / np.max(mean_chroma)

    best_score = -100 # Lower starting point for negative weights
    best_guess = "..."

    for name, template in TEMPLATES.items():
        template_np = np.array(template)
        for root_idx in range(12):
            shifted_template = np.roll(template_np, root_idx)
            # Dot product now includes the negative penalties
            score = np.dot(mean_chroma, shifted_template)
            
            if score > best_score:
                best_score = score
                best_guess = f"{CHORD_NAMES[root_idx]} {name}"

    # 4. Smoothing: Return the most common guess in the last 8 frames
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
print("--- Jam Buddy Backend Live ---")
print("Tracking every ~0.5s beat ~120 bpm.")
with sd.InputStream(channels=1, samplerate=SAMPLING_RATE, 
                    blocksize=CHUNK_SIZE, callback=audio_callback):
    while True:
        sd.sleep(1000)