import torch
import demucs.separate
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
import collections
import time

# --- CONFIGURATION ---
SAMPLING_RATE = 44100
RECORD_SECONDS = 8     # 8s is a sweet spot for Demucs accuracy vs speed
CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Weighted Templates for Major vs Minor
TEMPLATES = {
    'Major': np.array([1.0, 0, 0, -0.5, 1.0, 0, 0, 1.0, 0, 0, 0, 0]),
    'Minor': np.array([1.0, 0, 0, 1.0, -0.5, 0, 0, 1.0, 0, 0, 0, 0])
}

def identify_chord(y, sr):
    # Use CQT for better alignment with musical semitones
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    
    # Normalization and "Squaring" to enhance dominant notes
    if np.max(mean_chroma) > 0:
        mean_chroma = (mean_chroma / np.max(mean_chroma)) ** 2
    
    best_score = -99
    best_chord = "Unknown"
    
    for label, template in TEMPLATES.items():
        for i in range(12):
            score = np.dot(mean_chroma, np.roll(template, i))
            if score > best_score:
                best_score = score
                best_chord = f"{CHORD_NAMES[i]} {label}"
    return best_chord

def run_test():
    print(f"\n[1/4] RECORDING: Speak or play music with lyrics for {RECORD_SECONDS}s...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
    sd.wait()
    sf.write("test_input.wav", audio, SAMPLING_RATE)
    
    print("[2/4] AI SEPARATION: Demucs is removing vocals... (This may take 5-10s)")
    device = "mps" if torch.backends.mps.is_available() else "cpu" # Use Mac GPU
    
    # Run Demucs locally
    demucs.separate.main(["--two-stems", "vocals", "-d", device, "test_input.wav"])
    
    # Path logic for local files
    acc_path = "separated/htdemucs/test_input/no_vocals.wav"
    
    print("[3/4] ANALYSIS: Extracting chords from instrumental stem...")
    y_acc, sr_acc = librosa.load(acc_path, sr=22050)
    chord = identify_chord(y_acc, sr_acc)
    
    print("-" * 40)
    print(f"FINAL RESULT: {chord}")
    print("-" * 40)
    
    # Verification Step: Play the 'clean' audio so you can check the AI's work
    print("[4/4] VERIFICATION: Playing the AI-separated (No Vocals) track...")
    data, fs = sf.read(acc_path)
    sd.play(data, fs)
    sd.wait()

if __name__ == "__main__":
    try:
        while True:
            run_test()
            input("\nPress Enter to run another test or Ctrl+C to stop...")
    except KeyboardInterrupt:
        print("\nTest finished.")