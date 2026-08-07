import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Read Audio
audio, sample_rate = sf.read("input/voice.wav")

# Audio Information
duration = len(audio) / sample_rate

print("=" * 45)
print("DTSP MINI PROJECT")
print("FIR-Based Audio Noise Reduction System")
print("=" * 45)

print(f"Sampling Rate : {sample_rate} Hz")
print(f"Total Samples : {len(audio)}")
print(f"Duration      : {duration:.2f} seconds")

# Time Axis
time = np.linspace(0, duration, len(audio))

# Plot Waveform
plt.figure(figsize=(12,4))
plt.plot(time, audio)

plt.title("Original Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid(True)

# Save graph
plt.savefig("graphs/original_waveform.png", dpi=300)

# Display graph
plt.show()  
