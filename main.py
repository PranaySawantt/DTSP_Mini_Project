import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import firwin, lfilter


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

# ==============================
# Frequency Spectrum (FFT)
# ==============================

N = len(audio)

fft_result = fft(audio)

frequency = fftfreq(N, 1 / sample_rate)

plt.figure(figsize=(14,5), facecolor="white")

plt.plot(frequency[:N//2], np.abs(fft_result[:N//2]))

plt.title("Original Audio Frequency Spectrum")

plt.xlabel("Frequency (Hz)")

plt.ylabel("Magnitude")

plt.grid(True, linestyle="--", alpha=0.6)

plt.savefig("graphs/original_fft.png", dpi=300)

plt.show()

# ==============================
# FIR Filter Design
# ==============================

cutoff_frequency = 4000      # Hz
filter_order = 101

fir_coefficients = firwin(
    filter_order,
    cutoff_frequency,
    fs=sample_rate
)

print("\nFIR Filter Information")
print("---------------------------")
print(f"Filter Type      : Low Pass")
print(f"Filter Order     : {filter_order}")
print(f"Cutoff Frequency : {cutoff_frequency} Hz")

# Apply Filter
filtered_audio = lfilter(fir_coefficients, 1.0, audio)

print("Filtering Completed Successfully!")