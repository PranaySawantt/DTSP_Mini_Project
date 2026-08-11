import os
import time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import firwin, lfilter, spectrogram, freqz

# ============================================================
# DTSP MINI PROJECT
# FIR LOW-PASS FILTER FOR VOICE SIGNALS
# ============================================================

INPUT_DIR = "input"
OUTPUT_DIR = "output"
GRAPH_DIR = "graphs"

VOICE1 = os.path.join(INPUT_DIR, "voice1.wav")
VOICE2 = os.path.join(INPUT_DIR, "voice2.wav")

FILTER_ORDER = 101
CUTOFF_HZ = 4000
COMPARISON_ORDERS = [31, 61, 101, 151]

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


def load_audio(path):
    """Load WAV file and convert stereo/multichannel audio to mono."""
    data, fs = sf.read(path)
    data = np.asarray(data, dtype=np.float64)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data, fs


def signal_stats(x):
    return {
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }


def fft_data(x, fs):
    n = len(x)
    freqs = rfftfreq(n, 1 / fs)
    spectrum = np.abs(rfft(x))
    return freqs, spectrum


def save_waveform(x, fs, filename, title):
    t = np.arange(len(x)) / fs

    plt.figure(figsize=(11, 4.5))
    plt.plot(t, x, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=160)
    plt.close()


def save_fft(x, fs, filename, title):
    freqs, spectrum = fft_data(x, fs)

    plt.figure(figsize=(11, 4.5))
    plt.plot(freqs, spectrum, linewidth=0.8)
    plt.xlim(0, min(fs / 2, 12000))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, filename), dpi=160)
    plt.close()


def save_spectrogram(x, fs, filename, title):
    f, t, sxx = spectrogram(
        x,
        fs=fs,
        nperseg=1024,
        noverlap=768
    )

    plt.figure(figsize=(11, 5))

    plt.pcolormesh(
        t,
        f,
        10 * np.log10(sxx + 1e-12),
        shading="gouraud"
    )

    plt.ylim(0, min(fs / 2, 12000))

    plt.colorbar(label="Power/Frequency (dB)")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, filename),
        dpi=160
    )

    plt.close()


def apply_fir(x, fs, order, cutoff):
    """Apply an FIR low-pass filter."""

    taps = firwin(
        order,
        cutoff,
        fs=fs,
        pass_zero="lowpass"
    )

    filtered = lfilter(
        taps,
        [1.0],
        x
    )

    return filtered, taps


def high_frequency_power_percentage(x, fs, cutoff):
    """Calculate percentage of FFT power above cutoff."""

    freqs, spectrum = fft_data(x, fs)

    power = spectrum ** 2

    total = np.sum(power)

    high = np.sum(
        power[freqs > cutoff]
    )

    if total == 0:
        return 0.0

    return 100.0 * high / total


def save_filter_analysis(
    taps,
    fs,
    order,
    cutoff,
    filename
):
    w, h = freqz(
        taps,
        worN=4096,
        fs=fs
    )

    plt.figure(figsize=(11, 4.8))

    plt.plot(
        w,
        20 * np.log10(
            np.maximum(
                np.abs(h),
                1e-8
            )
        ),
        linewidth=1.0
    )

    plt.axvline(
        cutoff,
        linestyle="--",
        linewidth=1.2,
        label=f"Cutoff = {cutoff} Hz"
    )

    plt.xlim(
        0,
        min(fs / 2, 12000)
    )

    plt.ylim(-100, 5)

    plt.title(
        f"FIR Low-Pass Filter Frequency Response "
        f"(Order {order})"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(GRAPH_DIR, filename),
        dpi=160
    )

    plt.close()


def save_order_comparison(
    orders,
    times
):
    plt.figure(figsize=(9, 5))

    plt.plot(
        orders,
        times,
        marker="o",
        linewidth=1.5
    )

    plt.title(
        "FIR Filter Order vs Processing Time"
    )

    plt.xlabel("Filter Order")
    plt.ylabel("Processing Time (ms)")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "09_filter_order_comparison.png"
        ),
        dpi=160
    )

    plt.close()


def save_voice2_comparison(
    original,
    filtered,
    fs
):
    t_original = (
        np.arange(len(original)) / fs
    )

    t_filtered = (
        np.arange(len(filtered)) / fs
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        t_original,
        original,
        linewidth=0.7,
        label="Voice 2 - Original"
    )

    plt.plot(
        t_filtered,
        filtered,
        linewidth=0.7,
        label="Voice 2 - Filtered"
    )

    plt.title(
        "Voice 2: Original vs FIR Filtered Waveform"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "14_voice2_waveform_comparison.png"
        ),
        dpi=160
    )

    plt.close()


def print_signal_info(
    name,
    x,
    fs
):
    stats = signal_stats(x)

    print(f"\n{name}")

    print(
        f"Sampling Rate       : {fs} Hz"
    )

    print(
        f"Duration            : "
        f"{len(x) / fs:.2f} s"
    )

    print(
        f"Samples             : {len(x)}"
    )

    print("\nSignal Statistics")

    print(
        f"Maximum Amplitude   : "
        f"{stats['max']:.4f}"
    )

    print(
        f"Minimum Amplitude   : "
        f"{stats['min']:.4f}"
    )

    print(
        f"Mean                : "
        f"{stats['mean']:.4f}"
    )

    print(
        f"RMS                 : "
        f"{stats['rms']:.4f}"
    )


def main():

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    if not os.path.exists(VOICE1):
        raise FileNotFoundError(
            f"Missing input file: {VOICE1}"
        )

    if not os.path.exists(VOICE2):
        raise FileNotFoundError(
            f"Missing input file: {VOICE2}"
        )

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    voice1, fs1 = load_audio(
        VOICE1
    )

    voice2, fs2 = load_audio(
        VOICE2
    )

    if fs1 != fs2:
        raise ValueError(
            f"Sampling rates must match. "
            f"voice1={fs1}, voice2={fs2}"
        )

    fs = fs1

    if CUTOFF_HZ >= fs / 2:
        raise ValueError(
            "Cutoff frequency must be below "
            "the Nyquist frequency."
        )

    # ========================================================
    # PROJECT HEADER
    # ========================================================

    print("=" * 64)

    print(
        "DTSP MINI PROJECT - "
        "FIR VOICE SIGNAL FILTERING"
    )

    print("=" * 64)

    # ========================================================
    # VOICE 1 INFORMATION
    # ========================================================

    print_signal_info(
        "Input File          : input/voice1.wav",
        voice1,
        fs
    )

    # ========================================================
    # MAIN FIR FILTER
    # ========================================================

    print("\n## Main FIR Filter")

    print(
        f"\nFilter Type         : FIR Low Pass"
    )

    print(
        f"Filter Order        : "
        f"{FILTER_ORDER}"
    )

    print(
        f"Cutoff Frequency    : "
        f"{CUTOFF_HZ} Hz"
    )

    start = time.perf_counter()

    filtered1, taps = apply_fir(
        voice1,
        fs,
        FILTER_ORDER,
        CUTOFF_HZ
    )

    processing_time = (
        time.perf_counter() - start
    ) * 1000

    sf.write(
        os.path.join(
            OUTPUT_DIR,
            "filtered.wav"
        ),
        filtered1,
        fs
    )

    print(
        f"Processing Time     : "
        f"{processing_time:.3f} ms"
    )

    print(
        "Output File         : "
        "output/filtered.wav"
    )

    # ========================================================
    # VOICE 1 GRAPHS
    # ========================================================

    save_waveform(
        voice1,
        fs,
        "01_original_waveform.png",
        "Original Voice 1 Waveform"
    )

    save_waveform(
        filtered1,
        fs,
        "02_filtered_waveform.png",
        "FIR Filtered Voice 1 Waveform"
    )

    save_fft(
        voice1,
        fs,
        "03_original_fft.png",
        "Original Voice 1 FFT"
    )

    save_fft(
        filtered1,
        fs,
        "04_filtered_fft.png",
        "Filtered Voice 1 FFT"
    )

    save_spectrogram(
        voice1,
        fs,
        "05_original_spectrogram.png",
        "Original Voice 1 Spectrogram"
    )

    save_spectrogram(
        filtered1,
        fs,
        "06_filtered_spectrogram.png",
        "Filtered Voice 1 Spectrogram"
    )

    # ========================================================
    # SIGNAL ANALYSIS GRAPH
    # ========================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7)
    )

    t1 = (
        np.arange(len(voice1)) / fs
    )

    tf1 = (
        np.arange(len(filtered1)) / fs
    )

    axes[0].plot(
        t1,
        voice1,
        linewidth=0.7
    )

    axes[0].set_title(
        "Voice 1 - Original Signal"
    )

    axes[0].set_xlabel(
        "Time (s)"
    )

    axes[0].set_ylabel(
        "Amplitude"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    axes[1].plot(
        tf1,
        filtered1,
        linewidth=0.7
    )

    axes[1].set_title(
        "Voice 1 - Filtered Signal"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "07_signal_analysis.png"
        ),
        dpi=160
    )

    plt.close(fig)

    # ========================================================
    # FILTER ANALYSIS
    # ========================================================

    save_filter_analysis(
        taps,
        fs,
        FILTER_ORDER,
        CUTOFF_HZ,
        "08_filter_analysis.png"
    )

    # ========================================================
    # FILTER ORDER COMPARISON
    # ========================================================

    print(
        "\n## FIR Filter Order Comparison"
    )

    order_times = []

    for order in COMPARISON_ORDERS:

        start = time.perf_counter()

        _filtered, _taps = apply_fir(
            voice1,
            fs,
            order,
            CUTOFF_HZ
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        order_times.append(
            elapsed
        )

        print(
            f"Order {order:3d}          : "
            f"{elapsed:.3f} ms"
        )

    save_order_comparison(
        COMPARISON_ORDERS,
        order_times
    )

    # ========================================================
    # VOICE 2
    # SECOND / NOISIER TEST CASE
    # ========================================================

    print(
        "\n## Voice 2 - Secondary Test Case"
    )

    print(
        "Input File          : "
        "input/voice2.wav"
    )

    stats2 = signal_stats(
        voice2
    )

    high2_before = (
        high_frequency_power_percentage(
            voice2,
            fs,
            CUTOFF_HZ
        )
    )

    # --------------------------------------------------------
    # FILTER VOICE 2
    # --------------------------------------------------------

    start = time.perf_counter()

    filtered2, _taps2 = apply_fir(
        voice2,
        fs,
        FILTER_ORDER,
        CUTOFF_HZ
    )

    voice2_processing_time = (
        time.perf_counter() - start
    ) * 1000

    high2_after = (
        high_frequency_power_percentage(
            filtered2,
            fs,
            CUTOFF_HZ
        )
    )

    sf.write(
        os.path.join(
            OUTPUT_DIR,
            "filtered_voice2.wav"
        ),
        filtered2,
        fs
    )

    print(
        f"Sampling Rate       : "
        f"{fs} Hz"
    )

    print(
        f"Duration            : "
        f"{len(voice2) / fs:.2f} s"
    )

    print(
        f"Samples             : "
        f"{len(voice2)}"
    )

    print(
        f"Maximum Amplitude   : "
        f"{stats2['max']:.4f}"
    )

    print(
        f"Minimum Amplitude   : "
        f"{stats2['min']:.4f}"
    )

    print(
        f"Mean                : "
        f"{stats2['mean']:.4f}"
    )

    print(
        f"RMS                 : "
        f"{stats2['rms']:.4f}"
    )

    print(
        f"Power Above 4 kHz   : "
        f"{high2_before:.3f}%"
    )

    print(
        f"After Filtering     : "
        f"{high2_after:.3f}%"
    )

    print(
        f"Processing Time     : "
        f"{voice2_processing_time:.3f} ms"
    )

    print(
        "Output File         : "
        "output/filtered_voice2.wav"
    )

    # ========================================================
    # VOICE 2 GRAPHS
    # ========================================================

    save_waveform(
        voice2,
        fs,
        "10_voice2_original_waveform.png",
        "Voice 2 - Original / Noisier Test Signal"
    )

    save_waveform(
        filtered2,
        fs,
        "11_voice2_filtered_waveform.png",
        "Voice 2 - FIR Filtered Signal"
    )

    save_fft(
        voice2,
        fs,
        "12_voice2_original_fft.png",
        "Voice 2 - Original FFT"
    )

    save_fft(
        filtered2,
        fs,
        "13_voice2_filtered_fft.png",
        "Voice 2 - Filtered FFT"
    )

    save_voice2_comparison(
        voice2,
        filtered2,
        fs
    )

    # ========================================================
    # FILTER COEFFICIENTS
    # ========================================================

    coefficients_path = os.path.join(
        OUTPUT_DIR,
        "filter_coefficients.txt"
    )

    with open(
        coefficients_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DTSP MINI PROJECT - "
            "FIR LOW-PASS FILTER\n"
        )

        f.write("=" * 55 + "\n")

        f.write(
            "Filter Type: FIR Low Pass\n"
        )

        f.write(
            f"Filter Order: {FILTER_ORDER}\n"
        )

        f.write(
            f"Cutoff Frequency: "
            f"{CUTOFF_HZ} Hz\n"
        )

        f.write(
            f"Sampling Rate: "
            f"{fs} Hz\n\n"
        )

        f.write(
            "Filter Coefficients:\n"
        )

        for i, coefficient in enumerate(
            taps
        ):

            f.write(
                f"h[{i:03d}] = "
                f"{coefficient:.12f}\n"
            )

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    info_path = os.path.join(
        OUTPUT_DIR,
        "project_info.txt"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DTSP MINI PROJECT\n"
        )

        f.write("=" * 55 + "\n\n")

        f.write(
            "Topic: FIR Low-Pass Filtering "
            "of Voice Signals\n\n"
        )

        f.write(
            "Main Input: input/voice1.wav\n"
        )

        f.write(
            "Secondary Test Input: "
            "input/voice2.wav\n\n"
        )

        f.write(
            "Main Filter:\n"
        )

        f.write(
            f"  Type: FIR Low Pass\n"
        )

        f.write(
            f"  Order: {FILTER_ORDER}\n"
        )

        f.write(
            f"  Cutoff: {CUTOFF_HZ} Hz\n"
        )

        f.write(
            f"  Sampling Rate: {fs} Hz\n\n"
        )

        f.write(
            "Voice 2 Analysis:\n"
        )

        f.write(
            "  Used as a second/noisier "
            "test recording.\n"
        )

        f.write(
            f"  Power above 4 kHz before "
            f"filtering: {high2_before:.3f}%\n"
        )

        f.write(
            f"  Power above 4 kHz after "
            f"filtering: {high2_after:.3f}%\n\n"
        )

        f.write(
            "Filter Order Comparison:\n"
        )

        for order, elapsed in zip(
            COMPARISON_ORDERS,
            order_times
        ):

            f.write(
                f"  Order {order}: "
                f"{elapsed:.3f} ms\n"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n## Generated Graphs"
    )

    generated_graphs = [

        "01_original_waveform.png",

        "02_filtered_waveform.png",

        "03_original_fft.png",

        "04_filtered_fft.png",

        "05_original_spectrogram.png",

        "06_filtered_spectrogram.png",

        "07_signal_analysis.png",

        "08_filter_analysis.png",

        "09_filter_order_comparison.png",

        "10_voice2_original_waveform.png",

        "11_voice2_filtered_waveform.png",

        "12_voice2_original_fft.png",

        "13_voice2_filtered_fft.png",

        "14_voice2_waveform_comparison.png",
    ]

    for filename in generated_graphs:

        print(
            os.path.join(
                GRAPH_DIR,
                filename
            )
        )

    print(
        "\n## Generated Output Files"
    )

    print(
        "output/filtered.wav"
    )

    print(
        "output/filtered_voice2.wav"
    )

    print(
        "output/filter_coefficients.txt"
    )

    print(
        "output/project_info.txt"
    )

    print(
        f"\nMain Filter Processing Time : "
        f"{processing_time:.3f} ms"
    )

    print(
        f"Voice 2 Processing Time     : "
        f"{voice2_processing_time:.3f} ms"
    )

    print(
        "\n" + "=" * 64
    )

    print(
        "PROJECT COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()