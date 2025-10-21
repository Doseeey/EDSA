import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def denoise(ekg: np.array, time_sec: np.array, fs: int, lowcut: int, highcut: int, derivative_filter: bool = False, plot_freq: bool = False) -> np.array:
    fft_values = np.fft.fft(ekg)
    fft_freq = np.fft.fftfreq(len(ekg), d=1/fs)

    pos_mask = fft_freq >= 0
    fft_freq = fft_freq[pos_mask]
    fft_values = np.abs(fft_values[pos_mask])

    if plot_freq:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, fft_values)
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda')
        plt.title('EKG FFT')
        # plt.xlim(0, 50)
        plt.ylim(0, 200000)
        plt.show()

    b, a = butter(N=3, Wn=[lowcut, highcut], btype='band', fs=fs)
    filtered_ekg = filtfilt(b, a, ekg)

    if plot_freq:
        # Wykres filtracji sygnalu
        plt.figure(figsize=(12, 5))
        plt.plot(time_sec, ekg, label='Sygnał oryginalny', alpha=0.5)
        plt.plot(time_sec, filtered_ekg, label='Sygnał po filtracji', linewidth=2)
        plt.xlabel('Czas [s]')
        plt.ylabel('Sygnał')
        plt.title('Sygnał przed i po filtracji zakłóceń')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if derivative_filter:
        # pan tompkins 
        h = np.array([1, 2, 0, -2, -1])
        filtered_ekg = filtfilt(h, 1, filtered_ekg)

    # normalize
    filtered_ekg = filtered_ekg/ np.max(np.abs(filtered_ekg))

    return filtered_ekg