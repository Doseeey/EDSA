import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def denoise(ekg: np.array, fs: int, lowcut: int, highcut: int, plot_freq: bool = False) -> np.array:
    fft_values = np.fft.fft(ekg)
    fft_freq = np.fft.fftfreq(len(ekg), d=1/fs)

    pos_mask = fft_freq >= 0
    fft_freq = fft_freq[pos_mask]
    fft_values = np.abs(fft_values[pos_mask])

    if plot_freq:
        plt.figure(figsize=(10, 5))
        plt.plot(fft_freq, fft_values)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.title('EKG FFT')
        plt.xlim(0, 50)
        plt.ylim(0, 200000)
        plt.show()

    # lowcut = 10.0
    # highcut = 48.0
    b, a = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs)
    filtered_ekg = filtfilt(b, a, ekg)

    return filtered_ekg