from get_data import get_data
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks

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


arr = get_data("arm_in_move")[:, [2, 6]] #Date and value

ekg = arr[:, 1].astype(float)
timestamps = arr[:, 0]

time_sec = np.array([ts.timestamp() for ts in [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in timestamps]])
time_sec -= time_sec[0]

fs = 200 #Sampling per 0.005

filtered_ekg = denoise(ekg, fs, 10, 48, False)


min_distance = int(200 * 60 / 180)

r_peaks, _ = find_peaks(filtered_ekg, distance=min_distance, height=np.mean(filtered_ekg)+0.5*np.std(filtered_ekg))

print(f"Detected {len(r_peaks)} R-peaks")

p_peaks = []
t_peaks = []

for r in r_peaks:
    start_p = max(0, r - int(0.2 * fs))
    end_p = r - int(0.05 * fs)
    p_window = filtered_ekg[start_p:end_p]
    if len(p_window) > 0:
        p = np.argmax(p_window) + start_p
        p_peaks.append(p)

    start_t = r + int(0.05 * fs)
    end_t = min(len(filtered_ekg), r + int(0.4 * fs))
    t_window = filtered_ekg[start_t:end_t]
    if len(t_window) > 0:
        t = np.argmax(t_window) + start_t
        t_peaks.append(t)

rr_intervals = np.diff(time_sec[r_peaks])
hr_avg = 60 / np.mean(rr_intervals)

print(f"Average Heart Rate: {hr_avg:.1f} bpm")

hr_instant = 60 / rr_intervals  # bpm per RR interval


plt.figure(figsize=(12, 5))
plt.plot(time_sec, ekg, label='Original', alpha=0.5)
plt.plot(time_sec, filtered_ekg, label='Filtered', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('EKG Value')
plt.title('Original vs Filtered EKG Signal')
plt.legend()
plt.tight_layout()
plt.show()

t_end = 2  # seconds
window_mask = time_sec <= t_end

plt.figure(figsize=(12, 5))
plt.plot(time_sec[window_mask], filtered_ekg[window_mask], label='Filtered EKG')

r_peaks_window = [r for r in r_peaks if time_sec[r] <= t_end]
p_peaks_window = [p for p in p_peaks if time_sec[p] <= t_end]
t_peaks_window = [t for t in t_peaks if time_sec[t] <= t_end]

plt.plot(time_sec[r_peaks_window], filtered_ekg[r_peaks_window], 'ro', label='R-peaks')
plt.plot(time_sec[p_peaks_window], filtered_ekg[p_peaks_window], 'go', label='P-waves')
plt.plot(time_sec[t_peaks_window], filtered_ekg[t_peaks_window], 'mo', label='T-waves')

plt.xlabel('Time [s]')
plt.ylabel('EKG Value')
plt.title('Detected R, P, and T Peaks (first 5 seconds)')
plt.legend()
plt.show()
