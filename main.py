from get_data import get_data
from denoise import denoise
from rpt import rpt
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

ekg_data = get_data("arm_in_move")[:, [2, 6]] 

ekg = ekg_data[:, 1].astype(float)
timestamps = ekg_data[:, 0]
time_sec = np.array([ts.timestamp() for ts in [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in timestamps]])
time_sec -= time_sec[0] # Zmiana daty z kolumny trzeciej na sekundy

fs = 200 #Sampling per 0.005

# Usuniecie zaklocen
filtered_ekg = denoise(ekg, time_sec, fs, 5, 15, True, True)

# Szczyty R, Zalamania P i T
r, p, t = rpt(filtered_ekg, fs)

# Tetno
rr_intervals = np.diff(time_sec[r])
hr_avg = 60 / np.mean(rr_intervals)

print(f"Tetno: {hr_avg:.1f} bpm")


#Fragment wykresu szczytow R i zalamkow P i T
t_end = 5  # [s]

r_range = [r for r in r if time_sec[r] <= t_end]
p_range = [p for p in p if time_sec[p] <= t_end]
t_range = [t for t in t if time_sec[t] <= t_end]

plt.figure(figsize=(12, 5))
plt.plot(time_sec[time_sec <= t_end], filtered_ekg[time_sec <= t_end], label='EKG')
plt.plot(time_sec[r_range], filtered_ekg[r_range], 'ro', label='Szczyt R')
plt.plot(time_sec[p_range], filtered_ekg[p_range], 'go', label='Załamek P')
plt.plot(time_sec[t_range], filtered_ekg[t_range], 'bo', label='Załamek T')
plt.xlabel('Czas [s]')
plt.ylabel('EKG')
plt.title('Szczyty R i załamki P i T (2 sekundy sygnału)')
plt.legend()
plt.tight_layout()
plt.show()
