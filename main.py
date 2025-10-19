from get_data import get_data
from denoise import denoise
from rpt import rpt
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

######## Filter bandwith ########
# arm_in_move
# low: 10 high: 48
# arm_no_move
# low:    high:
# chest_in_move
# low:    high:
# chest_no_move
# low:    high:
################################

ekg_data = get_data("arm_in_move")[:, [2, 6]] 

ekg = ekg_data[:, 1].astype(float)
timestamps = ekg_data[:, 0]
time_sec = np.array([ts.timestamp() for ts in [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in timestamps]])
time_sec -= time_sec[0] # Convert datetime to seconds

fs = 200 #Sampling per 0.005

filtered_ekg = denoise(ekg, fs, 10, 48, False)

r, p, t = rpt(filtered_ekg, fs)

rr_intervals = np.diff(time_sec[r])
hr_avg = 60 / np.mean(rr_intervals)

print(f"Average Heart Rate: {hr_avg:.1f} bpm")

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


#Fragment wykresu szczytow R i zalamkow P i T
t_end = 2  # [s]

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
