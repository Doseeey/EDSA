import numpy as np
from scipy.signal import find_peaks

def rpt(filtered_ekg: np.array, fs: int):
    min_distance = int(fs * 60 / 180) # najmniejszy dystans miedzy uderzeniami serca
    min_height = np.mean(filtered_ekg)+0.7*np.std(filtered_ekg) # szczyt kiedy odchylony jest o 0.5 odchylenia standardowego powyzej sredniej

    r_index, _ = find_peaks(filtered_ekg, distance=min_distance, height=min_height, prominence=0.15)

    # print(r_index) # indeksy szczytow R

    p_index = []
    t_index = []

    for r in r_index:
        # Załamek P - do 0.12 [s] przed szczytem R
        start_p = max(0, r - int(0.12 * fs))
        end_p = r - int(0.05 * fs)
        p_window = filtered_ekg[start_p:end_p]
        if len(p_window) > 0:
            # indeks maks wartosci w zasiegu
            p = np.argmax(p_window) + start_p
            p_index.append(p)

        # Załamek R - 0.08 [s] po szczycie, trwa 0.13-0.16 [s]
        start_t = r + int(0.05 * fs)
        end_t = min(len(filtered_ekg), r + int(0.24 * fs))
        t_window = filtered_ekg[start_t:end_t]
        if len(t_window) > 0:
            t = np.argmax(t_window) + start_t
            t_index.append(t)

    return r_index, p_index, t_index