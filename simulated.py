import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal
import time
from scipy.io import wavfile
from matplotlib.widgets import TextBox

amp_factor = 2
sample_rate = 5800
chunk = 1024
buffer_size = 4096
window_size = int(0.04 * sample_rate)
overlap = int(0.02 * sample_rate)
output_file = 'test.wav'

voltage_buffer = np.zeros(buffer_size)
time_buffer = np.linspace(0, buffer_size/sample_rate, buffer_size)
sound_time = np.linspace(0, chunk/sample_rate, chunk)
start_time = time.time()

audio_buffer = []
recording = True

freq_1= 500
freq_2= 1000

lowcut = 100
highcut = 2800

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_sig = signal.lfilter(b, a, data)
    return filtered_sig


def generate_voltage_wave():
    t = np.linspace(0, chunk/sample_rate, chunk)
    wave_1 = np.sin(2 * np.pi * freq_1 * t)
    wave_2 = np.sin(2 * np.pi * freq_2 * t)
    humming = np.sin(2 * np.pi * 60 * t)
    full = wave_1+wave_2+humming
    signal = butter_bandpass_filter(full, lowcut, highcut, sample_rate, order=5)
    return 2.5 + 0.1 * (signal)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), height_ratios=[1, 1.5, 1])
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, hspace=0.3)

line1, = ax1.plot(time_buffer, voltage_buffer, 'b-')
ax1.set_ylim(0, 5)
ax1.set_ylabel('Voltage')
ax1.set_xlabel('Time')
ax1.set_title('Moving Voltage')

# Initial spectrogram data
frequencies, times, Sxx = signal.spectrogram(voltage_buffer, sample_rate, nperseg=window_size, noverlap=overlap)
spec = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
ax2.set_ylim(20, sample_rate/2)
ax2.set_ylabel('Freq')
ax2.set_xlabel('Time')
plt.colorbar(spec, ax=ax2)

line2, = ax3.plot(sound_time, np.zeros(chunk), 'g-')
ax3.set_ylim(0 , 5 * amp_factor)
ax3.set_title('Sound wave(Amplified)')

# Stuff for changing the frequencies
ax_input_wave_1 = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_input_wave_2 = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')

box_freq1 = TextBox(ax_input_wave_1, 'Freq 1', initial=str(freq_1))
box_freq2 = TextBox(ax_input_wave_2, 'Freq 2', initial=str(freq_2))

def update_freq(val):
    global freq_1, freq_2
    try:
        freq_1 = float(box_freq1.text)
        freq_2 = float(box_freq2.text)
    except ValueError:
        box_freq1.set_val(str(freq_1))
        box_freq2.set_val(str(freq_1))

box_freq1.on_submit(update_freq)
box_freq2.on_submit(update_freq)

def update(frame):
    global start_time, voltage_buffer, time_buffer
    current_time = time.time() - start_time
    
    # Simulating incoming data by using voltage_wave and updating voltage_buffer
    new_voltage = generate_voltage_wave()  # Use the generated wave to simulate incoming voltage signal

    voltage_buffer[:-chunk] = voltage_buffer[chunk:]
    voltage_buffer[-chunk:] = new_voltage

    # Update the time buffer
    time_buffer = np.linspace(current_time - buffer_size/sample_rate, current_time, buffer_size)

    # Update the plot with the new voltage data
    line1.set_xdata(time_buffer)
    line1.set_ydata(voltage_buffer)
    ax1.set_xlim(time_buffer[0], time_buffer[-1])

    # Update the spectrogram
    frequencies, times, Sxx = signal.spectrogram(voltage_buffer, sample_rate, nperseg=window_size, noverlap=overlap)
    spec.set_array(10 * np.log10(Sxx + 1e-10).ravel())
    spec.set_clim(vmin=-80, vmax=0)
    ax2.set_xlim(times[0], times[-1])

    # Update the sound wave plot (displaying the current wave)
    amped_voltage = new_voltage * amp_factor # Add the amp factor only to the last graph
    line2.set_ydata(amped_voltage)
    ax3.set_xlim(sound_time[0], sound_time[-1])

    if recording:
        audio_buffer.extend(amped_voltage.tolist())

    return line1, spec, line2

def on_close(event):
    global recording, audio_buffer
    if recording:
        recording = False
        print("Closed")
        audio_array = np.array(audio_buffer)
        audio_noraml = (audio_array - 2.5)
        audio_int16 = np.int16(audio_noraml * 32767)
        wavfile.write(output_file, sample_rate, audio_int16)
        print("saved")

fig.canvas.mpl_connect('close_event', on_close) 

ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=False, cache_frame_data=False)

plt.show()
