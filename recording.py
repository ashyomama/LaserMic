import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from scipy import signal
import serial
import time

# Serial port settings
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAMPLE_RATE = 5800
CHUNK = 1024
BUFFER_SIZE = 4096
WINDOW_SIZE = int(0.04 * SAMPLE_RATE)
OVERLAP = int(0.02 * SAMPLE_RATE)
OUTPUT_FILE = 'glass.wav'
AMP_FACTOR = 2  # For the amplified sound wave plot

# Initialize serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
time.sleep(1)
ser.flushInput()

# Buffers
voltage_buffer = np.zeros(BUFFER_SIZE)
time_buffer = np.linspace(0, BUFFER_SIZE/SAMPLE_RATE, BUFFER_SIZE)
sound_time = np.linspace(0, CHUNK/SAMPLE_RATE, CHUNK)
start_time = time.time()

audio_buffer = []
recording = True


# Plot setup
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), height_ratios=[1, 1.5, 1])
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.95, hspace=0.3)

line1, = ax1.plot(time_buffer, voltage_buffer, 'b-')
ax1.set_ylim(0, 5)
ax1.set_ylabel('Voltage')
ax1.set_xlabel('Time')
ax1.set_title('Moving Voltage')

frequencies, times, Sxx = signal.spectrogram(voltage_buffer, SAMPLE_RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP)
spec = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
ax2.set_ylim(20, SAMPLE_RATE/2)
ax2.set_ylabel('Freq')
ax2.set_xlabel('Time')
ax2.set_title('Frequency Content from Photodiode')
plt.colorbar(spec, ax=ax2, label='Power [dB]')

line2, = ax3.plot(sound_time, np.zeros(CHUNK), 'g-')
ax3.set_ylim(0, 5 * AMP_FACTOR)
ax3.set_title('Sound Wave (Amplified)')

def update(frame):
    global start_time, voltage_buffer, time_buffer
    global audio_buffer, recording
    
    # Read serial data
    data = ser.read(CHUNK * 2)
    if len(data) < CHUNK * 2:
        return line1, spec, line2
    
    raw_samples = np.frombuffer(data, dtype=np.uint16)
    new_voltage = (raw_samples) * 5.0 / 1023.0
    
    # Apply gain for recording (no clipping)
    voltage_for_recording = new_voltage
    
    # Apply gain for plotting (with clipping)
    voltage_for_plotting = new_voltage
    voltage_for_plotting = np.clip(voltage_for_plotting, 0, 5)
    
    # Update buffers
    voltage_buffer[:-CHUNK] = voltage_buffer[CHUNK:]
    voltage_buffer[-CHUNK:] = voltage_for_plotting
    
    current_time = time.time() - start_time
    time_buffer = np.linspace(current_time - BUFFER_SIZE/SAMPLE_RATE, current_time, BUFFER_SIZE)
    
    # Update plots
    line1.set_xdata(time_buffer)
    line1.set_ydata(voltage_buffer)
    ax1.set_xlim(time_buffer[0], time_buffer[-1])
    
    frequencies, times, Sxx = signal.spectrogram(voltage_buffer - np.mean(voltage_buffer), SAMPLE_RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP)
    spec.set_array(10 * np.log10(Sxx + 1e-10).ravel())
    spec.set_clim(vmin=-80, vmax=0)
    ax2.set_xlim(times[0], times[-1])
    
    # Amplified sound wave for the third plot
    amped_voltage = new_voltage * AMP_FACTOR
    line2.set_ydata(amped_voltage)
    ax3.set_xlim(sound_time[0], sound_time[-1])
    
    # Record the amplified signal (not clipped)
    if recording:
        audio_buffer.extend(voltage_for_recording.tolist())
    
    return line1, spec, line2

def on_close(event):
    global recording, audio_buffer
    if recording:
        recording = False
        print("GUI closed, saving recording...")
        audio_array = np.array(audio_buffer)
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0:
            audio_normalized = audio_array / max_amplitude
        else:
            audio_normalized = audio_array
        audio_int16 = np.int16(audio_normalized * 32767)
        wavfile.write(OUTPUT_FILE, SAMPLE_RATE, audio_int16)
        print(f"Saved audio to {OUTPUT_FILE}")
    ser.close()

fig.canvas.mpl_connect('close_event', on_close)

ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=False, cache_frame_data=False)
plt.show()