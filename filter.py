import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf

def band_pass_filter(data, sample_rate, lowcut=200.0, highcut=1500.0, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def process_wav_file(input_file, output_file, lowcut=200.0, highcut=1500.0):
    data, sample_rate = sf.read(input_file)

    if len(data.shape) == 2:
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = band_pass_filter(data[:, i], sample_rate, lowcut, highcut)
    else:
        filtered_data = band_pass_filter(data, sample_rate, lowcut, highcut)

    sf.write(output_file, filtered_data, sample_rate)
    print(f"Band-pass filtered audio saved as: {output_file}")

# Example usage
input_wav = "test.wav"
output_wav = "output_bandpassed.wav"
process_wav_file(input_wav, output_wav)
