"""Audio loading and processing.

Class for loading, recording, and processing WAV audio samples. 

Author: Jigar Hira, Tritai Nguyen
"""

from typing import Tuple
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd

class Audio:
    # default audio parameters
    SAMPLING_RATE = 44100   # Hz
    DURATION = 3.0          # seconds
    MONO = True             # channels
    N_FFT = 1024            # Window size
    HOP_LENGTH = 512        # Length samples between windows
    N_MELS = 128            # Number of Mel filters
    BUFFER = np.array(np.zeros(SAMPLING_RATE*int(DURATION)*2, dtype=float))   # Buffer for recording audio
    EPSILON = 0.2           # delay to start recording

    @classmethod
    def load_sample(self, path: str, offset=0.0) -> Tuple[np.ndarray, int]:
        """Loads WAV audio sample.

        Args:
            path (str): File path.
            offset (float): Start reading audio after this time (seconds). Defaults to 0.0.

        Returns:
            np.ndarray: Numpy array audio time series.
            int: Output sampling rate.
        """
        # load audio file into numpy array
        audio, sr = librosa.load(path, sr=self.SAMPLING_RATE, mono=self.MONO, offset=offset, duration=self.DURATION)
        # if audio is less than 3 seconds, pad with zeros
        audio = np.pad(audio, (0, int(self.DURATION * self.SAMPLING_RATE) - len(audio)), constant_values=0)
        return audio, sr
    
    @classmethod
    def write_sample(self, path: str, audio: np.ndarray, sampling_rate: int) -> None:
        """Writes audio sample to WAV file.

        Args:
            path (str): File path.
            audio (np.ndarray): Numpy array audio time series.
            sampling_rate (int): Audio sampling rate (Hz).
        """
        sf.write(path, audio, sampling_rate)

    @classmethod
    def record_sample(self) -> np.ndarray:
        """Records audio.

        Returns:
            np.ndarray: Numpy array audio time series.
        """
        sampled_audio = sd.rec(int((self.DURATION+self.EPSILON)*self.SAMPLING_RATE), samplerate=self.SAMPLING_RATE, channels=1)
        sd.wait()
        sampled_audio = sampled_audio[int(self.EPSILON*self.SAMPLING_RATE):int((self.DURATION+self.EPSILON)*self.SAMPLING_RATE)]
        return sampled_audio        

    @classmethod
    def gen_spec(self, signal: np.ndarray) -> np.ndarray:
        """Generates log-based mel-spectrogram time series array.

        Args:
            signal (np.ndarray): Numpy array audio time series. 
            
        Returns:
            np.ndarray: Numpy array mel-spectrogram.
        """
        spec = librosa.feature.melspectrogram(signal, sr=self.SAMPLING_RATE, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH, n_mels=self.N_MELS)
        spec = librosa.power_to_db(spec, ref=np.max)
        return spec

    @classmethod
    def record_sample_mem(self) -> None:
        """Records audio into buffer for spectrogram generation.

        Returns:
            np.ndarray: Numpy array audio time series.
        """
        sampled_audio = sd.rec(int((self.DURATION+self.EPSILON)*self.SAMPLING_RATE), samplerate=self.SAMPLING_RATE, channels=1)
        sd.wait()
        sampled_audio = sampled_audio[int(self.EPSILON*self.SAMPLING_RATE):int((self.DURATION+self.EPSILON)*self.SAMPLING_RATE)]
        Audio.BUFFER = np.concatenate((np.squeeze(sampled_audio), Audio.BUFFER[0:int(Audio.DURATION)*2*Audio.SAMPLING_RATE-len(sampled_audio)]))
        return None    
  

if __name__ == "__main__":

    path = 'C:/UCI/Senior Year/159_senior_design/output.wav'
    # record, write, and load audio
    # signal = Audio.record_sample()
    # #print(signal)
    # Audio.write_sample(path, signal, 44100)
    # normalized, sr = Audio.load_sample(path)
    # spectrogram = Audio.gen_spec(normalized)
    

    # import librosa.display
    # import matplotlib.pyplot as plt
    # librosa.display.specshow(spectrogram, sr=44100, hop_length=512, x_axis='time', y_axis='mel') #display with frequency axis in mel scale
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()

    while True:
        Audio.record_sample_mem()
        spectrogram = Audio.gen_spec(Audio.BUFFER[44100*3*2-(3*44100):44100*3*2])
        # display mel-spectrogram
        import librosa.display
        import matplotlib.pyplot as plt
        librosa.display.specshow(spectrogram, sr=44100, hop_length=512, x_axis='time', y_axis='mel') #display with frequency axis in mel scale
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    # generate spectrograms for all folders in train
    # import os
    # rootdir = r'C:\Users\Tritai\Desktop\Audio'
 
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         wav = os.path.join(subdir, file)
    #         normalized, sr = Audio.load_sample(wav)
    #         audio = Audio.gen_spec(normalized)
    #         np.save(os.path.splitext(wav)[0], audio)

