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
    
    @classmethod
    def load_sample(self, path: str, offset=0.0) -> Tuple[np.ndarray, int]:
        """Loads WAV audio sample.

        Args:
            path (str): File path.
            offset (float, optional): Start reading audio after this time (seconds). Defaults to 0.0.

        Returns:
            np.ndarray: Numpy array audio time series.
            int: Output sampling rate.
        """
        return librosa.load(path, sr=self.SAMPLING_RATE, mono=self.MONO, offset=offset, duration=self.DURATION)
    
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
        sampled_audio = sd.rec(int(self.DURATION*self.SAMPLING_RATE), samplerate=self.SAMPLING_RATE, channels=1)
        sd.wait()
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

if __name__ == "__main__":

    path = 'C:/UCI/Senior Year/159_senior_design/output.wav'
    # record, write, and load audio
    signal = Audio.record_sample()
    Audio.write_sample(path, signal, 44100)
    normalized, sr = Audio.load_sample(path)
    spectrogram = Audio.gen_spec(normalized)
