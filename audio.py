"""Audio loading and processing.

Class for loading and processing WAV audio samples.

Author: Jigar Hira
"""

from typing import Tuple
import numpy as np
import librosa
import soundfile as sf


class Audio:
    # default audio parameters
    SAMPLING_RATE = 44100   # Hz
    DURATION = 3.0          # seconds
    MONO = True             # channels

    
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


if __name__ == "__main__":
    normalized, sr = Audio.load_sample('./dataset/test_32.wav')
    Audio.write_sample('./dataset/test_out.wav', normalized, sr)