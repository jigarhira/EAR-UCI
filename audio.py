"""Audio loading and processing.

Class for loading, recording, and processing WAV audio samples. 

Author: Jigar Hira, Tritai Nguyen
"""

from typing import Tuple
from scipy.io.wavfile import write
import numpy as np
import collections
import sys
import time
import copy
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
    SPECTROGRAM = []

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
        sampled_audio = sd.rec(int((self.DURATION)*self.SAMPLING_RATE), samplerate=self.SAMPLING_RATE, channels=1)
        sd.wait()
        write("example.wav", Audio.SAMPLING_RATE, sampled_audio.astype(np.float))
        Audio.BUFFER = np.concatenate((np.squeeze(sampled_audio), Audio.BUFFER[0:int(self.DURATION)*2*self.SAMPLING_RATE-len(sampled_audio)]))
        return None

    def input_stream_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback function for live audio recording.

        Args:
            indata (np.ndarray): audio data
            frames (int): number of samples returned
            time: sample time
            status: data status
        """
        # check for error status
        if status:
            print(status, file=sys.stderr)

        # add audio data to buffer
        self.buffer.extend(indata.copy())
    
    def start_live_record(self, device=None, buffer_duration=3):
        """Starts live audio recording from input device.

        Args:
            device (int): Input device. Defaults to None.
            buffer_duration (int): Buffer duration in seconds. Defaults to 3.
        """
        # create audio buffer queue
        self.buffer_size = buffer_duration * self.SAMPLING_RATE
        self.buffer = collections.deque(maxlen=self.buffer_size)

        # create audio input stream
        self.stream = sd.InputStream(
            samplerate=self.SAMPLING_RATE,
            blocksize=int(self.SAMPLING_RATE / 10),
            device=device,
            callback=self.input_stream_callback
        )

        # start stream
        self.stream.start()

    def get_audio_sample(self) -> np.ndarray:
        """Returns current audio sample from the live audio buffer.

        Returns:
            np.ndarray: audio sample as numpy array
        """
        # wait until buffer is full
        while len(self.buffer) != self.buffer_size:
            pass

        # return sample
        return np.array(list(self.buffer))


if __name__ == "__main__":
    audio = Audio()
    audio.start_live_record()

    spectrogram = []

    sample = audio.get_audio_sample()
    sample = librosa.to_mono([sample[:, 0], sample[:, 1]])

    spectrogram.append(Audio.gen_spec(sample))

    # path = 'C:/UCI/Senior Year/159_senior_design/output.wav'
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

    # import librosa.display
    # import matplotlib.pyplot as plt
    # from tensorflow.keras.models import load_model
    # model_path = './saved_models'
    # model = load_model(model_path, compile = True)

    # while True:
    #     Audio.record_sample_mem()
    #     spectrogram = Audio.gen_spec(Audio.BUFFER[3*44100:44100*3*2])
    #     predictions = model.predict(spectrogram)
    #     print(predictions)
    #     Audio.SPECTROGRAM = np.append(spectrogram)
    #     print(np.shape(spectrogram))

    #     # display mel-spectrogram
    #     librosa.display.specshow(spectrogram, sr=44100, hop_length=512, x_axis='time', y_axis='mel') #display with frequency axis in mel scale
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.show()

    # generate spectrograms for all folders in train
    # import os
    # rootdir = r'C:\Users\Tritai\Desktop\Audio'
 
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         wav = os.path.join(subdir, file)
    #         normalized, sr = Audio.load_sample(wav)
    #         audio = Audio.gen_spec(normalized)
    #         np.save(os.path.splitext(wav)[0], audio)

