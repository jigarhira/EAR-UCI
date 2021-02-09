import time
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

from audio import Audio


def main():
    # load model
    model_path = './saved_models/3conv_drop_2_small_batch_44pool_32dense_20210209-113537/'
    model = load_model(model_path, compile = True)

    # start microphone recording
    audio = Audio()
    audio.start_live_record()

    # prediction loop
    while True:
        time_start = time.time()

        # get audio sample and convert to mono
        sample = audio.get_audio_sample()
        sample = librosa.to_mono([sample[:, 0], sample[:, 1]])

        # generate spectrogram
        spectrogram = Audio.gen_spec(sample)

        # normalize and reshape input
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        # generate prediction
        prediction = model.predict(spectrogram, batch_size=1)
        prediction = np.argmax(prediction, axis=1)

        print(prediction)
        print('loop time: ' + str(time.time() - time_start))





if __name__ == '__main__':
    main()