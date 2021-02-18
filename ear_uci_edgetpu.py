import time
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite

from audio import Audio


def main():
    # load model
    model_path = './saved_models/3conv_drop_2_small_batch_44pool_32dense_20210209-113537/saved_model.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # get input and output information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
        interpreter.set_tensor(input_details[0]['index'], spectrogram)
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']), axis=1)

        print(prediction)
        print('loop time: ' + str(time.time() - time_start))


if __name__ == '__main__':
    main()