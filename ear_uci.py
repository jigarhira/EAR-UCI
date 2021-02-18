import time
import os
import numpy as np
import librosa
from tensorflow.python.keras.backend import switch
from twilio.rest import Client
import tensorflow as tf
from tensorflow.keras.models import load_model
from pycoral.utils import edgetpu
from pycoral.adapters import common, classify

from audio import Audio

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
twilio_num = os.getenv("TWILIO_PHONE_NUM")
user_num = os.getenv("PHONE_NUM")


def main():
    # load model
    model_path = r'C:\UCI\Senior Year\Winter_2021\159_senior_design\EAR-UCI\saved_models\3conv_drop_2_small_batch_44pool_32dense_3k_20210217-223630\saved_model.tflite'
    # model = load_model(model_path, compile = True)
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    # start microphone recording
    audio = Audio()
    audio.start_live_record()

    last_prediction = 10
    count = 0
    # prediction loop
    while True:
        # time_start = time.time()

        # get audio sample and convert to mono
        sample = audio.get_audio_sample()
        sample = librosa.to_mono([sample[:, 0], sample[:, 1]])

        # generate spectrogram
        spectrogram = Audio.gen_spec(sample)

        # normalize and reshape input
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spectrogram = np.expand_dims(spectrogram, axis=0)
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        common.set_input(interpreter, image)
        interpreter.invoke()
        prediction = classify.get_classes(interpreter, top_k=1)

        # generate prediction
        # prediction = model.predict(spectrogram, batch_size=1)
        prediction = np.argmax(prediction, axis=1)

        # increment counter if current prediction matches last prediction
        if(prediction == last_prediction):
            count = count + 1
        else:
            count = 0

        #update last_prediction
        last_prediction = prediction

        if(count >= 3):
            if(prediction == 1):
                message = client.messages \
                    .create(
                        # messaging_service_sid='SM724163fa074e42c8a1eac0ea276310af',
                        body="EAR has detected glass breaking.",
                        from_=twilio_num,
                        to=user_num
                    )
            elif(prediction == 2):
                message = client.messages \
                    .create(
                        # messaging_service_sid='SM724163fa074e42c8a1eac0ea276310af',
                        body="EAR has detected a gunshot.",
                        from_=twilio_num,
                        to=user_num
                    )
            elif(prediction == 3):
                message = client.messages \
                    .create(
                        # messaging_service_sid='SM724163fa074e42c8a1eac0ea276310af',
                        body="EAR has detected a scream.",
                        from_=twilio_num,
                        to=user_num
                    )                    

        print(prediction)
        # print('loop time: ' + str(time.time() - time_start))





if __name__ == '__main__':
    main()
