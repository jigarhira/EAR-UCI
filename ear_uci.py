import time
import os
import numpy as np
import librosa
from tensorflow.python.keras.backend import switch
from twilio.rest import Client
import tensorflow as tf
from tensorflow.keras.models import load_model

from audio import Audio

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
twilio_num = '***REMOVED***'
user_num = os.getenv("PHONE_NUM")


def main():
    # load model
    model_path = './saved_models/3conv_drop_2_small_batch_44pool_32dense_20210209-113537/'
    model = load_model(model_path, compile = True)

    # start microphone recording
    audio = Audio()
    audio.start_live_record()

    last_prediction = 10
    count = 0
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
                        messaging_service_sid='MG5731096200843a89dfe032f3e807ffe9',
                        body="EAR has detected glass breaking.",
                        to=user_num
                    )
            elif(prediction == 2):
                message = client.messages \
                    .create(
                        messaging_service_sid='MG5731096200843a89dfe032f3e807ffe9',
                        body="EAR has detected a gunshot.",
                        to=user_num
                    )
            elif(prediction == 3):
                message = client.messages \
                    .create(
                        messaging_service_sid='MG5731096200843a89dfe032f3e807ffe9',
                        body="EAR has detected a scream.",
                        to=user_num
                    )                    

        print(prediction)
        print('loop time: ' + str(time.time() - time_start))





if __name__ == '__main__':
    main()