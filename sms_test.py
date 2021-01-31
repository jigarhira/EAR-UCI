import os
from twilio.rest import Client


account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
twilio_num = '***REMOVED***'
user_num = '***REMOVED***'

message = client.messages \
                .create(
                    messaging_service_sid='MG5731096200843a89dfe032f3e807ffe9',
                    body="Testing twilio sms messaging.",
                    to=user_num
                )
print(message.sid)