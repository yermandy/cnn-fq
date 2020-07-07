import requests

def send_photo_telegram(photo_path):
    
    bot_token = ''
    bot_chat_id = ''

    response = requests.post(f'https://api.telegram.org/bot{bot_token}/sendPhoto', 
        data={'chat_id': bot_chat_id},
        files={'photo': open(photo_path, 'rb')}
    )

    return response.json()