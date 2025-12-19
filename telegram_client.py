# telegram_client.py
import logging
import requests

logger = logging.getLogger("telegram_client")

class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, text: str) -> None:
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                logger.error("Failed to send Telegram message: %s - %s", resp.status_code, resp.text)
        except Exception as e:
            logger.exception("Error sending Telegram message: %s", e)
