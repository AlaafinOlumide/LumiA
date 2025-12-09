import requests
import logging

logger = logging.getLogger(__name__)

class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_message(self, text: str) -> None:
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token or chat_id not set, skipping send.")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if not r.ok:
                logger.error(
                    "Failed to send Telegram message: %s - %s",
                    r.status_code,
                    r.text,
                )
        except Exception as e:
            logger.exception("Error sending Telegram message: %s", e)
