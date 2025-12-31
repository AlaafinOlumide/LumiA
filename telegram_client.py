# telegram_client.py
from __future__ import annotations

import logging
import requests

logger = logging.getLogger("telegram_client")


class TelegramClient:
    """
    Minimal Telegram Bot API client.

    Usage:
        tg = TelegramClient(bot_token, chat_id)
        tg.send_message("hello")
    """

    def __init__(self, bot_token: str, chat_id: str, timeout: int = 15) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

        if not self.bot_token:
            raise ValueError("Telegram bot token is missing.")
        if not self.chat_id:
            raise ValueError("Telegram chat id is missing.")

        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, text: str, *, parse_mode: str = "HTML", disable_web_page_preview: bool = True) -> None:
        """
        Sends a message to the configured chat.
        parse_mode: "HTML" or "Markdown"
        """
        if not text:
            return

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        try:
            r = requests.post(url, data=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok", False):
                raise RuntimeError(f"Telegram API returned ok=false: {data}")
        except Exception as e:
            # Don't crash the bot because Telegram failed
            logger.exception("Failed to send Telegram message: %s", e)