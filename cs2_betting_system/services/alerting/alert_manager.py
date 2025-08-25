import os
import requests
from cs2_betting_system.config import settings
from .notification_channels import DiscordChannel

class AlertManager:
    def __init__(self):
        self.discord = DiscordChannel(settings.DISCORD_WEBHOOK)

    def critical(self, message: str):
        if settings.DISCORD_WEBHOOK:
            self.discord.send(f"🚨 CRITICAL: {message}")
