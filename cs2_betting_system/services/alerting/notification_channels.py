import requests

class DiscordChannel:
    def __init__(self, webhook: str | None):
        self.webhook = webhook

    def send(self, content: str):
        if not self.webhook:
            return
        try:
            requests.post(self.webhook, json={"content": content}, timeout=10)
        except Exception:
            pass
