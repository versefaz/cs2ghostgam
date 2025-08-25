"""Anti-detection middleware for web scraping"""
import random
from typing import Dict, List
from fake_useragent import UserAgent

class AntiDetectionMiddleware:
    """Implements various anti-detection techniques"""

    def __init__(self):
        self.ua = UserAgent()
        self.fingerprints = self._load_fingerprints()

    def _load_fingerprints(self) -> List[Dict]:
        """Load browser fingerprint profiles"""
        return [
            {
                "screen": {"width": 1920, "height": 1080, "colorDepth": 24},
                "timezone": "America/New_York",
                "language": "en-US",
                "platform": "Win32",
                "hardwareConcurrency": 8,
                "deviceMemory": 8,
            },
            {
                "screen": {"width": 2560, "height": 1440, "colorDepth": 24},
                "timezone": "Europe/London",
                "language": "en-GB",
                "platform": "MacIntel",
                "hardwareConcurrency": 12,
                "deviceMemory": 16,
            },
            {
                "screen": {"width": 1366, "height": 768, "colorDepth": 24},
                "timezone": "America/Chicago",
                "language": "en-US",
                "platform": "Win32",
                "hardwareConcurrency": 4,
                "deviceMemory": 4,
            },
        ]

    def get_random_user_agent(self) -> str:
        """Get random user agent string"""
        return self.ua.random

    def get_stealth_script(self) -> str:
        """Get JavaScript for stealth mode"""
        fingerprint = random.choice(self.fingerprints)
        width = fingerprint['screen']['width']
        height = fingerprint['screen']['height']
        hw = fingerprint['hardwareConcurrency']
        mem = fingerprint['deviceMemory']
        lang = fingerprint['language']

        return f"""
        // Override webdriver
        Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined }});

        // Languages
        Object.defineProperty(navigator, 'languages', {{ get: () => ['{lang}', '{lang.split('-')[0]}'] }});

        // Plugins
        Object.defineProperty(navigator, 'plugins', {{ get: () => [1, 2, 3, 4, 5] }});

        // Chrome runtime
        window.chrome = {{ runtime: {{}} }};

        // Permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({{ state: Notification.permission }}) :
                originalQuery(parameters)
        );

        // Screen dimensions
        Object.defineProperty(screen, 'width', {{ get: () => {width} }});
        Object.defineProperty(screen, 'height', {{ get: () => {height} }});

        // Hardware
        Object.defineProperty(navigator, 'hardwareConcurrency', {{ get: () => {hw} }});
        Object.defineProperty(navigator, 'deviceMemory', {{ get: () => {mem} }});
        """
