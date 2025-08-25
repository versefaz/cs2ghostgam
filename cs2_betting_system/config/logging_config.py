import logging
from logging.handlers import RotatingFileHandler
import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_DIR = os.getenv('LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'cs2_betting.log')

os.makedirs(LOG_DIR, exist_ok=True)

formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setFormatter(formatter)

logging.basicConfig(level=LOG_LEVEL, handlers=[handler, console])
