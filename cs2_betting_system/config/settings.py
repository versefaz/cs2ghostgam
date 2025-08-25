import os
from dotenv import load_dotenv

load_dotenv()

# Trading Settings
INITIAL_BALANCE = int(os.getenv('INITIAL_BALANCE', '10000'))
MAX_BET_SIZE = float(os.getenv('MAX_BET_SIZE', '0.05'))  # 5% bankroll
MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.75'))
MIN_VALUE = float(os.getenv('MIN_VALUE', '1.15'))
MAX_EXPOSURE = float(os.getenv('MAX_EXPOSURE', '0.20'))
MAX_BETS_PER_DAY = int(os.getenv('MAX_BETS_PER_DAY', '10'))

# Scraping Settings
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', '60'))  # seconds
USER_AGENT = os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
SELENIUM_WAIT = int(os.getenv('SELENIUM_WAIT', '15'))
ODDSPORTAL_URL = os.getenv('ODDSPORTAL_URL', 'https://www.oddsportal.com/matches/counter-strike/')
ODDSPORTAL_SEARCH_URL = os.getenv('ODDSPORTAL_SEARCH_URL', 'https://www.oddsportal.com/search/results/all/?q=')
BOOKMAKER_PRIORITY = [
    b.strip() for b in os.getenv(
        'BOOKMAKER_PRIORITY', 'Pinnacle, bet365, 1xBet, Marathonbet, Unibet'
    ).split(',')
]
ODDS_CACHE_TTL = int(os.getenv('ODDS_CACHE_TTL', '300'))  # seconds
USE_UC = os.getenv('USE_UC', 'true').lower() in ('1', 'true', 'yes')
USE_PROXY = os.getenv('USE_PROXY', 'false').lower() in ('1', 'true', 'yes')
PROXY_URL = os.getenv('PROXY_URL', '')

# Database
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/cs2betting')

# Model Settings
MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved/best_model.pkl')
FEATURE_COLUMNS = [
    'team1_rating', 'team2_rating', 'team1_form', 'team2_form',
    'h2h_score', 'map_winrate_diff', 'odds_movement', 'avg_odds'
]
MODEL_PROBABILITY_THRESHOLD = float(os.getenv('MODEL_PROBABILITY_THRESHOLD', '0.5'))

# Alert Settings
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Metrics / Monitoring
METRICS_PORT_PREDICTION = int(os.getenv('METRICS_PORT_PREDICTION', '9092'))
METRICS_PORT_SCRAPER = int(os.getenv('METRICS_PORT_SCRAPER', '9091'))

