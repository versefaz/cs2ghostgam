#!/usr/bin/env python3
"""
Application Configuration - Load scraper settings and other configs
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def load_scraper_config() -> Dict[str, Any]:
    """Load scraper configuration from YAML file"""
    config_path = PROJECT_ROOT / "config" / "scraper_config.yaml"
    
    if not config_path.exists():
        # Return default config if file doesn't exist
        return get_default_scraper_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading scraper config: {e}")
        return get_default_scraper_config()

def get_default_scraper_config() -> Dict[str, Any]:
    """Default scraper configuration as fallback"""
    return {
        'hltv': {
            'base_interval_sec': 15,
            'jitter_pct': 0.20,
            'max_backoff': 3
        },
        'odds': {
            'base_interval_sec': 360,
            'jitter_pct': 0.25,
            'max_backoff': 3
        },
        'team_stats': {
            'base_interval_sec': 900,
            'jitter_pct': 0.25,
            'max_backoff': 4
        },
        'live_scores': {
            'base_interval_sec': 60,
            'jitter_pct': 0.40,
            'max_backoff': 5
        },
        'concurrency': {
            'max_connections': 8,
            'connection_timeout': 10,
            'read_timeout': 15
        },
        'user_agents': [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
        ],
        'rate_limiting': {
            'respect_robots_txt': True,
            'min_delay_between_requests': 0.5,
            'retry_attempts': 2
        },
        'http_handling': {
            'rate_limit_codes': [429, 503, 502],
            'success_codes': [200, 201, 202],
            'client_error_codes': [400, 401, 403, 404]
        }
    }

# Load configuration on import
SCRAPER_SETTINGS = load_scraper_config()

# Export commonly used settings
USER_AGENTS = SCRAPER_SETTINGS['user_agents']
CONCURRENCY_SETTINGS = SCRAPER_SETTINGS['concurrency']
RATE_LIMIT_CODES = SCRAPER_SETTINGS['http_handling']['rate_limit_codes']
SUCCESS_CODES = SCRAPER_SETTINGS['http_handling']['success_codes']
