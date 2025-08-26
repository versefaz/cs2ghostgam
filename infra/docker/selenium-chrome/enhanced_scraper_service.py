import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
import signal
import sys
from contextlib import asynccontextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import undetected_chromedriver as uc
import redis
import requests

from monitoring.prometheus_metrics import get_metrics
from monitoring.alert_system import AlertManager, AlertSeverity, AlertChannel

logger = logging.getLogger(__name__)


class RobustSeleniumManager:
    """Robust Selenium WebDriver management with auto-restart and health monitoring"""
    
    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.driver_start_time: Optional[datetime] = None
        self.restart_count = 0
        self.max_restarts_per_hour = 10
        self.restart_times: List[datetime] = []
        
        # Configuration
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        self.proxy_list = self._load_proxy_list()
        self.current_proxy_index = 0
        
        # Metrics
        self.metrics = get_metrics('selenium_scraper')
        self.alert_manager = AlertManager()
        
        # Health monitoring
        self.last_successful_request = datetime.now()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _load_proxy_list(self) -> List[str]:
        """Load proxy list from environment or config"""
        proxy_env = os.getenv('PROXY_LIST', '')
        if proxy_env:
            return [p.strip() for p in proxy_env.split(',') if p.strip()]
        
        # Default proxy list (replace with actual proxies)
        return [
            # 'http://proxy1:port',
            # 'http://proxy2:port',
        ]
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)
    
    async def initialize(self):
        """Initialize the driver"""
        await self._create_driver()
        
        # Configure alert channels
        discord_webhook = os.getenv('DISCORD_WEBHOOK')
        if discord_webhook:
            self.alert_manager.configure_channel(AlertChannel.DISCORD, {
                'webhook_url': discord_webhook
            })
        
        telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        if all(telegram_config.values()):
            self.alert_manager.configure_channel(AlertChannel.TELEGRAM, telegram_config)
    
    async def _create_driver(self):
        """Create new WebDriver instance with enhanced options"""
        try:
            options = Options()
            
            # Basic options
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')
            options.add_argument('--disable-javascript')  # Enable only when needed
            
            # Anti-detection options
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-web-security')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument('--disable-features=VizDisplayCompositor')
            
            # Performance options
            options.add_argument('--memory-pressure-off')
            options.add_argument('--max_old_space_size=4096')
            options.add_argument('--disable-background-timer-throttling')
            options.add_argument('--disable-renderer-backgrounding')
            
            # Window size
            options.add_argument('--window-size=1920,1080')
            
            # Random user agent
            user_agent = random.choice(self.user_agents)
            options.add_argument(f'--user-agent={user_agent}')
            
            # Proxy rotation
            if self.proxy_list:
                proxy = self.proxy_list[self.current_proxy_index]
                options.add_argument(f'--proxy-server={proxy}')
                self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
                logger.info(f"Using proxy: {proxy}")
            
            # Prefs to disable images and other resources
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_settings.popups": 0,
                "profile.managed_default_content_settings.media_stream": 2,
            }
            options.add_experimental_option("prefs", prefs)
            
            # Use undetected-chromedriver
            try:
                self.driver = uc.Chrome(options=options, version_main=None)
                logger.info("Created undetected Chrome driver")
            except Exception as e:
                logger.warning(f"Failed to create undetected driver: {e}, falling back to regular Chrome")
                self.driver = webdriver.Chrome(options=options)
            
            # Set timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
            
            # Execute stealth scripts
            await self._execute_stealth_scripts()
            
            self.driver_start_time = datetime.now()
            self.consecutive_failures = 0
            
            logger.info("WebDriver initialized successfully")
            self.metrics.record_scraper_event('driver_created', 'success')
            
        except Exception as e:
            logger.error(f"Failed to create WebDriver: {e}")
            self.metrics.record_scraper_event('driver_created', 'failed')
            raise
    
    async def _execute_stealth_scripts(self):
        """Execute JavaScript to make the browser less detectable"""
        stealth_scripts = [
            # Remove webdriver property
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})",
            
            # Mock plugins
            """
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            """,
            
            # Mock languages
            """
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            """,
            
            # Override permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
            """
        ]
        
        for script in stealth_scripts:
            try:
                self.driver.execute_script(script)
            except Exception as e:
                logger.warning(f"Failed to execute stealth script: {e}")
    
    async def get_page(self, url: str, max_retries: int = 3) -> bool:
        """Get page with retry logic and health monitoring"""
        for attempt in range(max_retries):
            try:
                if not self.driver:
                    await self._create_driver()
                
                start_time = time.time()
                
                # Add random delay to appear more human-like
                await asyncio.sleep(random.uniform(1, 3))
                
                self.driver.get(url)
                
                # Wait for page to load
                WebDriverWait(self.driver, 15).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                
                duration = time.time() - start_time
                self.metrics.record_scrape_request('selenium', 'success', duration)
                self.last_successful_request = datetime.now()
                self.consecutive_failures = 0
                
                return True
                
            except (WebDriverException, TimeoutException) as e:
                self.consecutive_failures += 1
                self.metrics.record_scraper_error('selenium', type(e).__name__)
                
                logger.warning(f"Page load failed (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    await asyncio.sleep(wait_time)
                    
                    # Try to restart driver if multiple failures
                    if self.consecutive_failures >= 3:
                        await self._restart_driver()
                else:
                    # Final attempt failed
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        await self._trigger_health_alert()
                    
                    return False
        
        return False
    
    async def find_elements_safe(self, by: By, value: str, timeout: int = 10) -> List:
        """Find elements with safe error handling"""
        try:
            if not self.driver:
                return []
            
            wait = WebDriverWait(self.driver, timeout)
            elements = wait.until(EC.presence_of_all_elements_located((by, value)))
            return elements
            
        except TimeoutException:
            logger.debug(f"Elements not found: {by}={value}")
            return []
        except Exception as e:
            logger.error(f"Error finding elements: {e}")
            self.metrics.record_scraper_error('selenium', 'element_find_error')
            return []
    
    async def execute_script_safe(self, script: str) -> Any:
        """Execute JavaScript with error handling"""
        try:
            if not self.driver:
                return None
            
            return self.driver.execute_script(script)
            
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            self.metrics.record_scraper_error('selenium', 'script_error')
            return None
    
    async def _restart_driver(self):
        """Restart the WebDriver"""
        # Check restart rate limiting
        now = datetime.now()
        self.restart_times = [t for t in self.restart_times if now - t < timedelta(hours=1)]
        
        if len(self.restart_times) >= self.max_restarts_per_hour:
            logger.error("Too many restarts in the last hour, not restarting")
            await self._trigger_restart_limit_alert()
            return
        
        logger.info("Restarting WebDriver...")
        
        try:
            if self.driver:
                self.driver.quit()
        except Exception:
            pass
        
        self.driver = None
        await asyncio.sleep(random.uniform(5, 10))  # Wait before restart
        
        await self._create_driver()
        
        self.restart_count += 1
        self.restart_times.append(now)
        
        logger.info(f"WebDriver restarted (count: {self.restart_count})")
        self.metrics.record_scraper_event('driver_restart', 'success')
    
    async def _trigger_health_alert(self):
        """Trigger health alert for consecutive failures"""
        alert_data = {
            'consecutive_failures': self.consecutive_failures,
            'last_successful_request': self.last_successful_request.isoformat(),
            'restart_count': self.restart_count
        }
        
        await self.alert_manager._trigger_alert(
            rule_id='selenium_health_critical',
            title='Selenium Driver Health Critical',
            message=f'Selenium driver has {self.consecutive_failures} consecutive failures',
            severity=AlertSeverity.CRITICAL,
            metadata=alert_data
        )
    
    async def _trigger_restart_limit_alert(self):
        """Trigger alert when restart limit is reached"""
        await self.alert_manager._trigger_alert(
            rule_id='selenium_restart_limit',
            title='Selenium Restart Limit Reached',
            message=f'Selenium driver has restarted {len(self.restart_times)} times in the last hour',
            severity=AlertSeverity.HIGH,
            metadata={'restart_count': len(self.restart_times)}
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        now = datetime.now()
        uptime = (now - self.driver_start_time).total_seconds() if self.driver_start_time else 0
        
        return {
            'status': 'healthy' if self.consecutive_failures < self.max_consecutive_failures else 'unhealthy',
            'driver_active': self.driver is not None,
            'uptime_seconds': uptime,
            'consecutive_failures': self.consecutive_failures,
            'restart_count': self.restart_count,
            'last_successful_request': self.last_successful_request.isoformat(),
            'restarts_last_hour': len([t for t in self.restart_times if now - t < timedelta(hours=1)])
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up Selenium resources...")
        
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self.driver = None


class EnhancedScraperService:
    """Enhanced scraper service with robust Selenium management"""
    
    def __init__(self):
        self.selenium_manager = RobustSeleniumManager()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.metrics = get_metrics('enhanced_scraper')
        self.running = False
        
        # Scraping targets
        self.scraping_targets = [
            {
                'name': 'hltv_matches',
                'url': 'https://www.hltv.org/matches',
                'interval': 300,  # 5 minutes
                'last_scrape': None
            },
            {
                'name': 'hltv_results',
                'url': 'https://www.hltv.org/results',
                'interval': 600,  # 10 minutes
                'last_scrape': None
            }
        ]
    
    async def start(self):
        """Start the scraper service"""
        logger.info("Starting Enhanced Scraper Service...")
        
        await self.selenium_manager.initialize()
        self.metrics.start_server()
        
        self.running = True
        
        # Start scraping loops
        tasks = [
            asyncio.create_task(self._scraping_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_reporting_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the scraper service"""
        logger.info("Stopping Enhanced Scraper Service...")
        self.running = False
        self.selenium_manager.cleanup()
    
    async def _scraping_loop(self):
        """Main scraping loop"""
        while self.running:
            try:
                for target in self.scraping_targets:
                    if self._should_scrape_target(target):
                        await self._scrape_target(target)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scraping loop: {e}")
                self.metrics.record_scraper_error('scraping_loop', 'general_error')
                await asyncio.sleep(60)  # Wait longer on error
    
    def _should_scrape_target(self, target: Dict[str, Any]) -> bool:
        """Check if target should be scraped"""
        if not target['last_scrape']:
            return True
        
        elapsed = (datetime.now() - target['last_scrape']).total_seconds()
        return elapsed >= target['interval']
    
    async def _scrape_target(self, target: Dict[str, Any]):
        """Scrape a specific target"""
        logger.info(f"Scraping {target['name']}")
        
        start_time = time.time()
        
        try:
            success = await self.selenium_manager.get_page(target['url'])
            
            if success:
                if target['name'] == 'hltv_matches':
                    await self._scrape_hltv_matches()
                elif target['name'] == 'hltv_results':
                    await self._scrape_hltv_results()
                
                target['last_scrape'] = datetime.now()
                self.metrics.record_match_scraped(target['name'], 'success')
            else:
                self.metrics.record_match_scraped(target['name'], 'failed')
            
            duration = time.time() - start_time
            self.metrics.record_scrape_request(target['name'], 'success' if success else 'failed', duration)
            
        except Exception as e:
            logger.error(f"Error scraping {target['name']}: {e}")
            self.metrics.record_scraper_error(target['name'], type(e).__name__)
    
    async def _scrape_hltv_matches(self):
        """Scrape HLTV matches page"""
        try:
            # Find live matches
            live_matches = await self.selenium_manager.find_elements_safe(
                By.CSS_SELECTOR, '.liveMatch-container'
            )
            
            matches_data = []
            
            for match_element in live_matches:
                try:
                    match_data = await self._extract_match_data(match_element)
                    if match_data:
                        matches_data.append(match_data)
                except Exception as e:
                    logger.warning(f"Error extracting match data: {e}")
                    continue
            
            # Store in Redis
            if matches_data:
                await self._store_matches_data(matches_data)
                logger.info(f"Scraped {len(matches_data)} live matches")
            
        except Exception as e:
            logger.error(f"Error scraping HLTV matches: {e}")
            raise
    
    async def _extract_match_data(self, match_element) -> Optional[Dict[str, Any]]:
        """Extract data from a match element"""
        try:
            # This would contain the actual extraction logic
            # For now, return a placeholder
            return {
                'match_id': f'match_{int(time.time())}',
                'team1': 'Team A',
                'team2': 'Team B',
                'status': 'live',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error extracting match data: {e}")
            return None
    
    async def _scrape_hltv_results(self):
        """Scrape HLTV results page"""
        # Similar implementation for results
        pass
    
    async def _store_matches_data(self, matches_data: List[Dict[str, Any]]):
        """Store matches data in Redis"""
        try:
            for match in matches_data:
                key = f"match:live:{match['match_id']}"
                self.redis_client.setex(key, 3600, json.dumps(match, default=str))
            
            # Also store in recent matches list
            self.redis_client.lpush('matches:recent', *[json.dumps(m, default=str) for m in matches_data])
            self.redis_client.ltrim('matches:recent', 0, 99)  # Keep last 100
            
        except Exception as e:
            logger.error(f"Error storing matches data: {e}")
            raise
    
    async def _health_monitoring_loop(self):
        """Monitor service health"""
        while self.running:
            try:
                health_status = self.selenium_manager.get_health_status()
                
                # Store health status in Redis
                health_key = "health:service:enhanced_scraper"
                health_data = {
                    **health_status,
                    'timestamp': datetime.now().isoformat(),
                    'service': 'enhanced_scraper'
                }
                
                self.redis_client.setex(health_key, 120, json.dumps(health_data))
                
                # Check for alerts
                await self._check_health_alerts(health_status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_health_alerts(self, health_status: Dict[str, Any]):
        """Check health status and trigger alerts if needed"""
        if health_status['status'] == 'unhealthy':
            metrics_data = {
                'service_health': {'enhanced_scraper': 'unhealthy'},
                'consecutive_failures': health_status['consecutive_failures']
            }
            
            await self.selenium_manager.alert_manager.check_conditions(metrics_data)
    
    async def _metrics_reporting_loop(self):
        """Report metrics periodically"""
        while self.running:
            try:
                # Update Prometheus metrics
                health_status = self.selenium_manager.get_health_status()
                
                # Report health metrics
                if health_status['status'] == 'healthy':
                    self.metrics.service_status.state('running')
                else:
                    self.metrics.service_status.state('error')
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"Error in metrics reporting: {e}")
                await asyncio.sleep(60)


async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = EnhancedScraperService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
