#!/usr/bin/env python3
import sys
import requests
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

def check_selenium_health():
    """Check if Selenium WebDriver is working"""
    try:
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        driver.get('data:text/html,<html><body>Health Check</body></html>')
        
        if "Health Check" in driver.page_source:
            driver.quit()
            return True
        
        driver.quit()
        return False
        
    except WebDriverException:
        return False
    except Exception:
        return False

def check_service_health():
    """Check if scraper service is responding"""
    try:
        response = requests.get('http://localhost:8080/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main health check"""
    selenium_ok = check_selenium_health()
    service_ok = check_service_health()
    
    if selenium_ok and service_ok:
        print("Health check passed")
        sys.exit(0)
    else:
        print(f"Health check failed - Selenium: {selenium_ok}, Service: {service_ok}")
        sys.exit(1)

if __name__ == "__main__":
    main()
