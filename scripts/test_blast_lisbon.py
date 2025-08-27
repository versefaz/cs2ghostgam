#!/usr/bin/env python3
"""
Test BLAST Lisbon Match Scraper - Fetch tomorrow's matches
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx
from bs4 import BeautifulSoup

# Fix Windows console encoding
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import setup_logger

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

class BLASTLisbonScraper:
    """Scraper for BLAST tournament matches"""
    
    def __init__(self):
        self.logger = setup_logger("blast_scraper")
        self.base_url = "https://www.hltv.org"
        
    async def scrape_upcoming_matches(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """Scrape upcoming matches for specified days ahead"""
        try:
            url = f"{self.base_url}/matches"
            
            async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
                self.logger.info(f"Fetching matches from: {url}")
                response = await client.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                matches = []
                
                # Look for match containers
                match_containers = soup.find_all(['div'], class_=['upcomingMatch', 'liveMatch'])
                
                self.logger.info(f"Found {len(match_containers)} match containers")
                
                for container in match_containers:
                    try:
                        match_data = await self._parse_match_container(container)
                        if match_data and self._is_blast_event(match_data.get('event_name', '')):
                            matches.append(match_data)
                    except Exception as e:
                        self.logger.warning(f"Error parsing match container: {e}")
                        continue
                
                # Also try alternative selectors
                if not matches:
                    self.logger.info("Trying alternative match selectors...")
                    alt_containers = soup.find_all(['a'], href=lambda x: x and '/matches/' in x)
                    
                    for container in alt_containers[:20]:  # Limit to prevent too many requests
                        try:
                            match_data = await self._parse_alternative_match(container)
                            if match_data and self._is_blast_event(match_data.get('event_name', '')):
                                matches.append(match_data)
                        except Exception as e:
                            continue
                
                return matches
                
        except Exception as e:
            self.logger.error(f"Error scraping matches: {e}")
            return []
    
    async def _parse_match_container(self, container) -> Optional[Dict[str, Any]]:
        """Parse individual match container"""
        try:
            # Extract match ID
            match_id = (container.get('data-zonedgrouping-entry-unix') or 
                       container.get('data-match-id') or 
                       container.get('href', '').split('/')[-1])
            
            # Extract team names
            team1_elem = container.find(['div', 'span'], class_=['team1', 'matchTeamName', 'teamName'])
            team2_elem = container.find(['div', 'span'], class_=['team2', 'matchTeamName', 'teamName'])
            
            if not team1_elem or not team2_elem:
                # Try alternative selectors
                team_elems = container.find_all(['div', 'span'], string=lambda x: x and len(x.strip()) > 2)
                if len(team_elems) >= 2:
                    team1_elem, team2_elem = team_elems[0], team_elems[1]
                else:
                    return None
            
            team1 = team1_elem.get_text(strip=True) if team1_elem else "Unknown"
            team2 = team2_elem.get_text(strip=True) if team2_elem else "Unknown"
            
            # Extract event name
            event_elem = container.find(['div', 'span'], class_=['matchEventName', 'event'])
            event_name = event_elem.get_text(strip=True) if event_elem else ""
            
            # Extract time
            time_elem = container.find(['div', 'span'], class_=['matchTime', 'time'])
            match_time = time_elem.get_text(strip=True) if time_elem else ""
            
            return {
                'match_id': match_id,
                'team1': team1,
                'team2': team2,
                'event_name': event_name,
                'match_time': match_time,
                'url': container.get('href', ''),
                'raw_html': str(container)[:200] + "..." if len(str(container)) > 200 else str(container)
            }
            
        except Exception as e:
            self.logger.warning(f"Error parsing match container: {e}")
            return None
    
    async def _parse_alternative_match(self, container) -> Optional[Dict[str, Any]]:
        """Parse match using alternative method"""
        try:
            href = container.get('href', '')
            if not href or '/matches/' not in href:
                return None
            
            match_id = href.split('/')[-1]
            text_content = container.get_text(strip=True)
            
            # Try to extract team names from text
            if ' vs ' in text_content:
                teams = text_content.split(' vs ')
                if len(teams) >= 2:
                    return {
                        'match_id': match_id,
                        'team1': teams[0].strip(),
                        'team2': teams[1].strip(),
                        'event_name': "BLAST (detected)",
                        'match_time': "TBD",
                        'url': href,
                        'raw_html': str(container)
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _is_blast_event(self, event_name: str) -> bool:
        """Check if event is BLAST related"""
        blast_keywords = ['blast', 'lisbon', 'london', 'premier']
        event_lower = event_name.lower()
        return any(keyword in event_lower for keyword in blast_keywords)
    
    async def scrape_blast_schedule(self) -> List[Dict[str, Any]]:
        """Try to scrape BLAST specific schedule"""
        try:
            # Try BLAST specific URLs
            blast_urls = [
                "https://www.hltv.org/events/7148/blast-premier-fall-final-2024",
                "https://www.hltv.org/matches?event=7148",
                "https://www.hltv.org/events"
            ]
            
            matches = []
            
            async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
                for url in blast_urls:
                    try:
                        self.logger.info(f"Trying BLAST URL: {url}")
                        response = await client.get(url)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for match elements
                        match_elements = soup.find_all(['div', 'a'], class_=lambda x: x and ('match' in x.lower() or 'fixture' in x.lower()))
                        
                        for elem in match_elements[:10]:  # Limit results
                            text = elem.get_text(strip=True)
                            if any(team in text for team in ['Vitality', 'FaZe', 'Navi', 'Astralis', 'G2']):
                                matches.append({
                                    'source': url,
                                    'text': text,
                                    'html': str(elem)[:200] + "..."
                                })
                        
                    except Exception as e:
                        self.logger.warning(f"Error with URL {url}: {e}")
                        continue
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error scraping BLAST schedule: {e}")
            return []


def print_header():
    """Print formatted header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print("\n" + "=" * 80)
    print("[*] BLAST LISBON MATCH SCRAPER TEST")
    print(f"[+] Current Time: {current_time}")
    print(f"[+] Target Date: {tomorrow}")
    print("=" * 80)


def print_matches(matches: List[Dict[str, Any]]):
    """Print formatted match results"""
    if not matches:
        print("\n[INFO] No BLAST matches found for tomorrow")
        return
    
    print(f"\n[MATCHES] Found {len(matches)} BLAST matches:")
    print("=" * 80)
    
    for i, match in enumerate(matches, 1):
        print(f"\n+-- Match {i} {'-' * 60}+")
        print(f"| ID: {match.get('match_id', 'N/A'):<65} |")
        print(f"| Teams: {match.get('team1', 'N/A')} vs {match.get('team2', 'N/A'):<45} |")
        print(f"| Event: {match.get('event_name', 'N/A'):<60} |")
        print(f"| Time: {match.get('match_time', 'N/A'):<62} |")
        if match.get('url'):
            print(f"| URL: {match.get('url', 'N/A'):<64} |")
        print(f"+{'-' * 70}+")


async def main():
    """Main test function"""
    logger = setup_logger("blast_test")
    
    try:
        print_header()
        
        # Initialize scraper
        print("\n[INIT] Initializing BLAST Lisbon scraper...")
        scraper = BLASTLisbonScraper()
        
        # Test 1: Scrape upcoming matches
        print("\n[TEST 1] Scraping upcoming matches...")
        matches = await scraper.scrape_upcoming_matches()
        print_matches(matches)
        
        # Test 2: Try BLAST specific scraping
        print("\n[TEST 2] Trying BLAST specific scraping...")
        blast_matches = await scraper.scrape_blast_schedule()
        
        if blast_matches:
            print(f"\n[BLAST] Found {len(blast_matches)} potential BLAST entries:")
            for i, match in enumerate(blast_matches[:5], 1):
                print(f"{i}. Source: {match.get('source', 'N/A')}")
                print(f"   Text: {match.get('text', 'N/A')[:100]}...")
                print()
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "upcoming_matches": matches,
            "blast_specific": blast_matches
        }
        
        output_file = Path("data/test_blast_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVE] Results saved to: {output_file}")
        print("=" * 80)
        
        logger.info(f"Test completed. Found {len(matches)} matches, {len(blast_matches)} BLAST entries")
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Test cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        logger.error(f"Test error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
