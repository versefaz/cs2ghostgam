#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast HLTV Match Checker - Optimized for speed
เช็คแมตช์ CS2 แบบเร็ว โดยใช้ simple scraper
"""

import asyncio
import sys
import aiohttp
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent))

class FastHLTVScraper:
    """Fast HLTV scraper with minimal delays"""
    
    def __init__(self):
        self.base_url = "https://www.hltv.org"
        self.session = None
        
    async def __aenter__(self):
        # Fast connection settings
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=10,
            connect=5
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.1)
    
    async def get_upcoming_matches(self, limit=20):
        """Get upcoming matches quickly"""
        try:
            url = f"{self.base_url}/matches"
            
            # Add small delay to appear more human-like
            await asyncio.sleep(0.5)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"❌ HTTP {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                matches = []
                
                # Try multiple selectors for robustness
                selectors = [
                    '.upcomingMatch',
                    '.matchEvent',
                    '[data-zonedgrouping-entry-unix]',
                    '.upcoming-match'
                ]
                
                match_elements = []
                for selector in selectors:
                    elements = soup.select(selector)
                    if elements:
                        match_elements = elements[:limit]
                        break
                
                if not match_elements:
                    print("⚠️ ไม่พบ match elements")
                    return []
                
                for elem in match_elements:
                    try:
                        # Extract teams
                        team_selectors = [
                            '.team',
                            '.teamName', 
                            '.matchTeamName',
                            '.team-name'
                        ]
                        
                        teams = []
                        for selector in team_selectors:
                            team_elems = elem.select(selector)
                            if len(team_elems) >= 2:
                                teams = [t.get_text(strip=True) for t in team_elems[:2]]
                                break
                        
                        if len(teams) < 2:
                            continue
                        
                        # Extract time
                        time_selectors = ['.matchTime', '.match-time', '.time']
                        match_time = "TBD"
                        for selector in time_selectors:
                            time_elem = elem.select_one(selector)
                            if time_elem:
                                match_time = time_elem.get_text(strip=True)
                                break
                        
                        # Extract event
                        event_selectors = ['.matchEvent', '.event', '.tournament', '.matchEventName']
                        event = "Unknown Event"
                        for selector in event_selectors:
                            event_elem = elem.select_one(selector)
                            if event_elem:
                                event = event_elem.get_text(strip=True)
                                break
                        
                        matches.append({
                            'team1': teams[0],
                            'team2': teams[1], 
                            'time': match_time,
                            'event': event
                        })
                        
                    except Exception as e:
                        continue
                
                return matches
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

async def main():
    """Main function - fast match check"""
    print("🚀 Fast HLTV Match Check")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        async with FastHLTVScraper() as scraper:
            matches = await scraper.get_upcoming_matches(limit=15)
            
            if not matches:
                print("❌ ไม่พบแมตช์")
                return
            
            print(f"📊 พบ {len(matches)} แมตช์:")
            print("-" * 60)
            
            for i, match in enumerate(matches, 1):
                team1 = match['team1']
                team2 = match['team2']
                time = match['time']
                event = match['event']
                
                # Truncate long event names
                if len(event) > 25:
                    event = event[:22] + "..."
                
                print(f"{i:2d}. {team1:>12} vs {team2:<12} | {time:>8} | {event}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("-" * 60)
            print(f"⚡ เสร็จใน {duration:.2f} วินาที")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    asyncio.run(main())
