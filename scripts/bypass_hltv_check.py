#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HLTV Bypass Checker - Multiple strategies to avoid 403 blocks
"""

import asyncio
import sys
import httpx
import random
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# User agents pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0"
]

class BypassHLTVScraper:
    """HLTV scraper with anti-detection measures"""
    
    def __init__(self):
        self.base_url = "https://www.hltv.org"
        
    def get_headers(self):
        """Get randomized headers"""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
            "Pragma": "no-cache"
        }
    
    async def try_method_1_httpx(self):
        """Method 1: Using httpx with rotation"""
        print("🔄 ลองวิธีที่ 1: httpx + headers rotation...")
        
        try:
            async with httpx.AsyncClient(
                headers=self.get_headers(),
                timeout=15.0,
                follow_redirects=True
            ) as client:
                
                await asyncio.sleep(random.uniform(1, 3))
                response = await client.get(f"{self.base_url}/matches")
                
                if response.status_code == 200:
                    return await self.parse_matches(response.text)
                else:
                    print(f"   ❌ HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    async def try_method_2_mobile(self):
        """Method 2: Mobile user agent"""
        print("🔄 ลองวิธีที่ 2: Mobile user agent...")
        
        mobile_headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        try:
            async with httpx.AsyncClient(
                headers=mobile_headers,
                timeout=15.0
            ) as client:
                
                await asyncio.sleep(random.uniform(2, 4))
                response = await client.get(f"{self.base_url}/matches")
                
                if response.status_code == 200:
                    return await self.parse_matches(response.text)
                else:
                    print(f"   ❌ HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    async def try_method_3_referer(self):
        """Method 3: With referer header"""
        print("🔄 ลองวิธีที่ 3: With referer...")
        
        headers = self.get_headers()
        headers["Referer"] = "https://www.hltv.org/"
        
        try:
            async with httpx.AsyncClient(
                headers=headers,
                timeout=15.0
            ) as client:
                
                # First visit homepage
                await client.get(self.base_url)
                await asyncio.sleep(random.uniform(1, 2))
                
                # Then visit matches
                response = await client.get(f"{self.base_url}/matches")
                
                if response.status_code == 200:
                    return await self.parse_matches(response.text)
                else:
                    print(f"   ❌ HTTP {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    async def parse_matches(self, html):
        """Parse matches from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            matches = []
            
            # Multiple selectors to try
            selectors = [
                '.upcomingMatch',
                '.matchEvent', 
                '[data-zonedgrouping-entry-unix]',
                '.upcoming-match',
                '.match-row'
            ]
            
            match_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    match_elements = elements[:15]
                    print(f"   ✅ พบ elements ด้วย selector: {selector}")
                    break
            
            if not match_elements:
                print("   ⚠️ ไม่พบ match elements")
                return []
            
            for elem in match_elements:
                try:
                    # Find teams
                    team_selectors = ['.team', '.teamName', '.matchTeamName', '.team-name']
                    teams = []
                    
                    for selector in team_selectors:
                        team_elems = elem.select(selector)
                        if len(team_elems) >= 2:
                            teams = [t.get_text(strip=True) for t in team_elems[:2]]
                            break
                    
                    if len(teams) < 2:
                        continue
                    
                    # Find time
                    time_selectors = ['.matchTime', '.match-time', '.time']
                    match_time = "TBD"
                    for selector in time_selectors:
                        time_elem = elem.select_one(selector)
                        if time_elem:
                            match_time = time_elem.get_text(strip=True)
                            break
                    
                    # Find event
                    event_selectors = ['.matchEvent', '.event', '.tournament', '.matchEventName']
                    event = "Unknown"
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
                    
                except Exception:
                    continue
            
            return matches
            
        except Exception as e:
            print(f"   ❌ Parse error: {e}")
            return []
    
    async def get_matches(self):
        """Try multiple methods to get matches"""
        methods = [
            self.try_method_1_httpx,
            self.try_method_2_mobile, 
            self.try_method_3_referer
        ]
        
        for method in methods:
            try:
                result = await method()
                if result:
                    return result
                await asyncio.sleep(random.uniform(2, 5))
            except Exception as e:
                print(f"   ❌ Method failed: {e}")
                continue
        
        return []

async def main():
    """Main function"""
    print("🚀 HLTV Bypass Checker")
    print("=" * 50)
    
    start_time = datetime.now()
    
    try:
        scraper = BypassHLTVScraper()
        matches = await scraper.get_matches()
        
        if not matches:
            print("\n❌ ไม่สามารถดึงข้อมูลได้ - HLTV อาจบล็อก IP")
            print("\n💡 แนะนำ:")
            print("   - ใช้ VPN เปลี่ยน IP")
            print("   - รอสักพัก แล้วลองใหม่")
            print("   - ใช้ API อื่นแทน เช่น Liquipedia")
            return
        
        print(f"\n✅ พบ {len(matches)} แมตช์:")
        print("-" * 50)
        
        for i, match in enumerate(matches, 1):
            team1 = match['team1']
            team2 = match['team2'] 
            time = match['time']
            event = match['event']
            
            if len(event) > 20:
                event = event[:17] + "..."
            
            print(f"{i:2d}. {team1:>10} vs {team2:<10} | {time:>8} | {event}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n⚡ เสร็จใน {duration:.2f} วินาที")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    asyncio.run(main())
