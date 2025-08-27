#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Odds Fetcher - ดึงราคาเดิมพันจริงจากเว็บพนันชั้นนำ
Created by KoJao - ระบบวิเคราะห์ CS2 ระดับโลก
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup

@dataclass
class LiveOdds:
    """ราคาเดิมพันสด"""
    bookmaker: str
    match: str
    bet_type: str
    selection: str
    odds: float
    timestamp: datetime
    market_volume: Optional[float] = None
    
@dataclass
class BestOdds:
    """ราคาดีที่สุด"""
    bet_type: str
    selection: str
    best_odds: float
    best_bookmaker: str
    all_odds: List[LiveOdds]
    expected_value: float
    confidence: str
    risk_level: str

class LiveOddsFetcher:
    """ตัวดึงราคาเดิมพันสด"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Headers สำหรับหลบ detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # API endpoints สำหรับเว็บพนันต่างๆ
        self.odds_sources = {
            'pinnacle': {
                'url': 'https://www.pinnacle.com/en/esports/counter-strike/matchups',
                'api': 'https://guest.api.arcadia.pinnacle.com/0.1/leagues/4281/matchups',
                'parser': self._parse_pinnacle_odds
            },
            'bet365': {
                'url': 'https://www.bet365.com/#/AC/B1/C1/D1002/E174/F2/',
                'parser': self._parse_bet365_odds
            },
            'betway': {
                'url': 'https://betway.com/en/sports/grp/esports/ctr/counter-strike',
                'parser': self._parse_betway_odds
            },
            'ggbet': {
                'url': 'https://gg.bet/en/counter-strike',
                'api': 'https://gg.bet/api/sport/line',
                'parser': self._parse_ggbet_odds
            }
        }
    
    async def __aenter__(self):
        """เริ่มต้น async session"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ปิด async session"""
        if self.session:
            await self.session.close()
    
    async def fetch_live_odds(self, match_teams: List[Tuple[str, str]]) -> Dict[str, List[BestOdds]]:
        """ดึงราคาเดิมพันสดสำหรับแมตช์ที่กำหนด"""
        
        all_match_odds = {}
        
        for team1, team2 in match_teams:
            match_key = f"{team1}_vs_{team2}"
            self.logger.info(f"🔍 กำลังดึงราคาสำหรับ {team1} vs {team2}")
            
            # ดึงราคาจากแต่ละเว็บ
            all_odds = []
            
            # ดึงจาก Pinnacle (มีความแม่นยำสูง)
            pinnacle_odds = await self._fetch_pinnacle_odds(team1, team2)
            all_odds.extend(pinnacle_odds)
            
            # ดึงจาก GG.bet (เชี่ยวชาญ esports)
            ggbet_odds = await self._fetch_ggbet_odds(team1, team2)
            all_odds.extend(ggbet_odds)
            
            # ดึงจาก Betway
            betway_odds = await self._fetch_betway_odds(team1, team2)
            all_odds.extend(betway_odds)
            
            # วิเคราะห์ราคาดีที่สุด
            best_odds = self._analyze_best_odds(all_odds, team1, team2)
            all_match_odds[match_key] = best_odds
            
            # หน่วงเวลาเพื่อไม่ให้โดน rate limit
            await asyncio.sleep(2)
        
        return all_match_odds
    
    async def _fetch_pinnacle_odds(self, team1: str, team2: str) -> List[LiveOdds]:
        """ดึงราคาจาก Pinnacle"""
        odds = []
        
        try:
            # ลอง API endpoint ก่อน
            api_url = "https://guest.api.arcadia.pinnacle.com/0.1/leagues/4281/matchups"
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for match in data:
                        if self._match_teams(match.get('participants', []), team1, team2):
                            # Match Winner
                            if 'periods' in match:
                                for period in match['periods']:
                                    if period.get('number') == 0:  # Full match
                                        for market in period.get('moneyline', []):
                                            odds.append(LiveOdds(
                                                bookmaker="Pinnacle",
                                                match=f"{team1} vs {team2}",
                                                bet_type="Match Winner",
                                                selection=market.get('designation'),
                                                odds=float(market.get('price', 0)),
                                                timestamp=datetime.now()
                                            ))
                            
                            # Handicap
                            if 'spreads' in match:
                                for spread in match['spreads']:
                                    odds.append(LiveOdds(
                                        bookmaker="Pinnacle",
                                        match=f"{team1} vs {team2}",
                                        bet_type="Handicap",
                                        selection=f"{spread.get('designation')} {spread.get('hdp')}",
                                        odds=float(spread.get('price', 0)),
                                        timestamp=datetime.now()
                                    ))
                            
                            # Total Maps
                            if 'totals' in match:
                                for total in match['totals']:
                                    odds.append(LiveOdds(
                                        bookmaker="Pinnacle",
                                        match=f"{team1} vs {team2}",
                                        bet_type="Total Maps",
                                        selection=f"{total.get('designation')} {total.get('points')}",
                                        odds=float(total.get('price', 0)),
                                        timestamp=datetime.now()
                                    ))
                            
                            break
        
        except Exception as e:
            self.logger.warning(f"ไม่สามารถดึงราคาจาก Pinnacle: {e}")
        
        return odds
    
    async def _fetch_ggbet_odds(self, team1: str, team2: str) -> List[LiveOdds]:
        """ดึงราคาจาก GG.bet"""
        odds = []
        
        try:
            # GG.bet API
            api_url = "https://gg.bet/api/sport/line"
            params = {
                'sport': 'counter-strike',
                'limit': 50
            }
            
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'events' in data:
                        for event in data['events']:
                            event_name = event.get('name', '')
                            
                            if (team1.lower() in event_name.lower() and 
                                team2.lower() in event_name.lower()):
                                
                                # Parse markets
                                for market in event.get('markets', []):
                                    market_name = market.get('name', '')
                                    
                                    for outcome in market.get('outcomes', []):
                                        bet_type = self._normalize_bet_type(market_name)
                                        
                                        odds.append(LiveOdds(
                                            bookmaker="GG.bet",
                                            match=f"{team1} vs {team2}",
                                            bet_type=bet_type,
                                            selection=outcome.get('name'),
                                            odds=float(outcome.get('odds', 0)),
                                            timestamp=datetime.now()
                                        ))
                                
                                break
        
        except Exception as e:
            self.logger.warning(f"ไม่สามารถดึงราคาจาก GG.bet: {e}")
        
        return odds
    
    async def _fetch_betway_odds(self, team1: str, team2: str) -> List[LiveOdds]:
        """ดึงราคาจาก Betway"""
        odds = []
        
        try:
            # Betway esports page
            url = "https://betway.com/en/sports/grp/esports/ctr/counter-strike"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # หา match elements
                    match_elements = soup.find_all('div', class_='eventRow')
                    
                    for element in match_elements:
                        match_text = element.get_text()
                        
                        if (team1.lower() in match_text.lower() and 
                            team2.lower() in match_text.lower()):
                            
                            # หา odds buttons
                            odds_buttons = element.find_all('button', class_='oddsButton')
                            
                            for i, button in enumerate(odds_buttons):
                                odds_value = button.get_text().strip()
                                
                                try:
                                    odds_float = float(odds_value)
                                    
                                    # กำหนด selection ตาม position
                                    if i == 0:
                                        selection = team1
                                        bet_type = "Match Winner"
                                    elif i == 1:
                                        selection = team2
                                        bet_type = "Match Winner"
                                    else:
                                        continue
                                    
                                    odds.append(LiveOdds(
                                        bookmaker="Betway",
                                        match=f"{team1} vs {team2}",
                                        bet_type=bet_type,
                                        selection=selection,
                                        odds=odds_float,
                                        timestamp=datetime.now()
                                    ))
                                
                                except ValueError:
                                    continue
                            
                            break
        
        except Exception as e:
            self.logger.warning(f"ไม่สามารถดึงราคาจาก Betway: {e}")
        
        return odds
    
    def _match_teams(self, participants: List, team1: str, team2: str) -> bool:
        """ตรวจสอบว่าแมตช์ตรงกับทีมที่ต้องการ"""
        if len(participants) < 2:
            return False
        
        team_names = [p.get('name', '').lower() for p in participants]
        
        return (team1.lower() in ' '.join(team_names) and 
                team2.lower() in ' '.join(team_names))
    
    def _normalize_bet_type(self, market_name: str) -> str:
        """แปลงชื่อ market เป็นรูปแบบมาตรฐาน"""
        market_lower = market_name.lower()
        
        if 'winner' in market_lower or 'moneyline' in market_lower:
            return "Match Winner"
        elif 'handicap' in market_lower or 'spread' in market_lower:
            return "Handicap"
        elif 'total' in market_lower or 'over/under' in market_lower:
            return "Total Maps"
        elif 'first map' in market_lower:
            return "First Map Winner"
        else:
            return market_name
    
    def _analyze_best_odds(self, all_odds: List[LiveOdds], team1: str, team2: str) -> List[BestOdds]:
        """วิเคราะห์ราคาดีที่สุดสำหรับแต่ละตลาด"""
        
        # จัดกลุ่มตาม bet_type และ selection
        grouped_odds = {}
        
        for odd in all_odds:
            key = f"{odd.bet_type}_{odd.selection}"
            
            if key not in grouped_odds:
                grouped_odds[key] = []
            
            grouped_odds[key].append(odd)
        
        best_odds_list = []
        
        for key, odds_group in grouped_odds.items():
            if not odds_group:
                continue
            
            # หาราคาดีที่สุด (สูงสุด)
            best_odd = max(odds_group, key=lambda x: x.odds)
            
            # คำนวณ Expected Value (ประมาณการ)
            if len(odds_group) > 0 and all(o.odds > 0 for o in odds_group):
                avg_odds = sum(o.odds for o in odds_group) / len(odds_group)
                expected_value = ((best_odd.odds - avg_odds) / avg_odds) * 100 if avg_odds > 0 else 0
            else:
                expected_value = 0
            
            # กำหนดระดับความเชื่อมั่นและความเสี่ยง
            confidence = "HIGH" if len(odds_group) >= 3 else "MEDIUM" if len(odds_group) == 2 else "LOW"
            
            if expected_value > 10:
                risk_level = "LOW"
            elif expected_value > 5:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            best_odds_list.append(BestOdds(
                bet_type=best_odd.bet_type,
                selection=best_odd.selection,
                best_odds=best_odd.odds,
                best_bookmaker=best_odd.bookmaker,
                all_odds=odds_group,
                expected_value=expected_value,
                confidence=confidence,
                risk_level=risk_level
            ))
        
        # เรียงตาม Expected Value
        best_odds_list.sort(key=lambda x: x.expected_value, reverse=True)
        
        return best_odds_list
    
    def _parse_pinnacle_odds(self, data: dict) -> List[LiveOdds]:
        """Parse ข้อมูลจาก Pinnacle API"""
        # Implementation สำหรับ Pinnacle
        pass
    
    def _parse_bet365_odds(self, html: str) -> List[LiveOdds]:
        """Parse ข้อมูลจาก Bet365"""
        # Implementation สำหรับ Bet365
        pass
    
    def _parse_betway_odds(self, html: str) -> List[LiveOdds]:
        """Parse ข้อมูลจาก Betway"""
        # Implementation สำหรับ Betway
        pass
    
    def _parse_ggbet_odds(self, data: dict) -> List[LiveOdds]:
        """Parse ข้อมูลจาก GG.bet API"""
        # Implementation สำหรับ GG.bet
        pass

# Singleton instance
_live_odds_fetcher = None

def get_live_odds_fetcher() -> LiveOddsFetcher:
    """ได้ instance ของ LiveOddsFetcher"""
    global _live_odds_fetcher
    if _live_odds_fetcher is None:
        _live_odds_fetcher = LiveOddsFetcher()
    return _live_odds_fetcher
