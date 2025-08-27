#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Odds API - ดึงราคาจริงจากเว็บพนันชั้นนำ (ไม่ใช่ Mock Up)
Created by KoJao - Professional CS2 Betting Analytics
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import re
from bs4 import BeautifulSoup
import time

class RealOddsAPI:
    """API สำหรับดึงราคาเดิมพันจริง"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # Headers แบบ real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    async def __aenter__(self):
        """เริ่มต้น session"""
        connector = aiohttp.TCPConnector(limit=10, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ปิด session"""
        if self.session:
            await self.session.close()
    
    async def get_real_odds(self) -> Dict:
        """ดึงราคาจริงจากหลายเว็บ"""
        
        print("🔄 กำลังดึงราคาจริงจากเว็บพนันชั้นนำ...")
        
        all_odds = {}
        
        # 1. ดึงจาก Pinnacle (มีความแม่นยำสูงสุด)
        print("📡 กำลังเชื่อมต่อ Pinnacle...")
        pinnacle_odds = await self._fetch_pinnacle_real()
        if pinnacle_odds:
            all_odds['pinnacle'] = pinnacle_odds
            print(f"✅ Pinnacle: พบ {len(pinnacle_odds)} ราคา")
        
        # 2. ดึงจาก Bet365 
        print("📡 กำลังเชื่อมต่อ Bet365...")
        bet365_odds = await self._fetch_bet365_real()
        if bet365_odds:
            all_odds['bet365'] = bet365_odds
            print(f"✅ Bet365: พบ {len(bet365_odds)} ราคา")
        
        # 3. ดึงจาก GG.bet (เชี่ยวชาญ esports)
        print("📡 กำลังเชื่อมต่อ GG.bet...")
        ggbet_odds = await self._fetch_ggbet_real()
        if ggbet_odds:
            all_odds['ggbet'] = ggbet_odds
            print(f"✅ GG.bet: พบ {len(ggbet_odds)} ราคา")
        
        # 4. ดึงจาก Betway
        print("📡 กำลังเชื่อมต่อ Betway...")
        betway_odds = await self._fetch_betway_real()
        if betway_odds:
            all_odds['betway'] = betway_odds
            print(f"✅ Betway: พบ {len(betway_odds)} ราคา")
        
        return all_odds
    
    async def _fetch_pinnacle_real(self) -> List[Dict]:
        """ดึงราคาจริงจาก Pinnacle"""
        odds = []
        
        try:
            # Pinnacle Guest API (สาธารณะ)
            url = "https://guest.api.arcadia.pinnacle.com/0.1/leagues/4281/matchups"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for match in data:
                        if 'participants' in match and len(match['participants']) >= 2:
                            team1 = match['participants'][0].get('name', '')
                            team2 = match['participants'][1].get('name', '')
                            
                            # ตรวจสอบแมตช์ BLAST
                            if self._is_blast_match(team1, team2):
                                match_name = f"{team1} vs {team2}"
                                
                                # Match Winner
                                if 'periods' in match:
                                    for period in match['periods']:
                                        if period.get('number') == 0:
                                            moneyline = period.get('moneyline', [])
                                            for ml in moneyline:
                                                odds.append({
                                                    'match': match_name,
                                                    'bet_type': 'Match Winner',
                                                    'selection': ml.get('designation'),
                                                    'odds': float(ml.get('price', 0)),
                                                    'bookmaker': 'Pinnacle',
                                                    'timestamp': datetime.now().isoformat()
                                                })
                                
                                # Handicap
                                if 'spreads' in match:
                                    for spread in match['spreads']:
                                        odds.append({
                                            'match': match_name,
                                            'bet_type': 'Handicap',
                                            'selection': f"{spread.get('designation')} {spread.get('hdp')}",
                                            'odds': float(spread.get('price', 0)),
                                            'bookmaker': 'Pinnacle',
                                            'timestamp': datetime.now().isoformat()
                                        })
                                
                                # Total Maps
                                if 'totals' in match:
                                    for total in match['totals']:
                                        odds.append({
                                            'match': match_name,
                                            'bet_type': 'Total Maps',
                                            'selection': f"{total.get('designation')} {total.get('points')}",
                                            'odds': float(total.get('price', 0)),
                                            'bookmaker': 'Pinnacle',
                                            'timestamp': datetime.now().isoformat()
                                        })
        
        except Exception as e:
            self.logger.warning(f"Pinnacle error: {e}")
        
        return odds
    
    async def _fetch_bet365_real(self) -> List[Dict]:
        """ดึงราคาจริงจาก Bet365"""
        odds = []
        
        try:
            # Bet365 esports page
            url = "https://www.bet365.com/#/AC/B1/C1/D1002/E174/F2/"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # หา CS:GO matches
                    match_elements = soup.find_all('div', class_=['gl-Market', 'gl-Market_General'])
                    
                    for element in match_elements:
                        match_text = element.get_text()
                        
                        # ตรวจสอบแมตช์ BLAST
                        if self._contains_blast_teams(match_text):
                            # หา odds
                            odds_elements = element.find_all('span', class_='gl-ParticipantOddsOnly_Odds')
                            
                            for i, odds_elem in enumerate(odds_elements):
                                try:
                                    odds_value = float(odds_elem.get_text().strip())
                                    
                                    odds.append({
                                        'match': self._extract_match_name(match_text),
                                        'bet_type': 'Match Winner',
                                        'selection': f"Team {i+1}",
                                        'odds': odds_value,
                                        'bookmaker': 'Bet365',
                                        'timestamp': datetime.now().isoformat()
                                    })
                                
                                except (ValueError, AttributeError):
                                    continue
        
        except Exception as e:
            self.logger.warning(f"Bet365 error: {e}")
        
        return odds
    
    async def _fetch_ggbet_real(self) -> List[Dict]:
        """ดึงราคาจริงจาก GG.bet"""
        odds = []
        
        try:
            # GG.bet API endpoint
            url = "https://gg.bet/api/sport/line"
            params = {
                'sport': 'counter-strike',
                'limit': 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'events' in data:
                        for event in data['events']:
                            event_name = event.get('name', '')
                            
                            # ตรวจสอบแมตช์ BLAST
                            if self._contains_blast_teams(event_name):
                                
                                for market in event.get('markets', []):
                                    market_name = market.get('name', '')
                                    
                                    for outcome in market.get('outcomes', []):
                                        odds.append({
                                            'match': event_name,
                                            'bet_type': self._normalize_market(market_name),
                                            'selection': outcome.get('name'),
                                            'odds': float(outcome.get('odds', 0)),
                                            'bookmaker': 'GG.bet',
                                            'timestamp': datetime.now().isoformat()
                                        })
        
        except Exception as e:
            self.logger.warning(f"GG.bet error: {e}")
        
        return odds
    
    async def _fetch_betway_real(self) -> List[Dict]:
        """ดึงราคาจริงจาก Betway"""
        odds = []
        
        try:
            # Betway esports API
            url = "https://sports.betway.com/api/v2/sports/esports/events"
            params = {
                'game': 'counter-strike',
                'limit': 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'events' in data:
                        for event in data['events']:
                            event_name = event.get('name', '')
                            
                            # ตรวจสอบแมตช์ BLAST
                            if self._contains_blast_teams(event_name):
                                
                                for market in event.get('markets', []):
                                    for selection in market.get('selections', []):
                                        odds.append({
                                            'match': event_name,
                                            'bet_type': market.get('name', ''),
                                            'selection': selection.get('name'),
                                            'odds': float(selection.get('odds', 0)),
                                            'bookmaker': 'Betway',
                                            'timestamp': datetime.now().isoformat()
                                        })
        
        except Exception as e:
            self.logger.warning(f"Betway error: {e}")
        
        return odds
    
    def _is_blast_match(self, team1: str, team2: str) -> bool:
        """ตรวจสอบว่าเป็นแมตช์ BLAST"""
        blast_teams = [
            'virtus.pro', 'gamerlegion', 'ecstatic', 'faze', 'navi', 'fnatic',
            'vitality', 'm80', 'liquid', 'g2', 'astralis', 'heroic'
        ]
        
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        
        return any(team in team1_lower for team in blast_teams) and \
               any(team in team2_lower for team in blast_teams)
    
    def _contains_blast_teams(self, text: str) -> bool:
        """ตรวจสอบว่าข้อความมีชื่อทีม BLAST"""
        blast_teams = [
            'virtus.pro', 'gamerlegion', 'ecstatic', 'faze', 'navi', 'fnatic',
            'vitality', 'm80', 'liquid', 'g2', 'astralis', 'heroic'
        ]
        
        text_lower = text.lower()
        return any(team in text_lower for team in blast_teams)
    
    def _extract_match_name(self, text: str) -> str:
        """แยกชื่อแมตช์จากข้อความ"""
        # ลบข้อความที่ไม่จำเป็น
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', text)
        words = cleaned.split()
        
        # หาชื่อทีม
        blast_teams = [
            'Virtus.pro', 'GamerLegion', 'ECSTATIC', 'FaZe', 'NAVI', 'Fnatic',
            'Vitality', 'M80', 'Liquid', 'G2', 'Astralis', 'Heroic'
        ]
        
        found_teams = []
        for word in words:
            for team in blast_teams:
                if team.lower() in word.lower():
                    found_teams.append(team)
                    break
        
        if len(found_teams) >= 2:
            return f"{found_teams[0]} vs {found_teams[1]}"
        
        return text[:50]  # fallback
    
    def _normalize_market(self, market_name: str) -> str:
        """แปลงชื่อตลาดเป็นมาตรฐาน"""
        market_lower = market_name.lower()
        
        if 'winner' in market_lower or 'moneyline' in market_lower:
            return 'Match Winner'
        elif 'handicap' in market_lower or 'spread' in market_lower:
            return 'Handicap'
        elif 'total' in market_lower or 'over/under' in market_lower:
            return 'Total Maps'
        elif 'first map' in market_lower:
            return 'First Map'
        else:
            return market_name

def analyze_real_odds(all_odds: Dict) -> Dict:
    """วิเคราะห์ราคาจริงที่ดึงมา"""
    
    analysis = {
        'total_odds_found': 0,
        'bookmakers_available': [],
        'matches_found': set(),
        'best_values': [],
        'summary': {}
    }
    
    all_odds_list = []
    
    # รวมราคาจากทุกเว็บ
    for bookmaker, odds_list in all_odds.items():
        analysis['bookmakers_available'].append(bookmaker)
        analysis['total_odds_found'] += len(odds_list)
        
        for odd in odds_list:
            analysis['matches_found'].add(odd['match'])
            all_odds_list.append(odd)
    
    # หาราคาดีที่สุดสำหรับแต่ละตลาด
    market_groups = {}
    
    for odd in all_odds_list:
        key = f"{odd['match']}_{odd['bet_type']}_{odd['selection']}"
        
        if key not in market_groups:
            market_groups[key] = []
        
        market_groups[key].append(odd)
    
    # วิเคราะห์ value
    for key, odds_group in market_groups.items():
        if len(odds_group) >= 2:  # ต้องมีอย่างน้อย 2 เว็บ
            best_odd = max(odds_group, key=lambda x: x['odds'])
            avg_odds = sum(o['odds'] for o in odds_group) / len(odds_group)
            
            if avg_odds > 0:
                value_percent = ((best_odd['odds'] - avg_odds) / avg_odds) * 100
                
                if value_percent > 3:  # มี value มากกว่า 3%
                    analysis['best_values'].append({
                        'match': best_odd['match'],
                        'bet_type': best_odd['bet_type'],
                        'selection': best_odd['selection'],
                        'best_odds': best_odd['odds'],
                        'best_bookmaker': best_odd['bookmaker'],
                        'average_odds': avg_odds,
                        'value_percentage': value_percent,
                        'num_bookmakers': len(odds_group)
                    })
    
    # เรียงตาม value
    analysis['best_values'].sort(key=lambda x: x['value_percentage'], reverse=True)
    
    # สรุป
    analysis['summary'] = {
        'matches_found': list(analysis['matches_found']),
        'total_bookmakers': len(analysis['bookmakers_available']),
        'best_value_count': len(analysis['best_values']),
        'timestamp': datetime.now().isoformat()
    }
    
    return analysis

# Singleton instance
_real_odds_api = None

def get_real_odds_api() -> RealOddsAPI:
    """ได้ instance ของ RealOddsAPI"""
    global _real_odds_api
    if _real_odds_api is None:
        _real_odds_api = RealOddsAPI()
    return _real_odds_api
