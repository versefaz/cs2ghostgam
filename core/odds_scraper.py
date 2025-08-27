#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Source Odds Scraper for CS2 Betting
ระบบดึงราคาต่อรองจากเว็บพนันหลายแห่ง
"""

import sys
import asyncio
import aiohttp
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import re
import time
from urllib.parse import urljoin, quote

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except AttributeError:
        # Already detached or not available
        pass

@dataclass
class BookmakerOdds:
    """ราคาต่อรองจากเว็บพนัน"""
    bookmaker: str
    match_id: str
    team1: str
    team2: str
    team1_odds: float
    team2_odds: float
    markets: Dict[str, Any]
    timestamp: datetime
    url: str = ""
    confidence: float = 1.0

@dataclass
class OddsComparison:
    """เปรียบเทียบราคาต่อรอง"""
    match_id: str
    team1: str
    team2: str
    best_team1_odds: BookmakerOdds
    best_team2_odds: BookmakerOdds
    average_team1_odds: float
    average_team2_odds: float
    arbitrage_opportunity: bool
    arbitrage_percentage: float
    value_bets: List[Dict[str, Any]]

class MultiSourceOddsScraper:
    """ระบบดึงราคาต่อรองจากหลายแหล่ง"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # รายการเว็บพนันที่รองรับ
        self.bookmakers = {
            "pinnacle": {
                "name": "Pinnacle",
                "base_url": "https://www.pinnacle.com",
                "api_endpoint": "/api/v1/odds",
                "priority": 1,
                "reliability": 0.95
            },
            "bet365": {
                "name": "Bet365", 
                "base_url": "https://www.bet365.com",
                "api_endpoint": "/api/odds",
                "priority": 2,
                "reliability": 0.90
            },
            "ggbet": {
                "name": "GG.BET",
                "base_url": "https://gg.bet",
                "api_endpoint": "/api/esports/odds",
                "priority": 3,
                "reliability": 0.85
            },
            "unikrn": {
                "name": "Unikrn",
                "base_url": "https://unikrn.com",
                "api_endpoint": "/api/betting/odds",
                "priority": 4,
                "reliability": 0.80
            },
            "rivalry": {
                "name": "Rivalry",
                "base_url": "https://www.rivalry.com",
                "api_endpoint": "/api/odds",
                "priority": 5,
                "reliability": 0.85
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def scrape_all_odds(self, matches: List[Dict[str, Any]]) -> Dict[str, List[BookmakerOdds]]:
        """ดึงราคาต่อรองจากทุกเว็บพนัน"""
        
        print("💰 เริ่มดึงราคาต่อรองจากเว็บพนันหลายแห่ง...")
        
        all_odds = {}
        
        for match in matches:
            match_id = match.get("match_id", f"match_{len(all_odds)}")
            team1 = match.get("team1", "Team1")
            team2 = match.get("team2", "Team2")
            
            print(f"🎯 กำลังดึงราคา: {team1} vs {team2}")
            
            match_odds = []
            
            # ดึงจากแต่ละเว็บพนัน
            for bookmaker_key, bookmaker_info in self.bookmakers.items():
                try:
                    odds = await self._scrape_bookmaker_odds(
                        bookmaker_key, bookmaker_info, match
                    )
                    if odds:
                        match_odds.append(odds)
                        print(f"  ✅ {bookmaker_info['name']}: {odds.team1_odds:.2f} / {odds.team2_odds:.2f}")
                    else:
                        print(f"  ❌ {bookmaker_info['name']}: ไม่พบข้อมูล")
                        
                except Exception as e:
                    print(f"  ⚠️ {bookmaker_info['name']}: {str(e)}")
                
                # หน่วงเวลาเพื่อไม่ให้ถูก rate limit
                await asyncio.sleep(1)
            
            if match_odds:
                all_odds[match_id] = match_odds
        
        return all_odds

    async def _scrape_bookmaker_odds(self, bookmaker_key: str, 
                                   bookmaker_info: Dict, 
                                   match: Dict) -> Optional[BookmakerOdds]:
        """ดึงราคาจากเว็บพนันแต่ละแห่ง"""
        
        # สำหรับ demo ใช้การจำลองข้อมูล
        # ในการใช้งานจริงจะต้องใช้ API หรือ web scraping
        return await self._simulate_odds_scraping(bookmaker_key, bookmaker_info, match)

    async def _simulate_odds_scraping(self, bookmaker_key: str,
                                    bookmaker_info: Dict,
                                    match: Dict) -> Optional[BookmakerOdds]:
        """จำลองการดึงราคาต่อรอง (แทนที่ด้วยการ scrape จริง)"""
        
        import random
        
        # จำลองความน่าจะเป็นที่จะหาข้อมูลได้
        if random.random() > bookmaker_info["reliability"]:
            return None
        
        team1 = match.get("team1", "Team1")
        team2 = match.get("team2", "Team2")
        match_id = match.get("match_id", "unknown")
        
        # จำลองราคาต่อรองที่สมจริง
        base_prob = random.uniform(0.4, 0.6)  # ความน่าจะเป็นทีม 1 ชนะ
        
        # คำนวณราคาจากความน่าจะเป็น
        team1_odds = round(1 / base_prob, 2)
        team2_odds = round(1 / (1 - base_prob), 2)
        
        # เพิ่ม margin ของเว็บพนัน (5-10%)
        margin = random.uniform(0.05, 0.10)
        team1_odds = round(team1_odds * (1 - margin), 2)
        team2_odds = round(team2_odds * (1 - margin), 2)
        
        # สร้างตลาดเสริม
        markets = self._generate_additional_markets(team1_odds, team2_odds)
        
        return BookmakerOdds(
            bookmaker=bookmaker_info["name"],
            match_id=match_id,
            team1=team1,
            team2=team2,
            team1_odds=team1_odds,
            team2_odds=team2_odds,
            markets=markets,
            timestamp=datetime.now(),
            url=f"{bookmaker_info['base_url']}/esports/cs2",
            confidence=bookmaker_info["reliability"]
        )

    def _generate_additional_markets(self, team1_odds: float, team2_odds: float) -> Dict[str, Any]:
        """สร้างตลาดเดิมพันเสริม"""
        
        import random
        
        markets = {
            # Handicap markets
            "handicap": {
                f"team1_+1.5": round(team1_odds * random.uniform(0.6, 0.8), 2),
                f"team1_-1.5": round(team1_odds * random.uniform(1.8, 2.5), 2),
                f"team2_+1.5": round(team2_odds * random.uniform(0.6, 0.8), 2),
                f"team2_-1.5": round(team2_odds * random.uniform(1.8, 2.5), 2),
            },
            
            # Total maps
            "total_maps": {
                "over_2.5": round(random.uniform(1.7, 2.2), 2),
                "under_2.5": round(random.uniform(1.6, 2.1), 2),
                "exactly_2": round(random.uniform(2.8, 3.5), 2),
                "exactly_3": round(random.uniform(2.2, 2.8), 2),
            },
            
            # First map winner
            "first_map": {
                "team1": round(team1_odds * random.uniform(0.9, 1.1), 2),
                "team2": round(team2_odds * random.uniform(0.9, 1.1), 2),
            },
            
            # Correct score
            "correct_score": {
                "2-0": round(team1_odds * random.uniform(2.5, 3.5), 2),
                "2-1": round(random.uniform(2.8, 3.8), 2),
                "0-2": round(team2_odds * random.uniform(2.5, 3.5), 2),
                "1-2": round(random.uniform(2.8, 3.8), 2),
            }
        }
        
        return markets

    async def compare_odds(self, all_odds: Dict[str, List[BookmakerOdds]]) -> Dict[str, OddsComparison]:
        """เปรียบเทียบราคาต่อรองและหาโอกาสที่ดี"""
        
        print("\n📊 เปรียบเทียบราคาต่อรองและหาโอกาสการเดิมพัน...")
        
        comparisons = {}
        
        for match_id, odds_list in all_odds.items():
            if not odds_list:
                continue
            
            # หาราคาที่ดีที่สุดสำหรับแต่ละทีม
            best_team1 = max(odds_list, key=lambda x: x.team1_odds)
            best_team2 = max(odds_list, key=lambda x: x.team2_odds)
            
            # คำนวณราคาเฉลี่ย
            avg_team1 = sum(odds.team1_odds for odds in odds_list) / len(odds_list)
            avg_team2 = sum(odds.team2_odds for odds in odds_list) / len(odds_list)
            
            # ตรวจสอบโอกาส arbitrage
            arbitrage_opportunity, arbitrage_percentage = self._check_arbitrage(
                best_team1.team1_odds, best_team2.team2_odds
            )
            
            # หา value bets
            value_bets = self._find_value_bets(odds_list, avg_team1, avg_team2)
            
            comparison = OddsComparison(
                match_id=match_id,
                team1=odds_list[0].team1,
                team2=odds_list[0].team2,
                best_team1_odds=best_team1,
                best_team2_odds=best_team2,
                average_team1_odds=round(avg_team1, 2),
                average_team2_odds=round(avg_team2, 2),
                arbitrage_opportunity=arbitrage_opportunity,
                arbitrage_percentage=arbitrage_percentage,
                value_bets=value_bets
            )
            
            comparisons[match_id] = comparison
        
        return comparisons

    def _check_arbitrage(self, odds1: float, odds2: float) -> Tuple[bool, float]:
        """ตรวจสอบโอกาส arbitrage"""
        
        # คำนวณ implied probability
        implied_prob = (1 / odds1) + (1 / odds2)
        
        # ถ้า implied probability < 1 แสดงว่ามีโอกาส arbitrage
        if implied_prob < 1.0:
            arbitrage_percentage = ((1 - implied_prob) / implied_prob) * 100
            return True, round(arbitrage_percentage, 2)
        
        return False, 0.0

    def _find_value_bets(self, odds_list: List[BookmakerOdds], 
                        avg_team1: float, avg_team2: float) -> List[Dict[str, Any]]:
        """หา value bets (ราคาที่สูงกว่าค่าเฉลี่ย)"""
        
        value_bets = []
        threshold = 0.05  # 5% เหนือค่าเฉลี่ย
        
        for odds in odds_list:
            # ตรวจสอบทีม 1
            if odds.team1_odds > avg_team1 * (1 + threshold):
                value_percentage = ((odds.team1_odds - avg_team1) / avg_team1) * 100
                value_bets.append({
                    "bookmaker": odds.bookmaker,
                    "selection": odds.team1,
                    "odds": odds.team1_odds,
                    "average_odds": round(avg_team1, 2),
                    "value_percentage": round(value_percentage, 1),
                    "type": "match_winner"
                })
            
            # ตรวจสอบทีม 2
            if odds.team2_odds > avg_team2 * (1 + threshold):
                value_percentage = ((odds.team2_odds - avg_team2) / avg_team2) * 100
                value_bets.append({
                    "bookmaker": odds.bookmaker,
                    "selection": odds.team2,
                    "odds": odds.team2_odds,
                    "average_odds": round(avg_team2, 2),
                    "value_percentage": round(value_percentage, 1),
                    "type": "match_winner"
                })
        
        return value_bets

    async def display_odds_analysis(self, comparisons: Dict[str, OddsComparison]):
        """แสดงผลการวิเคราะห์ราคาต่อรอง"""
        
        print("\n" + "=" * 80)
        print("💰 การวิเคราะห์ราคาต่อรองและโอกาสการเดิมพัน")
        print("=" * 80)
        
        for match_id, comparison in comparisons.items():
            print(f"\n🎯 {comparison.team1} vs {comparison.team2}")
            print("-" * 50)
            
            # แสดงราคาที่ดีที่สุด
            print("🏆 ราคาที่ดีที่สุด:")
            print(f"  {comparison.team1}: {comparison.best_team1_odds.team1_odds:.2f} ({comparison.best_team1_odds.bookmaker})")
            print(f"  {comparison.team2}: {comparison.best_team2_odds.team2_odds:.2f} ({comparison.best_team2_odds.bookmaker})")
            
            # แสดงราคาเฉลี่ย
            print(f"\n📊 ราคาเฉลี่ย:")
            print(f"  {comparison.team1}: {comparison.average_team1_odds:.2f}")
            print(f"  {comparison.team2}: {comparison.average_team2_odds:.2f}")
            
            # แสดงโอกาส arbitrage
            if comparison.arbitrage_opportunity:
                print(f"\n🎰 โอกาส Arbitrage: {comparison.arbitrage_percentage:.2f}%")
                print("  💡 สามารถเดิมพันทั้งสองฝ่ายและรับประกันกำไร!")
            
            # แสดง value bets
            if comparison.value_bets:
                print(f"\n💎 Value Bets พบ {len(comparison.value_bets)} โอกาส:")
                for value_bet in comparison.value_bets[:3]:  # แสดง 3 อันดับแรก
                    print(f"  • {value_bet['selection']}: {value_bet['odds']:.2f} @ {value_bet['bookmaker']}")
                    print(f"    (สูงกว่าค่าเฉลี่ย {value_bet['value_percentage']:.1f}%)")

    async def save_odds_data(self, all_odds: Dict[str, List[BookmakerOdds]], 
                           comparisons: Dict[str, OddsComparison]):
        """บันทึกข้อมูลราคาต่อรอง"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # บันทึกข้อมูลราคาดิบ
        odds_file = Path("data") / f"odds_data_{timestamp}.json"
        odds_file.parent.mkdir(exist_ok=True)
        
        odds_data = {}
        for match_id, odds_list in all_odds.items():
            odds_data[match_id] = [asdict(odds) for odds in odds_list]
        
        with open(odds_file, 'w', encoding='utf-8') as f:
            json.dump(odds_data, f, ensure_ascii=False, indent=2, default=str)
        
        # บันทึกการเปรียบเทียบ
        comparison_file = Path("data") / f"odds_comparison_{timestamp}.json"
        
        comparison_data = {}
        for match_id, comparison in comparisons.items():
            comparison_data[match_id] = asdict(comparison)
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 บันทึกข้อมูลแล้ว:")
        print(f"  📄 ราคาต่อรอง: {odds_file}")
        print(f"  📊 การเปรียบเทียบ: {comparison_file}")

# ฟังก์ชันสำหรับทดสอบ
async def test_odds_scraper():
    """ทดสอบระบบดึงราคาต่อรอง"""
    
    # ข้อมูลแมตช์ตัวอย่าง
    sample_matches = [
        {
            "match_id": "test_1",
            "team1": "GamerLegion",
            "team2": "Virtus.pro",
            "event": "BLAST Open London 2025"
        },
        {
            "match_id": "test_2", 
            "team1": "Natus Vincere",
            "team2": "FaZe Clan",
            "event": "IEM Katowice 2025"
        }
    ]
    
    async with MultiSourceOddsScraper() as scraper:
        # ดึงราคาต่อรอง
        all_odds = await scraper.scrape_all_odds(sample_matches)
        
        # เปรียบเทียบราคา
        comparisons = await scraper.compare_odds(all_odds)
        
        # แสดงผล
        await scraper.display_odds_analysis(comparisons)
        
        # บันทึกข้อมูล
        await scraper.save_odds_data(all_odds, comparisons)

if __name__ == "__main__":
    asyncio.run(test_odds_scraper())
