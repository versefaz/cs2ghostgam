#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated CS2 Betting Intelligence Pipeline
ระบบวิเคราะห์และแนะนำการเดิมพัน CS2 แบบอัตโนมัติ
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except AttributeError:
        # Already detached or not available
        pass

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent))

from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper
from core.universal_match_analyzer import UniversalMatchAnalyzer, AnalysisDepth
from core.enhanced_team_analyzer import EnhancedTeamAnalyzer
from core.deep_betting_analyzer import DeepBettingAnalyzer

@dataclass
class MatchInfo:
    """ข้อมูลแมตช์"""
    team1: str
    team2: str
    time: str
    event: str
    importance: float
    tournament_tier: str
    match_id: str

@dataclass
class OddsData:
    """ข้อมูลราคาต่อรอง"""
    bookmaker: str
    team1_odds: float
    team2_odds: float
    markets: Dict[str, Any]
    timestamp: datetime

@dataclass
class BettingRecommendation:
    """คำแนะนำการเดิมพัน"""
    match_id: str
    primary_bet: Dict[str, Any]
    backup_bets: List[Dict[str, Any]]
    confidence_level: float
    reasoning: List[str]
    risk_level: str
    kelly_percentage: float
    expected_value: float

class AutomatedBettingPipeline:
    """ระบบวิเคราะห์การเดิมพัน CS2 แบบอัตโนมัติ"""
    
    def __init__(self):
        self.match_analyzer = UniversalMatchAnalyzer()
        self.team_analyzer = EnhancedTeamAnalyzer()
        self.betting_analyzer = DeepBettingAnalyzer()
        
        # ทัวร์นาเมนท์ที่สำคัญ
        self.major_tournaments = {
            "BLAST": {"tier": "S", "importance": 0.9},
            "IEM": {"tier": "S", "importance": 0.9},
            "ESL Pro League": {"tier": "A", "importance": 0.8},
            "EPICENTER": {"tier": "A", "importance": 0.8},
            "DreamHack": {"tier": "B", "importance": 0.7},
            "Flashpoint": {"tier": "B", "importance": 0.7},
            "WePlay": {"tier": "B", "importance": 0.6}
        }
        
        # เว็บพนันที่ติดตาม
        self.bookmakers = [
            "Pinnacle", "Bet365", "1xBet", "GG.BET", 
            "Unikrn", "Rivalry", "Betway", "888sport"
        ]

    async def run_full_pipeline(self) -> List[BettingRecommendation]:
        """รันระบบวิเคราะห์แบบเต็มรูปแบบ"""
        
        print("🚀 เริ่มต้น Automated CS2 Betting Intelligence Pipeline")
        print("=" * 80)
        
        # Step 1: ดึงแมตช์จาก HLTV
        matches = await self._scrape_upcoming_matches()
        print(f"📊 พบแมตช์ทั้งหมด: {len(matches)} คู่")
        
        # Step 2: กรองแมตช์ที่สำคัญ
        important_matches = self._filter_important_matches(matches)
        print(f"🎯 แมตช์สำคัญ: {len(important_matches)} คู่")
        
        # Step 3: ดึงราคาต่อรองสำหรับแต่ละแมตช์
        odds_data = await self._scrape_odds_for_matches(important_matches)
        print(f"💰 ดึงราคาต่อรองสำเร็จ: {len(odds_data)} แมตช์")
        
        # Step 4: วิเคราะห์แต่ละแมตช์เชิงลึก
        recommendations = []
        for match in important_matches:
            print(f"\n🔬 กำลังวิเคราะห์: {match.team1} vs {match.team2}")
            
            # วิเคราะห์แมตช์
            analysis = await self.match_analyzer.analyze_any_match(
                match.team1, match.team2, AnalysisDepth.WORLD_CLASS
            )
            
            # สร้างคำแนะนำการเดิมพัน
            recommendation = await self._generate_betting_recommendation(
                match, analysis, odds_data.get(match.match_id, [])
            )
            
            if recommendation:
                recommendations.append(recommendation)
        
        # Step 5: จัดอันดับและแสดงผล
        await self._display_final_recommendations(recommendations)
        
        return recommendations

    async def _scrape_upcoming_matches(self) -> List[MatchInfo]:
        """ดึงแมตช์ที่กำลังจะมาถึงจาก HLTV"""
        
        print("📡 กำลังดึงข้อมูลแมตช์จาก HLTV...")
        
        try:
            async with EnhancedHLTVScraper() as scraper:
                raw_matches = await scraper.get_upcoming_matches(limit=50)
                
                matches = []
                for i, match in enumerate(raw_matches):
                    match_info = MatchInfo(
                        team1=match.get("team1", "TBD"),
                        team2=match.get("team2", "TBD"),
                        time=match.get("time", "TBD"),
                        event=match.get("event", "Unknown"),
                        importance=self._calculate_match_importance(match.get("event", "")),
                        tournament_tier=self._get_tournament_tier(match.get("event", "")),
                        match_id=f"match_{i}_{datetime.now().strftime('%Y%m%d')}"
                    )
                    matches.append(match_info)
                
                return matches
                
        except Exception as e:
            print(f"⚠️ ข้อผิดพลาดในการดึงข้อมูล: {e}")
            return self._get_sample_matches()

    def _filter_important_matches(self, matches: List[MatchInfo]) -> List[MatchInfo]:
        """กรองแมตช์ที่สำคัญ"""
        
        # กรองตามความสำคัญและเวลา
        important_matches = []
        
        for match in matches:
            # เช็คว่าเป็นทัวร์นาเมนท์สำคัญ
            if match.importance >= 0.6:
                # เช็คว่าไม่ใช่ TBD
                if match.team1 != "TBD" and match.team2 != "TBD":
                    # เช็คว่าเป็นแมตช์ในอนาคตใกล้
                    important_matches.append(match)
        
        # จำกัดจำนวนแมตช์ที่จะวิเคราะห์ (เพื่อประสิทธิภาพ)
        return important_matches[:10]

    async def _scrape_odds_for_matches(self, matches: List[MatchInfo]) -> Dict[str, List[OddsData]]:
        """ดึงราคาต่อรองสำหรับแต่ละแมตช์"""
        
        print("💰 กำลังดึงราคาต่อรองจากเว็บพนัน...")
        
        odds_data = {}
        
        for match in matches:
            match_odds = []
            
            # จำลองการดึงราคาจากเว็บพนันต่างๆ
            for bookmaker in self.bookmakers[:4]:  # จำกัดจำนวนเพื่อความเร็ว
                odds = self._simulate_odds_scraping(match, bookmaker)
                if odds:
                    match_odds.append(odds)
            
            if match_odds:
                odds_data[match.match_id] = match_odds
        
        return odds_data

    async def _generate_betting_recommendation(self, match: MatchInfo, 
                                             analysis: Any, 
                                             odds: List[OddsData]) -> Optional[BettingRecommendation]:
        """สร้างคำแนะนำการเดิมพันจากการวิเคราะห์"""
        
        if not odds:
            return None
        
        # หาราคาที่ดีที่สุด
        best_odds = self._find_best_odds(odds)
        
        # คำนวณ Expected Value
        expected_value = self._calculate_expected_value(analysis, best_odds)
        
        # สร้างคำแนะนำหลัก
        primary_bet = self._create_primary_bet(match, analysis, best_odds, expected_value)
        
        # สร้างตัวเลือกสำรอง
        backup_bets = self._create_backup_bets(match, analysis, odds)
        
        # ประเมินความเสี่ยง
        risk_level = self._assess_risk_level(analysis, match.importance)
        
        # คำนวณ Kelly Criterion
        kelly_percentage = self._calculate_kelly_criterion(expected_value, best_odds)
        
        # สร้างเหตุผล
        reasoning = self._generate_reasoning(match, analysis, primary_bet)
        
        return BettingRecommendation(
            match_id=match.match_id,
            primary_bet=primary_bet,
            backup_bets=backup_bets,
            confidence_level=analysis.prediction_confidence,
            reasoning=reasoning,
            risk_level=risk_level,
            kelly_percentage=kelly_percentage,
            expected_value=expected_value
        )

    async def _display_final_recommendations(self, recommendations: List[BettingRecommendation]):
        """แสดงคำแนะนำการเดิมพันขั้นสุดท้าย"""
        
        print("\n" + "=" * 80)
        print("🎯 คำแนะนำการเดิมพัน CS2 - ระดับโลก")
        print("=" * 80)
        
        # จัดเรียงตาม Expected Value
        sorted_recommendations = sorted(
            recommendations, 
            key=lambda x: x.expected_value, 
            reverse=True
        )
        
        for i, rec in enumerate(sorted_recommendations[:5], 1):  # แสดง 5 อันดับแรก
            print(f"\n🏆 อันดับ {i}")
            print(f"📊 แมตช์: {rec.primary_bet.get('match', 'Unknown')}")
            print(f"🎯 เดิมพันหลัก: {rec.primary_bet.get('type', 'Unknown')} - {rec.primary_bet.get('selection', 'Unknown')}")
            print(f"💰 ราคา: {rec.primary_bet.get('odds', 0):.2f}")
            print(f"📈 Expected Value: {rec.expected_value:.3f}")
            print(f"🎲 Kelly %: {rec.kelly_percentage:.2f}%")
            
            # แสดงความเสี่ยง
            risk_emoji = "🔴" if rec.risk_level == "high" else "🟡" if rec.risk_level == "medium" else "🟢"
            print(f"{risk_emoji} ความเสี่ยง: {rec.risk_level}")
            
            # แสดงเหตุผล
            print("💡 เหตุผล:")
            for reason in rec.reasoning[:3]:  # แสดง 3 เหตุผลแรก
                print(f"   • {reason}")
            
            # แสดงตัวเลือกสำรอง
            if rec.backup_bets:
                print("🔄 ตัวเลือกสำรอง:")
                for backup in rec.backup_bets[:2]:  # แสดง 2 ตัวเลือกแรก
                    print(f"   • {backup.get('type', 'Unknown')}: {backup.get('selection', 'Unknown')} @ {backup.get('odds', 0):.2f}")

    # Helper methods
    def _calculate_match_importance(self, event_name: str) -> float:
        """คำนวณความสำคัญของแมตช์"""
        for tournament, info in self.major_tournaments.items():
            if tournament.lower() in event_name.lower():
                return info["importance"]
        return 0.5  # ค่าเริ่มต้น

    def _get_tournament_tier(self, event_name: str) -> str:
        """ระบุระดับทัวร์นาเมนท์"""
        for tournament, info in self.major_tournaments.items():
            if tournament.lower() in event_name.lower():
                return info["tier"]
        return "C"

    def _simulate_odds_scraping(self, match: MatchInfo, bookmaker: str) -> Optional[OddsData]:
        """จำลองการดึงราคาต่อรอง (จะแทนที่ด้วยการ scrape จริง)"""
        import random
        
        # จำลองราคาต่อรอง
        base_odds = random.uniform(1.5, 3.0)
        team1_odds = round(base_odds, 2)
        team2_odds = round(4.0 - base_odds + random.uniform(-0.3, 0.3), 2)
        
        return OddsData(
            bookmaker=bookmaker,
            team1_odds=team1_odds,
            team2_odds=team2_odds,
            markets={
                "match_winner": {"team1": team1_odds, "team2": team2_odds},
                "handicap": {"team1_+1.5": team1_odds * 0.6, "team2_-1.5": team2_odds * 1.4},
                "total_maps": {"over_2.5": 1.8, "under_2.5": 2.0}
            },
            timestamp=datetime.now()
        )

    def _find_best_odds(self, odds: List[OddsData]) -> Dict[str, Any]:
        """หาราคาที่ดีที่สุด"""
        if not odds:
            return {}
        
        best_team1 = max(odds, key=lambda x: x.team1_odds)
        best_team2 = max(odds, key=lambda x: x.team2_odds)
        
        return {
            "team1": {"odds": best_team1.team1_odds, "bookmaker": best_team1.bookmaker},
            "team2": {"odds": best_team2.team2_odds, "bookmaker": best_team2.bookmaker}
        }

    def _calculate_expected_value(self, analysis: Any, best_odds: Dict) -> float:
        """คำนวณ Expected Value"""
        # ใช้ confidence จากการวิเคราะห์เป็นความน่าจะเป็น
        confidence = analysis.prediction_confidence
        
        # สมมติว่าทีมที่แนะนำมีโอกาสชนะตาม confidence
        recommended_team = analysis.betting_recommendations.get("recommended_team", "")
        
        if "team1" in recommended_team.lower() or recommended_team in best_odds.get("team1", {}).get("bookmaker", ""):
            prob_win = confidence
            odds = best_odds.get("team1", {}).get("odds", 2.0)
        else:
            prob_win = confidence
            odds = best_odds.get("team2", {}).get("odds", 2.0)
        
        # EV = (probability × odds) - 1
        return (prob_win * odds) - 1

    def _calculate_kelly_criterion(self, expected_value: float, best_odds: Dict) -> float:
        """คำนวณ Kelly Criterion"""
        if expected_value <= 0:
            return 0
        
        # Kelly % = (bp - q) / b
        # โดยที่ b = odds - 1, p = probability, q = 1 - p
        avg_odds = (best_odds.get("team1", {}).get("odds", 2.0) + 
                   best_odds.get("team2", {}).get("odds", 2.0)) / 2
        
        b = avg_odds - 1
        p = 0.6  # สมมติความน่าจะเป็น
        q = 1 - p
        
        kelly = ((b * p) - q) / b
        return max(0, min(kelly * 100, 5))  # จำกัดไม่เกิน 5%

    def _create_primary_bet(self, match: MatchInfo, analysis: Any, 
                          best_odds: Dict, expected_value: float) -> Dict[str, Any]:
        """สร้างคำแนะนำการเดิมพันหลัก"""
        
        recommended_team = analysis.betting_recommendations.get("recommended_team", "")
        
        if expected_value > 0.1:  # มี EV ที่ดี
            if "team1" in recommended_team.lower() or match.team1 in recommended_team:
                return {
                    "match": f"{match.team1} vs {match.team2}",
                    "type": "Match Winner",
                    "selection": match.team1,
                    "odds": best_odds.get("team1", {}).get("odds", 2.0),
                    "bookmaker": best_odds.get("team1", {}).get("bookmaker", "Unknown")
                }
            else:
                return {
                    "match": f"{match.team1} vs {match.team2}",
                    "type": "Match Winner", 
                    "selection": match.team2,
                    "odds": best_odds.get("team2", {}).get("odds", 2.0),
                    "bookmaker": best_odds.get("team2", {}).get("bookmaker", "Unknown")
                }
        else:
            # ถ้า EV ไม่ดี แนะนำตลาดรอง
            return {
                "match": f"{match.team1} vs {match.team2}",
                "type": "Total Maps",
                "selection": "Over 2.5",
                "odds": 1.85,
                "bookmaker": "Average"
            }

    def _create_backup_bets(self, match: MatchInfo, analysis: Any, 
                          odds: List[OddsData]) -> List[Dict[str, Any]]:
        """สร้างตัวเลือกการเดิมพันสำรอง"""
        
        backups = []
        
        # Handicap bet
        backups.append({
            "type": "Handicap",
            "selection": f"{match.team1} +1.5",
            "odds": 1.6,
            "reason": "ความปลอดภัยสูง"
        })
        
        # Total maps
        backups.append({
            "type": "Total Maps",
            "selection": "Over 2.5",
            "odds": 1.8,
            "reason": "แมตช์น่าจะแข่งนาน"
        })
        
        # First map winner
        backups.append({
            "type": "First Map Winner",
            "selection": match.team1,
            "odds": 2.1,
            "reason": "เริ่มเกมแรงมักได้เปรียบ"
        })
        
        return backups

    def _assess_risk_level(self, analysis: Any, match_importance: float) -> str:
        """ประเมินระดับความเสี่ยง"""
        
        confidence = analysis.prediction_confidence
        
        if confidence > 0.8 and match_importance > 0.8:
            return "low"
        elif confidence > 0.6 and match_importance > 0.6:
            return "medium"
        else:
            return "high"

    def _generate_reasoning(self, match: MatchInfo, analysis: Any, 
                          primary_bet: Dict) -> List[str]:
        """สร้างเหตุผลสำหรับการเดิมพัน"""
        
        reasons = []
        
        # เหตุผลจากการวิเคราะห์
        if analysis.prediction_confidence > 0.8:
            reasons.append(f"ความมั่นใจสูง ({analysis.prediction_confidence:.1%}) จากการวิเคราะห์เชิงลึก")
        
        # เหตุผลจากความสำคัญของแมตช์
        if match.importance > 0.8:
            reasons.append(f"ทัวร์นาเมนท์สำคัญ ({match.event}) - ทีมจะเล่นจริงจัง")
        
        # เหตุผลจากจิตวิทยา
        if hasattr(analysis, 'team1_psychology') and hasattr(analysis, 'team2_psychology'):
            if analysis.team1_psychology.confidence_level > analysis.team2_psychology.confidence_level:
                reasons.append(f"{match.team1} มีความมั่นใจสูงกว่า")
            else:
                reasons.append(f"{match.team2} มีความมั่นใจสูงกว่า")
        
        # เหตุผลจากยุทธศาสตร์
        if hasattr(analysis, 'team1_tactics') and hasattr(analysis, 'team2_tactics'):
            if analysis.team1_tactics.adaptation_ability > 0.7:
                reasons.append(f"{match.team1} มีความสามารถในการปรับตัวสูง")
            if analysis.team2_tactics.adaptation_ability > 0.7:
                reasons.append(f"{match.team2} มีความสามารถในการปรับตัวสูง")
        
        # เหตุผลจากราคา
        if primary_bet.get("odds", 0) > 2.0:
            reasons.append("ราคาต่อรองให้ผลตอบแทนที่คุ้มค่า")
        
        return reasons[:5]  # จำกัด 5 เหตุผล

    def _get_sample_matches(self) -> List[MatchInfo]:
        """ข้อมูลแมตช์ตัวอย่างเมื่อไม่สามารถดึงจาก HLTV ได้"""
        
        return [
            MatchInfo(
                team1="GamerLegion", team2="Virtus.pro",
                time="20:30", event="BLAST Open London 2025",
                importance=0.9, tournament_tier="S", match_id="sample_1"
            ),
            MatchInfo(
                team1="Natus Vincere", team2="FaZe Clan",
                time="23:00", event="IEM Katowice 2025",
                importance=0.95, tournament_tier="S", match_id="sample_2"
            ),
            MatchInfo(
                team1="Astralis", team2="Vitality",
                time="01:30", event="ESL Pro League",
                importance=0.8, tournament_tier="A", match_id="sample_3"
            )
        ]

# ฟังก์ชันหลักสำหรับรัน
async def run_automated_pipeline():
    """รันระบบวิเคราะห์แบบอัตโนมัติ"""
    pipeline = AutomatedBettingPipeline()
    recommendations = await pipeline.run_full_pipeline()
    return recommendations

if __name__ == "__main__":
    asyncio.run(run_automated_pipeline())
