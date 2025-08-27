#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted BLAST Open London 2025 Analysis
วิเคราะห์เฉพาะคู่ที่เลือกจากรูปภาพ - ไม่ scrape ข้อมูลเยอะเกินไป
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what exists
try:
    from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper
except ImportError:
    EnhancedHLTVScraper = None

try:
    from core.enhanced_team_analyzer import EnhancedTeamAnalyzer
except ImportError:
    EnhancedTeamAnalyzer = None

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class TargetedBlastAnalyzer:
    """วิเคราะห์เฉพาะคู่ที่เลือกจาก BLAST Open London 2025"""
    
    def __init__(self):
        self.scraper = EnhancedHLTVScraper() if EnhancedHLTVScraper else None
        self.team_analyzer = EnhancedTeamAnalyzer() if EnhancedTeamAnalyzer else None
        
        # เฉพาะคู่ที่ต้องวิเคราะห์จากรูปภาพ
        self.target_matches = [
            {
                "team1": "Vitality",
                "team2": "M80",
                "time": "LIVE",
                "odds": {"team1": 1.16, "team2": 4.90},
                "status": "live"
            },
            {
                "team1": "GamerLegion", 
                "team2": "Virtus.pro",
                "time": "19:30",
                "odds": {"team1": 2.07, "team2": 1.78},
                "status": "upcoming"
            },
            {
                "team1": "FaZe",
                "team2": "ECSTATIC", 
                "time": "22:00",
                "odds": {"team1": 1.33, "team2": 3.29},
                "status": "upcoming"
            },
            {
                "team1": "Natus Vincere",
                "team2": "fnatic",
                "time": "00:30",
                "odds": {"team1": 1.27, "team2": 3.69},
                "status": "upcoming"
            }
        ]
    
    async def get_team_data_efficiently(self, team_name: str) -> Dict:
        """ดึงข้อมูลทีมแบบประหยัด - เฉพาะที่จำเป็น"""
        try:
            print(f"🔍 กำลังดึงข้อมูล {team_name}...")
            
            # ใช้ข้อมูลพื้นฐานที่รู้จักแล้ว
            team_data = {
                "name": team_name,
                "ranking": self._get_team_ranking(team_name),
                "recent_form": [],
                "players": [],
                "map_pool": {}
            }
            
            # ถ้ามี scraper ให้ลองดึงข้อมูลเพิ่ม
            if self.scraper:
                try:
                    scraped_data = await self.scraper.get_team_info(team_name)
                    if scraped_data:
                        team_data.update(scraped_data)
                except:
                    pass  # ใช้ข้อมูลพื้นฐาน
            
            return team_data
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถดึงข้อมูล {team_name}: {e}")
            return {
                "name": team_name,
                "ranking": "N/A", 
                "recent_form": [],
                "players": [],
                "map_pool": {}
            }
    
    def _get_team_ranking(self, team_name: str) -> str:
        """ข้อมูลอันดับทีมที่รู้จัก"""
        rankings = {
            "Vitality": "#6",
            "M80": "#25", 
            "GamerLegion": "#16",
            "Virtus.pro": "#15",
            "FaZe": "#9",
            "ECSTATIC": "#36",
            "Natus Vincere": "#6",
            "fnatic": "#34"
        }
        return rankings.get(team_name, "N/A")
    
    def analyze_match_efficiently(self, match: Dict, team1_data: Dict, team2_data: Dict) -> Dict:
        """วิเคราะห์แมตช์แบบเฉพาะจุด"""
        
        analysis = {
            "match_info": {
                "teams": f"{match['team1']} vs {match['team2']}",
                "time": match['time'],
                "status": match['status'],
                "tournament": "BLAST Open London 2025"
            },
            "odds_analysis": {
                "favorite": match['team1'] if match['odds']['team1'] < match['odds']['team2'] else match['team2'],
                "underdog": match['team2'] if match['odds']['team1'] < match['odds']['team2'] else match['team1'],
                "odds": match['odds'],
                "implied_probability": {
                    match['team1']: round(1/match['odds']['team1'] * 100, 1),
                    match['team2']: round(1/match['odds']['team2'] * 100, 1)
                }
            },
            "quick_assessment": self._get_quick_assessment(match, team1_data, team2_data),
            "betting_recommendation": self._get_betting_recommendation(match)
        }
        
        return analysis
    
    def _get_quick_assessment(self, match: Dict, team1_data: Dict, team2_data: Dict) -> Dict:
        """ประเมินแมตช์อย่างรวดเร็ว"""
        
        # ใช้ข้อมูลพื้นฐานที่มี
        team1_rank = team1_data.get('ranking', 'N/A')
        team2_rank = team2_data.get('ranking', 'N/A')
        
        assessment = {
            "team1_strengths": self._get_team_strengths(match['team1']),
            "team2_strengths": self._get_team_strengths(match['team2']),
            "key_factors": self._get_key_factors(match),
            "prediction_confidence": self._get_confidence_level(match)
        }
        
        return assessment
    
    def _get_team_strengths(self, team_name: str) -> List[str]:
        """จุดแข็งของทีมตามที่รู้จัก"""
        strengths_db = {
            "Vitality": ["ZywOo superstar", "Strong T-side", "Experience"],
            "M80": ["Upset potential", "Young talent", "Aggressive style"],
            "GamerLegion": ["REZ leadership", "Tactical discipline", "Map pool depth"],
            "Virtus.pro": ["electroNic skill", "CIS aggression", "Clutch ability"],
            "FaZe": ["Individual skill", "Firepower", "International experience"],
            "ECSTATIC": ["Team chemistry", "Tactical innovation", "Underdog motivation"],
            "Natus Vincere": ["s1mple factor", "Tournament experience", "Strategic depth"],
            "fnatic": ["Swedish CS legacy", "Tactical flexibility", "LAN experience"]
        }
        return strengths_db.get(team_name, ["Standard CS fundamentals"])
    
    def _get_key_factors(self, match: Dict) -> List[str]:
        """ปัจจัยสำคัญของแมตช์"""
        factors = []
        
        if match['status'] == 'live':
            factors.append("🔴 แมตช์กำลังแข่งอยู่")
        
        # วิเคราะห์ odds
        odds_diff = abs(match['odds']['team1'] - match['odds']['team2'])
        if odds_diff > 2.0:
            factors.append("📊 ต่างกันมาก - มีโอกาสอัปเซ็ต")
        elif odds_diff < 0.5:
            factors.append("⚖️ ใกล้เคียงกัน - แมตช์คาดเดายาก")
        
        # เวลาแข่ง
        if match['time'] in ['22:00', '00:30']:
            factors.append("🌙 แข่งดึก - อาจมีผลต่อฟอร์ม")
            
        return factors
    
    def _get_confidence_level(self, match: Dict) -> str:
        """ระดับความมั่นใจในการทำนาย"""
        odds_diff = abs(match['odds']['team1'] - match['odds']['team2'])
        
        if odds_diff > 2.5:
            return "🟢 สูง - เต็งชัดเจน"
        elif odds_diff > 1.0:
            return "🟡 ปานกลาง - มีเต็งแต่ไม่แน่นอน"
        else:
            return "🔴 ต่ำ - ใกล้เคียงกันมาก"
    
    def _get_betting_recommendation(self, match: Dict) -> Dict:
        """คำแนะนำการเดิมพัน"""
        
        favorite = match['team1'] if match['odds']['team1'] < match['odds']['team2'] else match['team2']
        underdog = match['team2'] if match['odds']['team1'] < match['odds']['team2'] else match['team1']
        
        fav_odds = min(match['odds']['team1'], match['odds']['team2'])
        dog_odds = max(match['odds']['team1'], match['odds']['team2'])
        
        recommendation = {
            "primary_bet": "",
            "alternative_bet": "",
            "stake_suggestion": "",
            "risk_level": ""
        }
        
        # วิเคราะห์ตาม odds
        if fav_odds < 1.4:  # เต็งแรงมาก
            recommendation["primary_bet"] = f"🎯 {favorite} ML (ราคาต่ำแต่ปลอดภัย)"
            recommendation["alternative_bet"] = f"💰 {underdog} +1.5 Maps (value bet)"
            recommendation["stake_suggestion"] = "เดิมพันปานกลาง"
            recommendation["risk_level"] = "🟢 ต่ำ"
            
        elif fav_odds < 1.8:  # เต็งปานกลาง
            recommendation["primary_bet"] = f"⚖️ {favorite} ML หรือ Handicap"
            recommendation["alternative_bet"] = f"🎲 Over 2.5 Maps"
            recommendation["stake_suggestion"] = "เดิมพันปกติ"
            recommendation["risk_level"] = "🟡 ปานกลาง"
            
        else:  # ใกล้เคียงกัน
            recommendation["primary_bet"] = f"🎰 {underdog} ML (value bet)"
            recommendation["alternative_bet"] = f"📊 Over 2.5 Maps"
            recommendation["stake_suggestion"] = "เดิมพันน้อย"
            recommendation["risk_level"] = "🔴 สูง"
        
        return recommendation
    
    async def run_targeted_analysis(self):
        """รันการวิเคราะห์เฉพาะคู่ที่เลือก"""
        
        print("=" * 80)
        print("🎯 BLAST Open London 2025 - การวิเคราะห์เฉพาะจุด")
        print("=" * 80)
        print(f"⏰ เวลาวิเคราะห์: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"📊 จำนวนแมตช์ที่วิเคราะห์: {len(self.target_matches)} คู่")
        print()
        
        all_analyses = []
        
        for i, match in enumerate(self.target_matches, 1):
            print(f"🔍 กำลังวิเคราะห์แมตช์ที่ {i}: {match['team1']} vs {match['team2']}")
            print("-" * 60)
            
            # ดึงข้อมูลทีมแบบประหยัด
            team1_data = await self.get_team_data_efficiently(match['team1'])
            team2_data = await self.get_team_data_efficiently(match['team2'])
            
            # วิเคราะห์แมตช์
            analysis = self.analyze_match_efficiently(match, team1_data, team2_data)
            all_analyses.append(analysis)
            
            # แสดงผลการวิเคราะห์
            self.display_match_analysis(analysis)
            print()
        
        # สรุปรวม
        self.display_summary(all_analyses)
        
        # บันทึกผลลัพธ์
        await self.save_analysis_results(all_analyses)
        
        return all_analyses
    
    def display_match_analysis(self, analysis: Dict):
        """แสดงผลการวิเคราะห์แมตช์"""
        
        match_info = analysis['match_info']
        odds_info = analysis['odds_analysis']
        assessment = analysis['quick_assessment']
        betting = analysis['betting_recommendation']
        
        print(f"🏆 {match_info['teams']}")
        print(f"⏰ เวลา: {match_info['time']} | สถานะ: {match_info['status']}")
        print(f"🎲 ราคาต่อรอง: {odds_info['odds']}")
        print(f"📊 เต็ง: {odds_info['favorite']} | อันเดอร์ด็อก: {odds_info['underdog']}")
        print()
        
        print("💪 จุดแข็งทีม:")
        print(f"  • {match_info['teams'].split(' vs ')[0]}: {', '.join(assessment['team1_strengths'])}")
        print(f"  • {match_info['teams'].split(' vs ')[1]}: {', '.join(assessment['team2_strengths'])}")
        print()
        
        print("🔑 ปัจจัยสำคัญ:")
        for factor in assessment['key_factors']:
            print(f"  • {factor}")
        print()
        
        print(f"🎯 ความมั่นใจ: {assessment['prediction_confidence']}")
        print()
        
        print("💰 คำแนะนำการเดิมพัน:")
        print(f"  🥇 เดิมพันหลัก: {betting['primary_bet']}")
        print(f"  🥈 เดิมพันสำรอง: {betting['alternative_bet']}")
        print(f"  💵 ขนาดเงินเดิมพัน: {betting['stake_suggestion']}")
        print(f"  ⚠️ ระดับความเสี่ยง: {betting['risk_level']}")
    
    def display_summary(self, analyses: List[Dict]):
        """แสดงสรุปการวิเคราะห์ทั้งหมด"""
        
        print("=" * 80)
        print("📋 สรุปการวิเคราะห์ BLAST Open London 2025")
        print("=" * 80)
        
        print("🎯 แมตช์ที่น่าสนใจที่สุด:")
        
        # หาแมตช์ที่มี value ที่สุด
        best_value = None
        highest_risk = None
        safest_bet = None
        
        for analysis in analyses:
            betting = analysis['betting_recommendation']
            
            if "value bet" in betting['primary_bet'].lower():
                best_value = analysis
            
            if betting['risk_level'] == "🔴 สูง":
                highest_risk = analysis
            elif betting['risk_level'] == "🟢 ต่ำ":
                safest_bet = analysis
        
        if best_value:
            print(f"💎 Value Bet: {best_value['match_info']['teams']}")
        
        if safest_bet:
            print(f"🛡️ เดิมพันปลอดภัย: {safest_bet['match_info']['teams']}")
        
        if highest_risk:
            print(f"🎰 High Risk/High Reward: {highest_risk['match_info']['teams']}")
        
        print()
        print("⚠️ คำเตือน: การเดิมพันมีความเสี่ยง กรุณาเดิมพันอย่างรับผิดชอบ")
        print("📊 ข้อมูลนี้เป็นเพียงการวิเคราะห์ ไม่ใช่คำแนะนำทางการเงิน")
    
    async def save_analysis_results(self, analyses: List[Dict]):
        """บันทึกผลการวิเคราะห์"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/targeted_blast_analysis_{timestamp}.json"
        
        try:
            os.makedirs("data", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "tournament": "BLAST Open London 2025",
                    "analysis_type": "targeted_matches",
                    "matches_analyzed": len(analyses),
                    "analyses": analyses
                }, f, ensure_ascii=False, indent=2)
            
            print(f"💾 บันทึกผลการวิเคราะห์ที่: {filename}")
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถบันทึกไฟล์: {e}")

async def main():
    """ฟังก์ชันหลัก"""
    try:
        analyzer = TargetedBlastAnalyzer()
        await analyzer.run_targeted_analysis()
        
    except KeyboardInterrupt:
        print("\n⏹️ หยุดการวิเคราะห์โดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
