#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep BLAST Open London 2025 Analysis
การวิเคราะห์เจาะลึกพร้อมกลยุทธ์การเดิมพันที่ทำกำไรได้มากที่สุด
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class DeepBlastAnalyzer:
    """การวิเคราะห์เจาะลึก BLAST Open London 2025 เพื่อกำไรสูงสุด"""
    
    def __init__(self):
        self.matches = [
            {
                "id": "vitality_m80",
                "team1": "Vitality", "team2": "M80",
                "time": "LIVE", "status": "live",
                "odds": {"team1": 1.16, "team2": 4.90},
                "rankings": {"team1": 6, "team2": 25},
                "recent_form": {"team1": "WWLWW", "team2": "LWLWL"},
                "map_pool_advantage": "Vitality",
                "key_players": {
                    "team1": ["ZywOo", "apEX", "Magisk", "flameZ", "Spinx"],
                    "team2": ["slaxz-", "malbsMd", "Swisher", "reck", "Lake"]
                }
            },
            {
                "id": "gamerlegion_virtuspro",
                "team1": "GamerLegion", "team2": "Virtus.pro",
                "time": "19:30", "status": "upcoming",
                "odds": {"team1": 2.07, "team2": 1.78},
                "rankings": {"team1": 16, "team2": 15},
                "recent_form": {"team1": "WLWWL", "team2": "LWWLW"},
                "map_pool_advantage": "Even",
                "key_players": {
                    "team1": ["REZ", "ztr", "Tauson", "Kursy", "PR"],
                    "team2": ["electroNic", "FL1T", "Perfecto", "fame", "ICY"]
                }
            },
            {
                "id": "faze_ecstatic",
                "team1": "FaZe", "team2": "ECSTATIC",
                "time": "22:00", "status": "upcoming",
                "odds": {"team1": 1.33, "team2": 3.29},
                "rankings": {"team1": 9, "team2": 36},
                "recent_form": {"team1": "WLWLW", "team2": "WWLWW"},
                "map_pool_advantage": "FaZe",
                "key_players": {
                    "team1": ["karrigan", "rain", "broky", "frozen", "ropz"],
                    "team2": ["jcobbb", "Anlelele", "Nodios", "Peppzor", "Dytor"]
                }
            },
            {
                "id": "navi_fnatic",
                "team1": "Natus Vincere", "team2": "fnatic",
                "time": "00:30", "status": "upcoming",
                "odds": {"team1": 1.27, "team2": 3.69},
                "rankings": {"team1": 6, "team2": 34},
                "recent_form": {"team1": "WWLWW", "team2": "LWWLW"},
                "map_pool_advantage": "NAVI",
                "key_players": {
                    "team1": ["s1mple", "electroNic", "Perfecto", "b1t", "sdy"],
                    "team2": ["KRIMZ", "nicoodoz", "roeJ", "afro", "matys"]
                }
            }
        ]
        
        # Kelly Criterion parameters
        self.bankroll_percentage = {
            "conservative": 0.02,  # 2%
            "moderate": 0.05,      # 5%
            "aggressive": 0.10     # 10%
        }
    
    def calculate_implied_probability(self, odds: float) -> float:
        """คำนวณความน่าจะเป็นจาก odds"""
        return 1 / odds
    
    def calculate_true_probability(self, match: Dict) -> Dict[str, float]:
        """คำนวณความน่าจะเป็นจริงจากการวิเคราะห์"""
        
        # Base probability from rankings
        rank_diff = abs(match["rankings"]["team1"] - match["rankings"]["team2"])
        
        if match["rankings"]["team1"] < match["rankings"]["team2"]:
            # Team1 ranked higher
            base_prob_team1 = 0.65 + (rank_diff * 0.02)
        else:
            # Team2 ranked higher  
            base_prob_team1 = 0.35 - (rank_diff * 0.02)
        
        # Adjust for recent form
        team1_form_score = self._calculate_form_score(match["recent_form"]["team1"])
        team2_form_score = self._calculate_form_score(match["recent_form"]["team2"])
        
        form_adjustment = (team1_form_score - team2_form_score) * 0.05
        base_prob_team1 += form_adjustment
        
        # Map pool advantage
        if match["map_pool_advantage"] == match["team1"]:
            base_prob_team1 += 0.05
        elif match["map_pool_advantage"] == match["team2"]:
            base_prob_team1 -= 0.05
        
        # Clamp between 0.1 and 0.9
        base_prob_team1 = max(0.1, min(0.9, base_prob_team1))
        
        return {
            "team1": base_prob_team1,
            "team2": 1 - base_prob_team1
        }
    
    def _calculate_form_score(self, form: str) -> float:
        """คำนวณคะแนนจากฟอร์มล่าสุด (W=1, L=0)"""
        score = 0
        weights = [0.4, 0.3, 0.2, 0.08, 0.02]  # น้ำหนักแมตช์ล่าสุดมากกว่า
        
        for i, result in enumerate(form):
            if i < len(weights):
                score += weights[i] * (1 if result == 'W' else 0)
        
        return score
    
    def calculate_kelly_criterion(self, odds: float, true_prob: float) -> float:
        """คำนวณ Kelly Criterion สำหรับขนาดเดิมพัน"""
        if true_prob <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # b = odds - 1, p = true probability, q = 1 - p
        b = odds - 1
        p = true_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Return positive Kelly only (no negative bets)
        return max(0, kelly_fraction)
    
    def find_value_bets(self, match: Dict) -> List[Dict]:
        """หา value bets ที่มีกำไรสูงสุด"""
        
        true_probs = self.calculate_true_probability(match)
        implied_probs = {
            "team1": self.calculate_implied_probability(match["odds"]["team1"]),
            "team2": self.calculate_implied_probability(match["odds"]["team2"])
        }
        
        value_bets = []
        
        # Check Team 1
        if true_probs["team1"] > implied_probs["team1"]:
            edge = true_probs["team1"] - implied_probs["team1"]
            kelly = self.calculate_kelly_criterion(match["odds"]["team1"], true_probs["team1"])
            
            value_bets.append({
                "team": match["team1"],
                "bet_type": "Moneyline",
                "odds": match["odds"]["team1"],
                "true_prob": true_probs["team1"],
                "implied_prob": implied_probs["team1"],
                "edge": edge,
                "kelly_fraction": kelly,
                "expected_value": (true_probs["team1"] * (match["odds"]["team1"] - 1)) - (1 - true_probs["team1"]),
                "confidence": "High" if edge > 0.1 else "Medium" if edge > 0.05 else "Low"
            })
        
        # Check Team 2
        if true_probs["team2"] > implied_probs["team2"]:
            edge = true_probs["team2"] - implied_probs["team2"]
            kelly = self.calculate_kelly_criterion(match["odds"]["team2"], true_probs["team2"])
            
            value_bets.append({
                "team": match["team2"],
                "bet_type": "Moneyline",
                "odds": match["odds"]["team2"],
                "true_prob": true_probs["team2"],
                "implied_prob": implied_probs["team2"],
                "edge": edge,
                "kelly_fraction": kelly,
                "expected_value": (true_probs["team2"] * (match["odds"]["team2"] - 1)) - (1 - true_probs["team2"]),
                "confidence": "High" if edge > 0.1 else "Medium" if edge > 0.05 else "Low"
            })
        
        return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)
    
    def generate_alternative_bets(self, match: Dict) -> List[Dict]:
        """สร้างเดิมพันทางเลือกที่มีกำไรสูง"""
        
        alternatives = []
        
        # Over/Under Maps (สมมติ line ที่ 2.5)
        if match["status"] != "live":
            # คำนวณโอกาสไป 3 แมป
            close_match = abs(match["rankings"]["team1"] - match["rankings"]["team2"]) <= 5
            
            if close_match:
                over_25_prob = 0.65
                alternatives.append({
                    "bet_type": "Over 2.5 Maps",
                    "estimated_odds": 1.85,
                    "true_prob": over_25_prob,
                    "reasoning": "คู่ใกล้เคียง มีโอกาสไป 3 แมปสูง",
                    "expected_value": (over_25_prob * 0.85) - (1 - over_25_prob),
                    "risk_level": "Medium"
                })
        
        # Handicap bets
        favorite = match["team1"] if match["odds"]["team1"] < match["odds"]["team2"] else match["team2"]
        underdog = match["team2"] if match["odds"]["team1"] < match["odds"]["team2"] else match["team1"]
        
        # +1.5 Maps for underdog
        handicap_prob = 0.45 if abs(match["rankings"]["team1"] - match["rankings"]["team2"]) > 10 else 0.55
        alternatives.append({
            "bet_type": f"{underdog} +1.5 Maps",
            "estimated_odds": 1.65,
            "true_prob": handicap_prob,
            "reasoning": f"{underdog} มีโอกาสได้อย่างน้อย 1 แมป",
            "expected_value": (handicap_prob * 0.65) - (1 - handicap_prob),
            "risk_level": "Low"
        })
        
        return alternatives
    
    def analyze_match_deep(self, match: Dict) -> Dict:
        """วิเคราะห์แมตช์เจาะลึก"""
        
        print(f"\n{'='*80}")
        print(f"🔍 การวิเคราะห์เจาะลึก: {match['team1']} vs {match['team2']}")
        print(f"{'='*80}")
        
        # Basic info
        print(f"⏰ เวลา: {match['time']} | สถานะ: {match['status']}")
        print(f"🏆 อันดับ: #{match['rankings']['team1']} vs #{match['rankings']['team2']}")
        print(f"📊 ฟอร์มล่าสุด: {match['recent_form']['team1']} vs {match['recent_form']['team2']}")
        print(f"🗺️ ความได้เปรียบ Map Pool: {match['map_pool_advantage']}")
        
        # Odds analysis
        print(f"\n💰 การวิเคราะห์ราคาต่อรอง:")
        print(f"   {match['team1']}: {match['odds']['team1']} ({self.calculate_implied_probability(match['odds']['team1']):.1%})")
        print(f"   {match['team2']}: {match['odds']['team2']} ({self.calculate_implied_probability(match['odds']['team2']):.1%})")
        
        # True probability
        true_probs = self.calculate_true_probability(match)
        print(f"\n🎯 ความน่าจะเป็นจริง (จากการวิเคราะห์):")
        print(f"   {match['team1']}: {true_probs['team1']:.1%}")
        print(f"   {match['team2']}: {true_probs['team2']:.1%}")
        
        # Value bets
        value_bets = self.find_value_bets(match)
        print(f"\n💎 Value Bets ที่พบ:")
        
        if value_bets:
            for bet in value_bets:
                print(f"   🎯 {bet['team']} ML @ {bet['odds']}")
                print(f"      Edge: {bet['edge']:.1%} | EV: {bet['expected_value']:.3f}")
                print(f"      Kelly: {bet['kelly_fraction']:.1%} | ความมั่นใจ: {bet['confidence']}")
                
                # Stake recommendations
                conservative_stake = bet['kelly_fraction'] * self.bankroll_percentage['conservative'] * 100
                moderate_stake = bet['kelly_fraction'] * self.bankroll_percentage['moderate'] * 100
                aggressive_stake = bet['kelly_fraction'] * self.bankroll_percentage['aggressive'] * 100
                
                print(f"      💵 แนะนำเดิมพัน (% ของ bankroll):")
                print(f"         🛡️ อนุรักษ์: {conservative_stake:.2f}%")
                print(f"         ⚖️ ปานกลาง: {moderate_stake:.2f}%")
                print(f"         🚀 ก้าวร้าว: {aggressive_stake:.2f}%")
        else:
            print("   ❌ ไม่พบ value bet ในราคา moneyline")
        
        # Alternative bets
        alternatives = self.generate_alternative_bets(match)
        print(f"\n🎲 เดิมพันทางเลือก:")
        
        for alt in alternatives:
            print(f"   📊 {alt['bet_type']} @ ~{alt['estimated_odds']}")
            print(f"      เหตุผล: {alt['reasoning']}")
            print(f"      EV: {alt['expected_value']:.3f} | ความเสี่ยง: {alt['risk_level']}")
        
        # Match-specific insights
        insights = self.get_match_insights(match)
        print(f"\n🧠 ข้อมูลเชิงลึก:")
        for insight in insights:
            print(f"   • {insight}")
        
        # Final recommendation
        recommendation = self.get_final_recommendation(match, value_bets, alternatives)
        print(f"\n🎯 คำแนะนำสุดท้าย:")
        print(f"   🥇 เดิมพันหลัก: {recommendation['primary']}")
        print(f"   🥈 เดิมพันสำรอง: {recommendation['secondary']}")
        print(f"   💰 กลยุทธ์: {recommendation['strategy']}")
        print(f"   ⚠️ ความเสี่ยง: {recommendation['risk_assessment']}")
        
        return {
            "match_id": match["id"],
            "true_probabilities": true_probs,
            "value_bets": value_bets,
            "alternative_bets": alternatives,
            "insights": insights,
            "recommendation": recommendation
        }
    
    def get_match_insights(self, match: Dict) -> List[str]:
        """ข้อมูลเชิงลึกเฉพาะแมตช์"""
        
        insights = []
        
        if match["id"] == "vitality_m80":
            insights.extend([
                "ZywOo เป็นปัจจัยสำคัญ - ถ้าเขาเล่นได้ดี Vitality ชนะง่าย",
                "M80 เป็นทีมอเมริกันที่มี upset potential สูง",
                "แมตช์ LIVE - สามารถดู momentum ก่อนเดิมพัน live betting",
                "Vitality มีประสบการณ์ระดับ Tier 1 มากกว่า"
            ])
        
        elif match["id"] == "gamerlegion_virtuspro":
            insights.extend([
                "คู่ที่ใกล้เคียงที่สุด - อันดับต่างกันแค่ 1 อันดับ",
                "REZ vs electroNic - การดวลของ star players",
                "Virtus.pro มีประสบการณ์ในทัวร์นาเมนต์ใหญ่มากกว่า",
                "GamerLegion มีการเตรียมตัวที่ดีและมีแผนการเล่นชัดเจน"
            ])
        
        elif match["id"] == "faze_ecstatic":
            insights.extend([
                "FaZe มี firepower สูงแต่บางครั้งไม่สม่ำเสมอ",
                "ECSTATIC มีฟอร์มดีล่าสุด (4 ชนะใน 5 แมตช์)",
                "jcobbb เป็นผู้เล่นใหม่ที่อาจสร้างความแปลกใจ",
                "แข่งเวลา 22:00 - เวลาที่ดีสำหรับทีมยุโรป"
            ])
        
        elif match["id"] == "navi_fnatic":
            insights.extend([
                "s1mple กลับมาแล้ว - เพิ่มพลังให้ NAVI มาก",
                "fnatic กำลังสร้างทีมใหม่ - อาจยังไม่เข้าขั้น",
                "แข่งเวลาดึก (00:30) - อาจมีผลต่อการเล่น",
                "NAVI มีประสบการณ์ Major มากกว่า"
            ])
        
        return insights
    
    def get_final_recommendation(self, match: Dict, value_bets: List[Dict], alternatives: List[Dict]) -> Dict:
        """คำแนะนำสุดท้ายสำหรับการเดิมพัน"""
        
        if value_bets:
            best_value = value_bets[0]
            primary = f"{best_value['team']} ML @ {best_value['odds']} (Kelly: {best_value['kelly_fraction']:.1%})"
        else:
            # ไม่มี value bet - แนะนำ alternative
            if match["odds"]["team1"] < match["odds"]["team2"]:
                primary = f"รอราคาดีขึ้น หรือ {match['team2']} +1.5 Maps"
            else:
                primary = f"รอราคาดีขึ้น หรือ {match['team1']} +1.5 Maps"
        
        # Secondary bet
        if alternatives:
            best_alt = max(alternatives, key=lambda x: x["expected_value"])
            secondary = f"{best_alt['bet_type']} @ ~{best_alt['estimated_odds']}"
        else:
            secondary = "Over 2.5 Maps (ถ้ามี)"
        
        # Strategy based on match
        if match["status"] == "live":
            strategy = "รอดู momentum ในแมตช์ก่อนเดิมพัน live betting"
        elif len(value_bets) > 0 and value_bets[0]["confidence"] == "High":
            strategy = "เดิมพันตาม Kelly Criterion เต็มจำนวน"
        else:
            strategy = "เดิมพันระมัดระวัง หรือข้ามแมตช์นี้"
        
        # Risk assessment
        if match["id"] == "vitality_m80":
            risk = "🟢 ต่ำ-ปานกลาง (Vitality เต็งแรง แต่ M80 มี upset potential)"
        elif match["id"] == "gamerlegion_virtuspro":
            risk = "🟡 ปานกลาง-สูง (คู่ใกล้เคียง คาดเดายาก)"
        elif match["id"] == "faze_ecstatic":
            risk = "🟡 ปานกลาง (FaZe ไม่สม่ำเสมอ, ECSTATIC ฟอร์มดี)"
        else:  # navi_fnatic
            risk = "🟢 ต่ำ-ปานกลาง (NAVI เต็งแต่เล่นดึก)"
        
        return {
            "primary": primary,
            "secondary": secondary,
            "strategy": strategy,
            "risk_assessment": risk
        }
    
    async def run_deep_analysis(self):
        """รันการวิเคราะห์เจาะลึกทั้งหมด"""
        
        print("🎯 BLAST Open London 2025 - การวิเคราะห์เจาะลึกเพื่อกำไรสูงสุด")
        print(f"⏰ เวลาวิเคราะห์: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        all_analyses = []
        
        for match in self.matches:
            analysis = self.analyze_match_deep(match)
            all_analyses.append(analysis)
        
        # Portfolio summary
        self.generate_portfolio_summary(all_analyses)
        
        # Save results
        await self.save_deep_analysis(all_analyses)
        
        return all_analyses
    
    def generate_portfolio_summary(self, analyses: List[Dict]):
        """สรุปพอร์ตการเดิมพันทั้งหมด"""
        
        print(f"\n{'='*80}")
        print("📊 สรุปพอร์ตการเดิมพัน BLAST Open London 2025")
        print(f"{'='*80}")
        
        total_value_bets = sum(len(analysis["value_bets"]) for analysis in analyses)
        
        print(f"🎯 Value Bets ที่พบทั้งหมด: {total_value_bets} เดิมพัน")
        
        if total_value_bets > 0:
            print(f"\n💎 Top Value Bets:")
            
            all_value_bets = []
            for analysis in analyses:
                for bet in analysis["value_bets"]:
                    bet["match_id"] = analysis["match_id"]
                    all_value_bets.append(bet)
            
            # Sort by expected value
            top_bets = sorted(all_value_bets, key=lambda x: x["expected_value"], reverse=True)[:3]
            
            for i, bet in enumerate(top_bets, 1):
                print(f"   {i}. {bet['team']} @ {bet['odds']} (EV: {bet['expected_value']:.3f})")
                print(f"      Kelly: {bet['kelly_fraction']:.1%} | Edge: {bet['edge']:.1%}")
        
        print(f"\n🎲 กลยุทธ์การจัดการเงิน:")
        print(f"   💰 แบ่ง bankroll เป็น 4 ส่วนสำหรับแต่ละแมตช์")
        print(f"   📊 ใช้ Kelly Criterion สำหรับขนาดเดิมพัน")
        print(f"   🛡️ ไม่เดิมพันเกิน 10% ของ bankroll ในแมตช์เดียว")
        print(f"   ⏰ รอดูแมตช์ LIVE ก่อนตัดสินใจ")
        
        print(f"\n⚠️ คำเตือนสำคัญ:")
        print(f"   • การเดิมพันมีความเสี่ยง อาจสูญเสียเงินได้")
        print(f"   • ข้อมูลนี้เป็นการวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำทางการเงิน")
        print(f"   • ควรเดิมพันด้วยเงินที่สามารถเสียได้เท่านั้น")
        print(f"   • หยุดเดิมพันหากขาดทุนเกิน 20% ของ bankroll")
    
    async def save_deep_analysis(self, analyses: List[Dict]):
        """บันทึกผลการวิเคราะห์เจาะลึก"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/deep_blast_analysis_{timestamp}.json"
        
        try:
            os.makedirs("data", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "tournament": "BLAST Open London 2025",
                    "analysis_type": "deep_profitable_analysis",
                    "total_matches": len(analyses),
                    "total_value_bets": sum(len(a["value_bets"]) for a in analyses),
                    "analyses": analyses
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 บันทึกผลการวิเคราะห์ที่: {filename}")
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถบันทึกไฟล์: {e}")

async def main():
    """ฟังก์ชันหลัก"""
    try:
        analyzer = DeepBlastAnalyzer()
        await analyzer.run_deep_analysis()
        
    except KeyboardInterrupt:
        print("\n⏹️ หยุดการวิเคราะห์โดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
