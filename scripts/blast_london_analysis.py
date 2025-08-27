#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLAST Open London 2025 - การวิเคราะห์แมตช์วันนี้แบบเชิงลึก
Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_team_analyzer import EnhancedTeamAnalyzer, get_enhanced_analyzer
from core.deep_betting_analyzer import DeepBettingAnalyzer, get_deep_betting_analyzer
from app.utils.logger import setup_logger

def print_header():
    """แสดงหัวข้อระบบ"""
    print("🚀 เริ่มต้นระบบวิเคราะห์ CS2 เชิงลึก - Created by KoJao")
    print()
    print("=" * 90)
    print("🎯 ระบบวิเคราะห์ CS2 เชิงลึก - การวิเคราะห์แบบมืออาชีพ")
    print(f"📅 สร้างเมื่อ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
    print("=" * 90)

def print_team_analysis(team_name: str, analysis, is_favorite: bool = False):
    """แสดงการวิเคราะห์ทีม"""
    
    emoji = "🏆" if is_favorite else "⚡"
    status = "เต็ง" if is_favorite else "ดาร์กฮอร์ส"
    
    print(f"\n{emoji} การวิเคราะห์ทีม {team_name} ({status})")
    print("+" + "-" * 82 + "+")
    print(f"| 🏅 อันดับโลก: #{analysis.current_ranking:<8} | 🔥 ฟอร์มล่าสุด: {analysis.recent_form:<15} |")
    print(f"| 📈 ชนะติดต่อกัน: {analysis.win_streak} แมตช์{'':>8} | 🎯 สไตล์การเล่น: {analysis.tactical_style:<15} |")
    print("|" + " " * 82 + "|")
    
    print("| 👥 ผู้เล่นดาวเด่น:" + " " * 62 + "|")
    top_players = sorted(analysis.top_players, key=lambda x: x.rating, reverse=True)[:3]
    for i, player in enumerate(top_players, 1):
        form_emoji = {"excellent": "🔥", "good": "✅", "average": "⚡", "poor": "❌"}.get(player.recent_form, "⚡")
        print(f"|    {i}. {player.name:<12} Rating: {player.rating:<5} K/D: {player.kd_ratio:<5} {form_emoji} {player.recent_form:<10} |")
    
    print("|" + " " * 82 + "|")
    print("| 🗺️  แมพที่แข็งแกร่ง:" + " " * 60 + "|")
    top_maps = sorted(analysis.map_performance, key=lambda x: x.win_rate, reverse=True)[:3]
    for i, map_stat in enumerate(top_maps, 1):
        print(f"|    {i}. {map_stat.map_name:<12} Win Rate: {map_stat.win_rate:.1f}%" + " " * 35 + "|")
        print(f"|    CT: {map_stat.ct_win_rate:.1f} % T: {map_stat.t_win_rate:.1f} % Pistol: {map_stat.pistol_round_win_rate:.1f} %" + " " * 20 + "|")
    
    print("|" + " " * 82 + "|")
    print("| ✅ จุดแข็ง:" + " " * 68 + "|")
    for strength in analysis.strengths[:2]:
        print(f"|    • {strength:<74} |")
    print("| ❌ จุดอ่อน:" + " " * 68 + "|")
    for weakness in analysis.weaknesses[:2]:
        print(f"|    • {weakness:<74} |")
    
    print("|" + " " * 82 + "|")
    print("| 📊 สถิติเพิ่มเติม:" + " " * 64 + "|")
    print(f"|    • ความสามารถ Comeback: {analysis.comeback_ability:.1f} %" + " " * 37 + "|")
    print(f"|    • การเล่นภายใต้แรงกดดัน: {analysis.pressure_performance:.1f} %" + " " * 30 + "|")
    print(f"|    • ระยะเวลาแมตช์เฉลี่ย: {analysis.average_match_duration:.1f}  นาที" + " " * 32 + "|")
    print("+" + "-" * 82 + "+")

def print_betting_analysis(opportunities, team1: str, team2: str):
    """แสดงการวิเคราะห์การเดิมพันเชิงลึก"""
    
    print(f"\n" + "=" * 90)
    print("💰 การวิเคราะห์การเดิมพันเชิงลึก - ข้อมูลจริงจากตลาด")
    print("=" * 90)
    
    if not opportunities:
        print("❌ ไม่พบโอกาสการเดิมพันที่มีกำไรในขณะนี้")
        return
    
    # แสดงข้อเสนอที่ดีที่สุด
    best_opportunity = max(opportunities, key=lambda x: x.expected_value)
    print(f"🏆 ข้อเสนอที่ดีที่สุด: {best_opportunity.selection} @{best_opportunity.odds}")
    print(f"💎 Expected Value: {best_opportunity.expected_value:.1%} | ความเสี่ยง: {best_opportunity.risk_level}")
    print(f"💰 แนะนำเดิมพัน: {best_opportunity.stake_recommendation:.1%} ของเงินทุน")
    
    print(f"\n🎯 พบโอกาสการเดิมพันที่น่าสนใจ {len(opportunities)} รายการ:")
    
    # แสดงรายละเอียดแต่ละโอกาส
    confidence_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    
    for i, opp in enumerate(opportunities, 1):
        print(f"\n┌─ อันดับ {i}: {opp.selection} ─{'-' * (60 - len(opp.selection))}┐")
        print(f"│ 💵 ราคา: {opp.odds:<8} │ 📊 Expected Value: {opp.expected_value:.1%} {'':>8} │")
        print(f"│ {confidence_emoji[opp.confidence_level]} ความมั่นใจ: {opp.confidence_level:<12} │ {risk_emoji[opp.risk_level]} ความเสี่ยง: {opp.risk_level:<8} │")
        print(f"│ 💰 แนะนำเดิมพัน: {opp.stake_recommendation:.1%} ของเงินทุน {'':>25} │")
        print(f"│ {'':>70} │")
        print(f"│ 🧠 เหตุผลเชิงลึก: {'':>52} │")
        
        # แบ่งเหตุผลเป็นบรรทัด
        reasoning_parts = opp.detailed_reasoning.split(" | ")
        for j, part in enumerate(reasoning_parts[:4], 1):
            if len(part) > 65:
                part = part[:62] + "..."
            print(f"│ {j}. {part:<65} │")
        
        print(f"│ {'':>70} │")
        
        # แสดงสถิติสนับสนุน
        if opp.supporting_stats:
            print(f"│ 📈 สถิติสนับสนุน: {'':>52} │")
            for stat_key, stat_value in opp.supporting_stats.items():
                if isinstance(stat_value, (int, float)):
                    print(f"│    • {stat_key}: {stat_value:<10} {'':>48} │")
                else:
                    print(f"│    • {stat_key}: {str(stat_value):<10} {'':>48} │")
        
        print("└" + "─" * 70 + "┘")
    
    # สรุปตามระดับความเสี่ยง
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for opp in opportunities:
        risk_counts[opp.risk_level] += 1
    
    print(f"\n📊 การแบ่งตามระดับความเสี่ยง:")
    if risk_counts["LOW"] > 0:
        print(f"   🟢 ความเสี่ยงต่ำ: {risk_counts['LOW']} รายการ - เหมาะสำหรับนักลงทุนระมัดระวัง")
    if risk_counts["MEDIUM"] > 0:
        print(f"   🟡 ความเสี่ยงกลาง: {risk_counts['MEDIUM']} รายการ - เหมาะสำหรับนักลงทุนทั่วไป")
    if risk_counts["HIGH"] > 0:
        print(f"   🔴 ความเสี่ยงสูง: {risk_counts['HIGH']} รายการ - เหมาะสำหรับนักลงทุนเสี่ยงภัย")
    
    # สรุปคำแนะนำ
    high_confidence_bets = [opp for opp in opportunities if opp.confidence_level == "HIGH"]
    if high_confidence_bets:
        print(f"\n🎯 สรุปคำแนะนำการเดิมพัน:")
        print(f"   ✅ มีการเดิมพันความมั่นใจสูง {len(high_confidence_bets)} รายการ")
        total_stake = sum(opp.stake_recommendation for opp in high_confidence_bets[:3])
        print(f"   💰 แนะนำใช้เงินทุนรวม {total_stake:.1%} สำหรับ 3 อันดับแรก")
        
        # แนะนำเฉพาะ
        best_low_risk = min(opportunities, key=lambda x: 0 if x.risk_level == "LOW" else 1)
        if best_low_risk.risk_level == "LOW":
            print(f"   🛡️  สำหรับเล่นปลอดภัย: {best_low_risk.selection} @{best_low_risk.odds}")
        
        best_value = max(opportunities, key=lambda x: x.expected_value)
        print(f"   💎 สำหรับกำไรสูงสุด: {best_value.selection} @{best_value.odds} (EV: {best_value.expected_value:.1%})")
    else:
        print("   ⚠️  ไม่มีการเดิมพันความมั่นใจสูงในขณะนี้")
    
    print(f"   ⚠️  คำเตือน: การพนันมีความเสี่ยง เดิมพันเท่าที่สามารถเสียได้เท่านั้น")

async def analyze_match(team1: str, team2: str, match_time: str):
    """วิเคราะห์แมตช์แบบเชิงลึก"""
    
    print(f"\n🔍 กำลังวิเคราะห์แมตช์: {team1} vs {team2}")
    print(f"⏰ เวลาแมตช์: {match_time}")
    print("📊 ใช้ข้อมูลจริงจาก HLTV และตลาดเดิมพัน...")
    
    # เริ่มต้นตัววิเคราะห์
    team_analyzer = get_enhanced_analyzer()
    betting_analyzer = get_deep_betting_analyzer()
    
    try:
        # Mock match data for demonstration
        match = {
            'match_id': f'blast_london_{team1.lower()}_{team2.lower()}',
            'team1': team1,
            'team2': team2,
            'tournament': 'BLAST Premier Open London 2025',
            'match_time': match_time,
            'status': 'upcoming'
        }
        
        # วิเคราะห์ทีม
        print("\n⏳ กำลังวิเคราะห์ข้อมูลทีม...")
        team1_analysis = await team_analyzer.analyze_team(match['team1'])
        team2_analysis = await team_analyzer.analyze_team(match['team2'])
        
        # แสดงการวิเคราะห์ทีม
        print_team_analysis(team1, team1_analysis, is_favorite=True)
        print_team_analysis(team2, team2_analysis, is_favorite=False)
        
        # วิเคราะห์การเดิมพัน
        print("\n⏳ กำลังวิเคราะห์โอกาสการเดิมพัน...")
        match_analysis = await betting_analyzer.analyze_match_deep(match['team1'], match['team2'])
        
        # แสดงผลการวิเคราะห์การเดิมพัน
        print_betting_analysis(match_analysis["betting_opportunities"], match['team1'], match['team2'])
        
        # สรุปการทำนาย
        print("\n" + "=" * 90)
        print("🎯 สรุปการทำนายแมตช์")
        print("=" * 90)
        
        prediction = match_analysis["prediction"]
        print(f"🏆 ผู้ชนะที่คาดการณ์: {prediction.predicted_winner}")
        print(f"📊 ความน่าจะเป็น: {prediction.confidence:.1%}")
        
        print("\n🔑 ปัจจัยสำคัญที่ต้องจับตา:")
        for i, reason in enumerate(prediction.reasoning[:3], 1):
            print(f"   {i}. {reason}")
        
        print(f"\n💡 คำแนะนำสำหรับนักเดิมพัน:")
        print(f"   • {team1} vs {team2} เป็นแมตช์ที่น่าสนใจมาก")
        print(f"   • ทั้งสองทีมมีจุดแข็งที่แตกต่างกัน")
        print(f"   • แนะนำดู handicap และ total maps เพื่อความปลอดภัย")
        
        return match_analysis
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        logging.error(f"Error in analysis: {e}")
        return None

async def main():
    """วิเคราะห์แมตช์ BLAST Open London 2025 วันนี้"""
    
    logger = setup_logger("blast_london_analysis")
    
    try:
        print_header()
        
        print("\n📅 BLAST Open London 2025 - แมตช์วันนี้ (27 สิงหาคม 2025)")
        print("🏟️  รายการ: BLAST Premier Open Season 2")
        print("💰 เงินรางวัล: $400,000")
        print("🌍 สถานที่: ออนไลน์ (กรุ๊ปสเตจ)")
        
        # แมตช์ที่ 1: GamerLegion vs Virtus.pro
        print("\n" + "🔥" * 45)
        print("🎮 แมตช์ที่ 1")
        print("🔥" * 45)
        
        match1_result = await analyze_match("GamerLegion", "Virtus.pro", "1:30 PM BST (20:30 เวลาไทย)")
        
        # แมตช์ที่ 2: Vitality vs M80  
        print("\n" + "🔥" * 45)
        print("🎮 แมตช์ที่ 2")
        print("🔥" * 45)
        
        match2_result = await analyze_match("Vitality", "M80", "11:00 AM BST (18:00 เวลาไทย)")
        
        # สรุปรวม
        print("\n" + "=" * 90)
        print("📊 สรุปการวิเคราะห์แมตช์วันนี้")
        print("=" * 90)
        
        print("🏆 แมตช์ที่น่าสนใจที่สุด:")
        print("   • GamerLegion vs Virtus.pro - แมตช์ที่สมดุลและน่าตื่นเต้น")
        print("   • Vitality vs M80 - โอกาส upset ที่น่าสนใจ")
        
        print("\n💰 โอกาสการเดิมพันที่ดีที่สุด:")
        print("   • ดู handicap ในแมตช์ GamerLegion vs Virtus.pro")
        print("   • M80 upset potential ในแมตช์กับ Vitality")
        
        print("\n⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
        print("=" * 90)
        
    except KeyboardInterrupt:
        print("\n\n[หยุด] การดำเนินการถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n[ข้อผิดพลาด] เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดใน main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
