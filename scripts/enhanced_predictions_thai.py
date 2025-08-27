#!/usr/bin/env python3
"""
Enhanced Predictions Thai - ระบบทำนายแบบเชิงลึกพร้อมข้อมูลจริง
Created by KoJao
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Fix Windows console encoding
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.enhanced_team_analyzer import get_enhanced_analyzer
from core.deep_betting_analyzer import get_deep_betting_analyzer
from app.utils.logger import setup_logger


def print_header():
    """พิมพ์หัวข้อพร้อมเวลาปัจจุบัน"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 90)
    print("🎯 ระบบวิเคราะห์ CS2 เชิงลึก - การวิเคราะห์แบบมืออาชีพ")
    print(f"📅 สร้างเมื่อ: {current_time}")
    print("⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
    print("=" * 90)


def print_team_analysis(team_name: str, analysis, is_favorite: bool = False):
    """แสดงการวิเคราะห์ทีมแบบเชิงลึก"""
    
    status = "🏆 ทีมเต็ง" if is_favorite else "🎯 ทีมรอง"
    print(f"\n+-- {status} {team_name} (อันดับ {analysis.current_ranking}) {'-' * (60 - len(team_name))}+")
    
    # ฟอร์มและสถิติ
    form_emoji = {"excellent": "🔥", "good": "✅", "average": "⚡", "poor": "❌"}
    print(f"| 📈 ฟอร์มปัจจุบัน: {form_emoji[analysis.recent_form]} {analysis.recent_form.upper()} (ชนะติดต่อกัน {analysis.win_streak} แมตช์) {'':>10} |")
    print(f"| 🎮 สไตล์การเล่น: {analysis.tactical_style:<50} |")
    
    # ผู้เล่นดาวเด่น
    print(f"| {'':>80} |")
    print(f"| 🌟 ผู้เล่นดาวเด่น: {'':>58} |")
    
    # เรียงผู้เล่นตาม rating
    sorted_players = sorted(analysis.players, key=lambda p: p.rating, reverse=True)
    
    for i, player in enumerate(sorted_players[:3], 1):
        form_status = {"excellent": "🔥", "good": "✅", "average": "⚡", "poor": "❌"}[player.recent_form]
        print(f"| {i}. {player.name:<12} Rating: {player.rating:<5} K/D: {player.kd_ratio:<5} {form_status} {player.recent_form:<8} |")
        print(f"|    ADR: {player.adr:<5} KAST: {player.kast:<5}% Clutch: {player.clutch_success_rate:<5}% {'':>20} |")
    
    # แมพที่เก่ง
    print(f"| {'':>80} |")
    print(f"| 🗺️  แมพที่เก่งที่สุด: {'':>56} |")
    
    best_maps = sorted(analysis.map_pool, key=lambda m: m.win_rate, reverse=True)[:3]
    for i, map_stat in enumerate(best_maps, 1):
        recent_form = "".join(map_stat.recent_performance[-3:])
        print(f"| {i}. {map_stat.map_name:<10} Win Rate: {map_stat.win_rate:<5.1f}% Form: {recent_form:<8} {'':>25} |")
        print(f"|    CT: {map_stat.ct_win_rate:<5.1f}% T: {map_stat.t_win_rate:<5.1f}% Pistol: {map_stat.pistol_round_win_rate:<5.1f}% {'':>18} |")
    
    # จุดแข็ง-จุดอ่อน
    print(f"| {'':>80} |")
    print(f"| ✅ จุดแข็ง: {'':>66} |")
    for strength in analysis.strengths[:2]:
        print(f"|    • {strength:<74} |")
    
    if analysis.weaknesses:
        print(f"| ❌ จุดอ่อน: {'':>66} |")
        for weakness in analysis.weaknesses[:2]:
            print(f"|    • {weakness:<74} |")
    
    # สถิติเพิ่มเติม
    print(f"| {'':>80} |")
    print(f"| 📊 สถิติเพิ่มเติม: {'':>62} |")
    print(f"|    • ความสามารถ Comeback: {analysis.comeback_ability:<5.1f}% {'':>35} |")
    print(f"|    • การเล่นภายใต้แรงกดดัน: {analysis.pressure_performance:<5.1f}% {'':>28} |")
    print(f"|    • ระยะเวลาแมตช์เฉลี่ย: {analysis.avg_match_duration:<5.1f} นาที {'':>30} |")
    
    print(f"+{'-' * 82}+")


def print_betting_analysis(opportunities: List, team1: str, team2: str):
    """แสดงการวิเคราะห์การเดิมพันเชิงลึก"""
    
    print(f"\n" + "=" * 90)
    print("💰 การวิเคราะห์การเดิมพันเชิงลึก - ข้อมูลจริงจากตลาด")
    print("=" * 90)
    
    if not opportunities:
        print("❌ ไม่พบโอกาสการเดิมพันที่มีกำไรในขณะนี้")
        return
    
    print(f"🎯 พบโอกาสการเดิมพันที่น่าสนใจ {len(opportunities)} รายการ:")
    print()
    
    for i, opp in enumerate(opportunities, 1):
        # กำหนดสีตามระดับความมั่นใจ
        confidence_emoji = {
            "HIGH": "🔥", "VERY_HIGH": "💎", 
            "MEDIUM": "⚡", "LOW": "⚠️", "VERY_LOW": "❌"
        }
        
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        
        print(f"┌─ อันดับ {i}: {opp.selection} ─{'-' * (60 - len(opp.selection))}┐")
        print(f"│ 💵 ราคา: {opp.odds:<8} │ 📊 Expected Value: {opp.expected_value:.1%} {'':>8} │")
        print(f"│ {confidence_emoji[opp.confidence_level]} ความมั่นใจ: {opp.confidence_level:<12} │ {risk_emoji[opp.risk_level]} ความเสี่ยง: {opp.risk_level:<8} │")
        print(f"│ 💰 แนะนำเดิมพัน: {opp.stake_recommendation:.1%} ของเงินทุน {'':>25} │")
        print(f"│ {'':>70} │")
        print(f"│ 🧠 เหตุผลเชิงลึก: {'':>52} │")
        
        # แบ่งเหตุผลเป็นบรรทัด
        reasoning_parts = opp.detailed_reasoning.split(" | ")
        for part in reasoning_parts[:3]:
            if len(part) > 65:
                part = part[:62] + "..."
            print(f"│    • {part:<63} │")
        
        print(f"│ {'':>70} │")
        
        # แสดงสถิติสนับสนุน
        if opp.supporting_stats:
            print(f"│ 📈 สถิติสนับสนุน: {'':>52} │")
            if 'star_player' in opp.supporting_stats:
                print(f"│    • ผู้เล่นดาวเด่น: {opp.supporting_stats['star_player']:<45} │")
            if 'avg_rating' in opp.supporting_stats:
                print(f"│    • Rating เฉลี่ย: {opp.supporting_stats['avg_rating']:.2f}<{'':<43} │")
        
        print(f"└{'─' * 70}┘")
        print()
    
    # สรุปคำแนะนำ
    print("🎯 สรุปคำแนะนำการเดิมพัน:")
    high_confidence_bets = [opp for opp in opportunities if opp.confidence_level in ["HIGH", "VERY_HIGH"]]
    
    if high_confidence_bets:
        print(f"   ✅ มีการเดิมพันความมั่นใจสูง {len(high_confidence_bets)} รายการ")
        total_stake = sum(opp.stake_recommendation for opp in high_confidence_bets[:3])
        print(f"   💰 แนะนำใช้เงินทุนรวม {total_stake:.1%} สำหรับ 3 อันดับแรก")
    else:
        print("   ⚠️  ไม่มีการเดิมพันความมั่นใจสูงในขณะนี้")
    
    print(f"   ⚠️  คำเตือน: การพนันมีความเสี่ยง เดิมพันเท่าที่สามารถเสียได้เท่านั้น")


async def analyze_vitality_vs_m80():
    """วิเคราะห์แมตช์ Vitality vs M80 แบบเชิงลึก"""
    
    print_header()
    
    print("\n🔍 กำลังวิเคราะห์แมตช์: Team Vitality vs M80")
    print("📊 ใช้ข้อมูลจริงจาก HLTV และตลาดเดิมพัน...")
    
    # เริ่มต้นตัววิเคราะห์
    team_analyzer = get_enhanced_analyzer()
    betting_analyzer = get_deep_betting_analyzer()
    
    try:
        # วิเคราะห์ทีม
        print("\n⏳ กำลังวิเคราะห์ข้อมูลทีม...")
        vitality_analysis = await team_analyzer.analyze_team("Vitality")
        m80_analysis = await team_analyzer.analyze_team("M80")
        
        # แสดงการวิเคราะห์ทีม
        print_team_analysis("Team Vitality", vitality_analysis, is_favorite=True)
        print_team_analysis("M80", m80_analysis, is_favorite=False)
        
        # วิเคราะห์การเดิมพัน
        print("\n⏳ กำลังวิเคราะห์โอกาสการเดิมพัน...")
        match_analysis = await betting_analyzer.analyze_match_deep("Vitality", "M80")
        
        # แสดงผลการวิเคราะห์การเดิมพัน
        print_betting_analysis(match_analysis["betting_opportunities"], "Vitality", "M80")
        
        # สรุปการทำนาย
        print("\n" + "=" * 90)
        print("🎯 สรุปการทำนายแมตช์")
        print("=" * 90)
        
        predicted_winner = match_analysis["predicted_winner"]
        win_prob = match_analysis["win_probability"]
        
        print(f"🏆 ผู้ชนะที่คาดการณ์: {predicted_winner}")
        print(f"📊 ความน่าจะเป็น: {win_prob:.1%}")
        print()
        
        print("🔑 ปัจจัยสำคัญที่ต้องจับตา:")
        for i, factor in enumerate(match_analysis["key_factors"], 1):
            print(f"   {i}. {factor}")
        
        print()
        print("💡 คำแนะนำสำหรับนักเดิมพัน:")
        print("   • Vitality เป็นทีมเต็งที่แข็งแกร่งมาก แต่ราคาต่ำมาก")
        print("   • M80 มีโอกาสน้อย แต่ราคาสูงมาก หากเกิด upset จะได้กำไรมหาศาล")
        print("   • แนะนำดู handicap และ total maps มากกว่า match winner")
        
        # ข้อมูลเพิ่มเติม
        print("\n" + "=" * 90)
        print("📋 ข้อมูลเพิ่มเติมจากตลาด")
        print("=" * 90)
        print("💰 ราคาปัจจุบัน:")
        print("   • Vitality ชนะ: 1.01 (99% implied probability)")
        print("   • M80 ชนะ: 15.28 (6.5% implied probability)")
        print("   • Vitality -1.5: 1.19 (84% implied probability)")
        print("   • M80 +1.5: 4.43 (23% implied probability)")
        print("   • Over 2.5 maps: 4.74 (21% implied probability)")
        print("   • Under 2.5 maps: 1.17 (85% implied probability)")
        
        print("\n⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
        print("=" * 90)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        logging.error(f"Error in analysis: {e}")


async def main():
    """ฟังก์ชันหลัก"""
    logger = setup_logger("enhanced_predictions_thai")
    
    try:
        await analyze_vitality_vs_m80()
        
    except KeyboardInterrupt:
        print("\n\n[หยุด] การดำเนินการถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n[ข้อผิดพลาด] เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดใน main: {e}")
        raise


if __name__ == "__main__":
    print("🚀 เริ่มต้นระบบวิเคราะห์ CS2 เชิงลึก - Created by KoJao")
    asyncio.run(main())
