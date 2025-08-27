#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLAST Open London 2025 - การวิเคราะห์ 3 แมตช์หลักเชิงลึก
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
    print("🚀 ระบบวิเคราะห์ CS2 เชิงลึก - Created by KoJao")
    print()
    print("=" * 100)
    print("🎯 การวิเคราะห์ 3 แมตช์หลัก BLAST Open London 2025")
    print(f"📅 วันที่: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
    print("=" * 100)

def print_match_header(match_num: int, team1: str, team2: str, time: str):
    """แสดงหัวข้อแมตช์"""
    print(f"\n{'🔥' * 50}")
    print(f"🎮 แมตช์ที่ {match_num}: {team1} vs {team2}")
    print(f"⏰ เวลา: {time}")
    print(f"{'🔥' * 50}")

async def analyze_match_comprehensive(team1: str, team2: str, match_time: str, favorite_team: str = None):
    """วิเคราะห์แมตช์แบบครบถ้วน"""
    
    # เริ่มต้นตัววิเคราะห์
    team_analyzer = get_enhanced_analyzer()
    betting_analyzer = get_deep_betting_analyzer()
    
    try:
        # วิเคราะห์ทีม
        print(f"\n⏳ กำลังวิเคราะห์ข้อมูลทีม {team1} vs {team2}...")
        team1_analysis = await team_analyzer.analyze_team(team1)
        team2_analysis = await team_analyzer.analyze_team(team2)
        
        # แสดงการเปรียบเทียบทีม
        print_team_comparison(team1, team2, team1_analysis, team2_analysis, favorite_team)
        
        # วิเคราะห์การเดิมพัน
        print(f"\n⏳ กำลังวิเคราะห์โอกาสการเดิมพัน...")
        match_analysis = await betting_analyzer.analyze_match_deep(team1, team2)
        
        # แสดงผลการวิเคราะห์การเดิมพัน
        print_betting_opportunities(match_analysis["betting_opportunities"], team1, team2)
        
        # สรุปการทำนาย
        prediction = match_analysis["prediction"]
        print_match_prediction(prediction, team1, team2)
        
        return match_analysis
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์: {e}")
        return None

def print_team_comparison(team1: str, team2: str, analysis1, analysis2, favorite_team: str = None):
    """แสดงการเปรียบเทียบทีม"""
    
    print(f"\n📊 การเปรียบเทียบทีม")
    print("+" + "-" * 98 + "+")
    print(f"| {'ข้อมูล':<20} | {team1:<35} | {team2:<35} |")
    print("+" + "-" * 98 + "+")
    print(f"| {'🏅 อันดับโลก':<20} | #{analysis1.current_ranking:<34} | #{analysis2.current_ranking:<34} |")
    print(f"| {'🔥 ฟอร์มล่าสุด':<20} | {analysis1.recent_form:<35} | {analysis2.recent_form:<35} |")
    print(f"| {'📈 ชนะติดต่อกัน':<20} | {analysis1.win_streak} แมตช์{'':<28} | {analysis2.win_streak} แมตช์{'':<28} |")
    print(f"| {'🎯 สไตล์การเล่น':<20} | {analysis1.tactical_style:<35} | {analysis2.tactical_style:<35} |")
    print("+" + "-" * 98 + "+")
    
    # แสดงผู้เล่นดาวเด่น
    print(f"\n👥 ผู้เล่นดาวเด่น:")
    print("+" + "-" * 98 + "+")
    print(f"| {'ลำดับ':<8} | {team1 + ' Players':<42} | {team2 + ' Players':<42} |")
    print("+" + "-" * 98 + "+")
    
    # เรียงผู้เล่นตาม rating
    players1 = sorted(analysis1.players, key=lambda x: x.rating, reverse=True)[:3]
    players2 = sorted(analysis2.players, key=lambda x: x.rating, reverse=True)[:3]
    
    for i in range(3):
        p1 = players1[i] if i < len(players1) else None
        p2 = players2[i] if i < len(players2) else None
        
        p1_text = f"{p1.name} ({p1.rating:.2f})" if p1 else ""
        p2_text = f"{p2.name} ({p2.rating:.2f})" if p2 else ""
        
        print(f"| {i+1:<8} | {p1_text:<42} | {p2_text:<42} |")
    
    print("+" + "-" * 98 + "+")
    
    # แสดงแมพที่แข็งแกร่ง
    print(f"\n🗺️  แมพที่แข็งแกร่ง:")
    print("+" + "-" * 98 + "+")
    print(f"| {'ลำดับ':<8} | {team1 + ' Maps':<42} | {team2 + ' Maps':<42} |")
    print("+" + "-" * 98 + "+")
    
    maps1 = sorted(analysis1.map_pool, key=lambda x: x.win_rate, reverse=True)[:3]
    maps2 = sorted(analysis2.map_pool, key=lambda x: x.win_rate, reverse=True)[:3]
    
    for i in range(3):
        m1 = maps1[i] if i < len(maps1) else None
        m2 = maps2[i] if i < len(maps2) else None
        
        m1_text = f"{m1.map_name} ({m1.win_rate:.1f}%)" if m1 else ""
        m2_text = f"{m2.map_name} ({m2.win_rate:.1f}%)" if m2 else ""
        
        print(f"| {i+1:<8} | {m1_text:<42} | {m2_text:<42} |")
    
    print("+" + "-" * 98 + "+")
    
    # แสดงจุดแข็ง-จุดอ่อน
    print(f"\n✅ จุดแข็ง vs ❌ จุดอ่อน:")
    print("+" + "-" * 98 + "+")
    print(f"| {'ประเภท':<12} | {team1:<42} | {team2:<42} |")
    print("+" + "-" * 98 + "+")
    
    for i in range(max(len(analysis1.strengths), len(analysis2.strengths))):
        s1 = analysis1.strengths[i] if i < len(analysis1.strengths) else ""
        s2 = analysis2.strengths[i] if i < len(analysis2.strengths) else ""
        label = "✅ จุดแข็ง" if i == 0 else ""
        print(f"| {label:<12} | {s1:<42} | {s2:<42} |")
    
    print("+" + "-" * 98 + "+")
    
    for i in range(max(len(analysis1.weaknesses), len(analysis2.weaknesses))):
        w1 = analysis1.weaknesses[i] if i < len(analysis1.weaknesses) else ""
        w2 = analysis2.weaknesses[i] if i < len(analysis2.weaknesses) else ""
        label = "❌ จุดอ่อน" if i == 0 else ""
        print(f"| {label:<12} | {w1:<42} | {w2:<42} |")
    
    print("+" + "-" * 98 + "+")

def print_betting_opportunities(opportunities, team1: str, team2: str):
    """แสดงโอกาสการเดิมพัน"""
    
    print(f"\n💰 โอกาสการเดิมพันที่ดีที่สุด")
    print("=" * 100)
    
    if not opportunities:
        print("❌ ไม่พบโอกาสการเดิมพันที่มีกำไรในขณะนี้")
        return
    
    # แสดงข้อเสนอที่ดีที่สุด
    best_opportunity = max(opportunities, key=lambda x: x.expected_value)
    print(f"🏆 ข้อเสนอที่ดีที่สุด: {best_opportunity.selection} @{best_opportunity.odds}")
    print(f"💎 Expected Value: {best_opportunity.expected_value:.1%} | ความเสี่ยง: {best_opportunity.risk_level}")
    print(f"💰 แนะนำเดิมพัน: {best_opportunity.stake_recommendation:.1%} ของเงินทุน")
    
    print(f"\n🎯 โอกาสการเดิมพันทั้งหมด {len(opportunities)} รายการ:")
    
    # แสดงรายละเอียดแต่ละโอกาส
    confidence_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    
    for i, opp in enumerate(opportunities, 1):
        print(f"\n┌─ อันดับ {i}: {opp.selection} ─{'-' * (70 - len(opp.selection))}┐")
        print(f"│ 💵 ราคา: {opp.odds:<10} │ 📊 EV: {opp.expected_value:.1%} │ {confidence_emoji[opp.confidence_level]} {opp.confidence_level} │ {risk_emoji[opp.risk_level]} {opp.risk_level} │")
        print(f"│ 💰 แนะนำเดิมพัน: {opp.stake_recommendation:.1%} ของเงินทุน {'':>35} │")
        print(f"│ {'':>78} │")
        print(f"│ 🧠 เหตุผลเชิงลึก: {'':>60} │")
        
        # แบ่งเหตุผลเป็นบรรทัด
        reasoning_parts = opp.detailed_reasoning.split(" | ")
        for j, part in enumerate(reasoning_parts[:4], 1):
            if len(part) > 75:
                part = part[:72] + "..."
            print(f"│ {j}. {part:<75} │")
        
        print("└" + "─" * 78 + "┘")

def print_match_prediction(prediction, team1: str, team2: str):
    """แสดงการทำนายแมตช์"""
    
    print(f"\n🎯 การทำนายแมตช์")
    print("=" * 100)
    print(f"🏆 ผู้ชนะที่คาดการณ์: {prediction.predicted_winner}")
    print(f"📊 ความน่าจะเป็น: {prediction.confidence:.1%}")
    
    print(f"\n🔑 ปัจจัยสำคัญที่ต้องจับตา:")
    for i, reason in enumerate(prediction.reasoning[:3], 1):
        print(f"   {i}. {reason}")

async def main():
    """วิเคราะห์ 3 แมตช์หลัก BLAST Open London 2025"""
    
    logger = setup_logger("triple_match_analysis")
    
    try:
        print_header()
        
        # แมตช์ที่ 1: Virtus.pro vs GamerLegion
        print_match_header(1, "Virtus.pro", "GamerLegion", "20:30 เวลาไทย (1:30 PM BST)")
        match1_result = await analyze_match_comprehensive("Virtus.pro", "GamerLegion", "20:30 เวลาไทย")
        
        # แมตช์ที่ 2: ECSTATIC vs FaZe
        print_match_header(2, "ECSTATIC", "FaZe", "23:00 เวลาไทย (4:00 PM BST)")
        match2_result = await analyze_match_comprehensive("ECSTATIC", "FaZe", "23:00 เวลาไทย", "FaZe")
        
        # แมตช์ที่ 3: NAVI vs Fnatic
        print_match_header(3, "NAVI", "Fnatic", "01:30 เวลาไทย (6:30 PM BST)")
        match3_result = await analyze_match_comprehensive("NAVI", "Fnatic", "01:30 เวลาไทย", "NAVI")
        
        # สรุปรวม
        print(f"\n" + "=" * 100)
        print("📊 สรุปการวิเคราะห์ทั้ง 3 แมตช์")
        print("=" * 100)
        
        print("🏆 แมตช์ที่น่าสนใจที่สุด:")
        print("   1. 🔥 Virtus.pro vs GamerLegion - แมตช์สมดุล, โอกาสเดิมพัน handicap ดี")
        print("   2. 🎯 ECSTATIC vs FaZe - jcobbb debut, FaZe เต็งแต่ ECSTATIC มี upset potential")
        print("   3. ⚡ NAVI vs Fnatic - NAVI ควรชนะ แต่ Fnatic มีประสบการณ์")
        
        print("\n💰 สรุปโอกาสการเดิมพันที่ดีที่สุด:")
        print("   🥇 Virtus.pro vs GamerLegion: ดู handicap และ total maps")
        print("   🥈 ECSTATIC upset vs FaZe: ราคาดี หาก ECSTATIC เล่นได้ดี")
        print("   🥉 NAVI -1.5 vs Fnatic: ปลอดภัย แต่ราคาอาจต่ำ")
        
        print("\n🎯 คำแนะนำการเดิมพันรวม:")
        print("   • แบ่งเงินทุน 3 ส่วน สำหรับ 3 แมตช์")
        print("   • เน้น handicap และ total maps มากกว่า match winner")
        print("   • ระวัง upset ในแมตช์ ECSTATIC vs FaZe")
        print("   • NAVI vs Fnatic อาจจบเร็ว (Under maps)")
        
        print(f"\n⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
        print("=" * 100)
        
    except KeyboardInterrupt:
        print("\n\n[หยุด] การดำเนินการถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n[ข้อผิดพลาด] เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดใน main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
