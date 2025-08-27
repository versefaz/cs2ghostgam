#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Odds Terminal - ดึงราคาจริงจากเว็บพนันชั้นนำ (ไม่ใช่ Mock Up)
Created by KoJao - Professional CS2 Betting Analytics
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
import json

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.real_odds_api import get_real_odds_api, analyze_real_odds

def print_header():
    """แสดงหัวข้อ"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀" * 60)
    print("💎 REAL ODDS TERMINAL - LIVE BETTING DATA 💎")
    print("⚡ Created by KoJao - Professional CS2 Analytics ⚡")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🎯 BLAST Open London 2025")
    print("🚀" * 60)
    print()

def print_odds_summary(analysis):
    """แสดงสรุปราคาที่ดึงมา"""
    
    print("📊 REAL ODDS SUMMARY")
    print("=" * 80)
    print(f"🔍 Total Odds Found: {analysis['total_odds_found']}")
    print(f"🏪 Bookmakers Available: {', '.join(analysis['bookmakers_available'])}")
    print(f"🎮 Matches Found: {len(analysis['matches_found'])}")
    print(f"💰 Best Value Opportunities: {len(analysis['best_values'])}")
    print()
    
    if analysis['matches_found']:
        print("🎯 Matches Available:")
        for i, match in enumerate(analysis['matches_found'], 1):
            print(f"   {i}. {match}")
        print()

def print_best_values(best_values):
    """แสดงราคาที่คุ้มค่าที่สุด"""
    
    if not best_values:
        print("❌ ไม่พบโอกาสเดิมพันที่มี Value ในขณะนี้")
        print("💡 ลองใหม่ในอีก 15-30 นาที")
        return
    
    print("💰 BEST VALUE OPPORTUNITIES (REAL ODDS)")
    print("=" * 120)
    
    for i, value in enumerate(best_values[:10], 1):  # แสดงแค่ 10 อันดับแรก
        value_emoji = "🟢" if value['value_percentage'] > 10 else "🟡" if value['value_percentage'] > 5 else "🔴"
        
        print(f"🏆 RANK #{i}")
        print(f"┌─ {value['match']} ─────────────────────────────────────────────────────────┐")
        print(f"│ 🎯 Bet Type: {value['bet_type']:<20} │ 🎲 Selection: {value['selection']:<25} │")
        print(f"│ 💵 Best Odds: {value['best_odds']:<8.2f} @ {value['best_bookmaker']:<12} │ 📊 Avg Odds: {value['average_odds']:<8.2f} │")
        print(f"│ {value_emoji} Value: {value['value_percentage']:+6.2f}% │ 🏪 Sources: {value['num_bookmakers']} bookmakers │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        print()

def print_detailed_odds(all_odds):
    """แสดงราคาทั้งหมดแบบละเอียด"""
    
    print("📋 DETAILED ODDS BREAKDOWN")
    print("=" * 120)
    
    for bookmaker, odds_list in all_odds.items():
        if not odds_list:
            continue
            
        print(f"🏪 {bookmaker.upper()} ({len(odds_list)} odds)")
        print("┌" + "─" * 118 + "┐")
        print("│ Match                          │ Bet Type          │ Selection               │ Odds   │ Time     │")
        print("├" + "─" * 118 + "┤")
        
        for odd in odds_list[:15]:  # แสดงแค่ 15 รายการแรก
            match_short = odd['match'][:30] if len(odd['match']) > 30 else odd['match']
            bet_type_short = odd['bet_type'][:17] if len(odd['bet_type']) > 17 else odd['bet_type']
            selection_short = odd['selection'][:23] if len(odd['selection']) > 23 else odd['selection']
            time_short = odd['timestamp'][11:16]  # แค่ HH:MM
            
            print(f"│ {match_short:<30} │ {bet_type_short:<17} │ {selection_short:<23} │ {odd['odds']:<6.2f} │ {time_short:<8} │")
        
        if len(odds_list) > 15:
            print(f"│ ... และอีก {len(odds_list) - 15} รายการ" + " " * 75 + "│")
        
        print("└" + "─" * 118 + "┘")
        print()

def calculate_recommended_bets(best_values):
    """คำนวณการเดิมพันที่แนะนำ"""
    
    if not best_values:
        return []
    
    recommendations = []
    
    for value in best_values[:5]:  # แค่ 5 อันดับแรก
        # คำนวณขนาดเดิมพัน
        if value['value_percentage'] > 15:
            stake = "5-6%"
            risk = "MEDIUM"
        elif value['value_percentage'] > 10:
            stake = "3-4%"
            risk = "LOW"
        elif value['value_percentage'] > 5:
            stake = "2-3%"
            risk = "LOW"
        else:
            stake = "1-2%"
            risk = "HIGH"
        
        # สร้างเหตุผล
        reasoning = f"Value {value['value_percentage']:.1f}% from {value['num_bookmakers']} sources"
        
        recommendations.append({
            'match': value['match'],
            'bet_type': value['bet_type'],
            'selection': value['selection'],
            'odds': value['best_odds'],
            'bookmaker': value['best_bookmaker'],
            'value': value['value_percentage'],
            'stake': stake,
            'risk': risk,
            'reasoning': reasoning
        })
    
    return recommendations

def print_betting_recommendations(recommendations):
    """แสดงคำแนะนำการเดิมพัน"""
    
    if not recommendations:
        return
    
    print("🎯 BETTING RECOMMENDATIONS (BASED ON REAL ODDS)")
    print("=" * 120)
    
    total_stake = 0
    
    for i, rec in enumerate(recommendations, 1):
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        stake_percent = float(rec['stake'].split('-')[0].rstrip('%'))
        total_stake += stake_percent
        
        print(f"💰 RECOMMENDATION #{i}")
        print(f"┌─ {rec['match']} ─────────────────────────────────────────────────────────┐")
        print(f"│ 🎯 {rec['bet_type']}: {rec['selection']:<50} │")
        print(f"│ 💵 Odds: {rec['odds']:.2f} @ {rec['bookmaker']:<15} │ 📊 Value: +{rec['value']:.1f}% │")
        print(f"│ 💰 Stake: {rec['stake']} of bankroll │ {risk_emoji[rec['risk']]} Risk: {rec['risk']:<6} │")
        print(f"│ 🧠 Reason: {rec['reasoning']:<60} │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        print()
    
    print(f"📊 TOTAL RECOMMENDED STAKE: {total_stake:.0f}% of bankroll")
    print()

async def main():
    """เริ่มต้นแอป"""
    
    print_header()
    
    print("🔄 CONNECTING TO REAL BETTING SITES...")
    print("⏳ This may take 30-60 seconds to fetch live odds...")
    print()
    
    try:
        # ดึงราคาจริง
        async with get_real_odds_api() as api:
            real_odds = await api.get_real_odds()
        
        if not real_odds or not any(real_odds.values()):
            print("❌ ไม่สามารถดึงราคาจริงได้ในขณะนี้")
            print()
            print("🔄 สาเหตุที่เป็นไปได้:")
            print("   • แมตช์ยังไม่เปิดให้เดิมพัน (เหลือ 2+ ชั่วโมง)")
            print("   • เว็บพนันบล็อกการเข้าถึงจาก IP นี้")
            print("   • ปัญหาการเชื่อมต่อเครือข่าย")
            print("   • แมตช์ถูกเลื่อนหรือยกเลิก")
            print()
            print("💡 แนะนำ:")
            print("   • ลองใหม่ในอีก 30 นาที")
            print("   • ใช้ VPN หากจำเป็น")
            print("   • ตรวจสอบราคาด้วยตนเองที่เว็บโดยตรง")
            return
        
        # วิเคราะห์ราคา
        analysis = analyze_real_odds(real_odds)
        
        # แสดงผล
        print_odds_summary(analysis)
        print_best_values(analysis['best_values'])
        
        # คำนวณคำแนะนำ
        recommendations = calculate_recommended_bets(analysis['best_values'])
        print_betting_recommendations(recommendations)
        
        # แสดงราคาละเอียด
        show_details = input("แสดงราคาทั้งหมดแบบละเอียด? (y/n): ").strip().lower()
        if show_details == 'y':
            print()
            print_detailed_odds(real_odds)
        
        # บันทึกข้อมูล
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/real_odds_{timestamp}.json"
        
        os.makedirs("data", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'raw_odds': real_odds,
                'analysis': analysis,
                'recommendations': recommendations
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 ข้อมูลถูกบันทึกที่: {filename}")
        print()
        print("🕐 ข้อมูลอัปเดตล่าสุด:", datetime.now().strftime('%H:%M:%S'))
        print("🔄 รีเฟรชทุก 10-15 นาทีสำหรับราคาล่าสุด")
        
    except KeyboardInterrupt:
        print("\n👋 หยุดการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("🔄 ลองใหม่หรือตรวจสอบการเชื่อมต่อ")

if __name__ == "__main__":
    asyncio.run(main())
