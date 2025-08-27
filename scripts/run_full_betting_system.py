#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete CS2 Betting System Runner
ระบบเดิมพัน CS2 แบบครบวงจร - รันทุกระบบพร้อมกัน
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

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

from core.automated_betting_pipeline import AutomatedBettingPipeline
from core.odds_scraper import MultiSourceOddsScraper
from core.universal_match_analyzer import UniversalMatchAnalyzer, AnalysisDepth

def print_system_banner():
    """แสดง banner ของระบบ"""
    print("🎯" + "=" * 78 + "🎯")
    print("🚀          CS2 BETTING INTELLIGENCE SYSTEM - ระบบเดิมพัน CS2          🚀")
    print("🎯" + "=" * 78 + "🎯")
    print("📊 ระบบวิเคราะห์และแนะนำการเดิมพัน CS2 แบบอัตโนมัติ")
    print("💰 ดึงราคาจากเว็บพนันหลายแห่ง + วิเคราะห์เชิงลึก + แนะนำการเดิมพัน")
    print("🧠 AI-Powered Analysis: Psychology + Tactics + Form + Value Betting")
    print("=" * 80)

async def run_complete_system():
    """รันระบบเดิมพันแบบครบวงจร"""
    
    print_system_banner()
    
    print("🔄 เริ่มต้นระบบวิเคราะห์การเดิมพัน CS2...")
    
    try:
        # Step 1: รันระบบหลัก
        print("\n📡 Step 1: ดึงข้อมูลแมตช์และวิเคราะห์")
        pipeline = AutomatedBettingPipeline()
        recommendations = await pipeline.run_full_pipeline()
        
        print(f"\n✅ ได้คำแนะนำการเดิมพันทั้งหมด: {len(recommendations)} แมตช์")
        
        # Step 2: บันทึกผลลัพธ์
        await save_system_results(recommendations)
        
        # Step 3: แสดงสรุปขั้นสุดท้าย
        display_final_summary(recommendations)
        
        return recommendations
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในระบบ: {e}")
        return []

async def save_system_results(recommendations):
    """บันทึกผลลัพธ์ของระบบ"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # สร้างโฟลเดอร์ results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # บันทึกคำแนะนำการเดิมพัน
    recommendations_file = results_dir / f"betting_recommendations_{timestamp}.json"
    
    recommendations_data = []
    for rec in recommendations:
        rec_dict = {
            "match_id": rec.match_id,
            "primary_bet": rec.primary_bet,
            "backup_bets": rec.backup_bets,
            "confidence_level": rec.confidence_level,
            "reasoning": rec.reasoning,
            "risk_level": rec.risk_level,
            "kelly_percentage": rec.kelly_percentage,
            "expected_value": rec.expected_value,
            "timestamp": timestamp
        }
        recommendations_data.append(rec_dict)
    
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 บันทึกคำแนะนำแล้ว: {recommendations_file}")

def display_final_summary(recommendations):
    """แสดงสรุปขั้นสุดท้าย"""
    
    print("\n" + "🎯" + "=" * 78 + "🎯")
    print("📋                    สรุปผลการวิเคราะห์ระบบ CS2 BETTING                    📋")
    print("🎯" + "=" * 78 + "🎯")
    
    if not recommendations:
        print("❌ ไม่พบคำแนะนำการเดิมพัน")
        return
    
    # สถิติรวม
    total_matches = len(recommendations)
    high_confidence = len([r for r in recommendations if r.confidence_level > 0.8])
    positive_ev = len([r for r in recommendations if r.expected_value > 0])
    low_risk = len([r for r in recommendations if r.risk_level == "low"])
    
    print(f"📊 สถิติรวม:")
    print(f"   🎯 แมตช์ทั้งหมด: {total_matches}")
    print(f"   🔥 ความมั่นใจสูง (>80%): {high_confidence}")
    print(f"   💰 Expected Value บวก: {positive_ev}")
    print(f"   🟢 ความเสี่ยงต่ำ: {low_risk}")
    
    # แนะนำ Top 3
    sorted_recs = sorted(recommendations, key=lambda x: x.expected_value, reverse=True)
    
    print(f"\n🏆 TOP 3 คำแนะนำการเดิมพันที่ดีที่สุด:")
    
    for i, rec in enumerate(sorted_recs[:3], 1):
        risk_emoji = "🔴" if rec.risk_level == "high" else "🟡" if rec.risk_level == "medium" else "🟢"
        
        print(f"\n   {i}. {rec.primary_bet.get('match', 'Unknown Match')}")
        print(f"      🎯 เดิมพัน: {rec.primary_bet.get('selection', 'Unknown')} @ {rec.primary_bet.get('odds', 0):.2f}")
        print(f"      📈 EV: {rec.expected_value:.3f} | Kelly: {rec.kelly_percentage:.1f}% | {risk_emoji} {rec.risk_level}")
        print(f"      💡 เหตุผล: {rec.reasoning[0] if rec.reasoning else 'ไม่ระบุ'}")
    
    # คำแนะนำการจัดการเงิน
    print(f"\n💼 คำแนะนำการจัดการเงิน:")
    print(f"   • ใช้ Kelly Criterion สำหรับการคำนวณเดิมพัน")
    print(f"   • เดิมพันไม่เกิน 2-5% ของเงินทุนต่อแมตช์")
    print(f"   • เน้นแมตช์ที่มี Expected Value บวก")
    print(f"   • หลีกเลี่ยงแมตช์ความเสี่ยงสูงถ้าเป็นมือใหม่")
    
    print("\n🎯" + "=" * 78 + "🎯")
    print("✅ ระบบวิเคราะห์เสร็จสมบูรณ์ - พร้อมใช้งาน!")
    print("🎯" + "=" * 78 + "🎯")

async def run_quick_analysis(team1: str, team2: str):
    """รันการวิเคราะห์แบบเร็วสำหรับแมตช์เดียว"""
    
    print(f"⚡ การวิเคราะห์แบบเร็ว: {team1} vs {team2}")
    print("-" * 50)
    
    analyzer = UniversalMatchAnalyzer()
    analysis = await analyzer.analyze_any_match(team1, team2, AnalysisDepth.WORLD_CLASS)
    
    print(f"🧠 ความมั่นใจ: {analysis.prediction_confidence:.1%}")
    print(f"🎯 แนะนำ: {analysis.betting_recommendations.get('recommended_team', 'ไม่ระบุ')}")
    
    # จำลองราคาต่อรอง
    print(f"💰 ราคาประมาณ: 1.85 / 1.95")
    
    return analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CS2 Betting System")
    parser.add_argument("--quick", nargs=2, metavar=("TEAM1", "TEAM2"), 
                       help="วิเคราะห์แบบเร็วสำหรับ 2 ทีม")
    parser.add_argument("--full", action="store_true", 
                       help="รันระบบแบบเต็มรูปแบบ")
    
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(run_quick_analysis(args.quick[0], args.quick[1]))
    elif args.full:
        asyncio.run(run_complete_system())
    else:
        # รันระบบแบบเต็มรูปแบบเป็นค่าเริ่มต้น
        asyncio.run(run_complete_system())
