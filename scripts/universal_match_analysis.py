#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Match Analysis Script - สคริปต์วิเคราะห์แมตช์สากล
ใช้งาน: python universal_match_analysis.py "Team1" "Team2"
"""

import sys
import asyncio
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent))

from core.universal_match_analyzer import UniversalMatchAnalyzer, AnalysisDepth

async def main():
    """ฟังก์ชันหลักสำหรับรันการวิเคราะห์"""
    
    print("🌟 ระบบวิเคราะห์แมตช์สากล - Universal Match Analysis Engine")
    print("=" * 80)
    
    if len(sys.argv) < 3:
        print("📋 วิธีใช้งาน:")
        print("python universal_match_analysis.py \"Team1\" \"Team2\" [depth]")
        print("\nตัวอย่าง:")
        print("python universal_match_analysis.py \"GamerLegion\" \"Virtus.pro\"")
        print("python universal_match_analysis.py \"Natus Vincere\" \"FaZe Clan\" \"professional\"")
        print("\nระดับการวิเคราะห์:")
        print("• basic - วิเคราะห์พื้นฐาน")
        print("• advanced - วิเคราะห์ขั้นสูง") 
        print("• world_class - วิเคราะห์ระดับโลก (default)")
        print("• professional - วิเคราะห์ระดับมืออาชีพ")
        return
    
    team1 = sys.argv[1]
    team2 = sys.argv[2]
    depth = sys.argv[3] if len(sys.argv) > 3 else "world_class"
    
    try:
        depth_enum = AnalysisDepth(depth)
    except ValueError:
        print(f"❌ ระดับการวิเคราะห์ '{depth}' ไม่ถูกต้อง")
        print("ใช้: basic, advanced, world_class, professional")
        return
    
    print(f"🎯 กำลังวิเคราะห์: {team1} vs {team2}")
    print(f"📊 ระดับการวิเคราะห์: {depth}")
    print("⏳ กรุณารอสักครู่...")
    print()
    
    try:
        analyzer = UniversalMatchAnalyzer()
        analysis = await analyzer.analyze_any_match(team1, team2, depth_enum)
        
        print("\n✅ การวิเคราะห์เสร็จสมบูรณ์!")
        print(f"📈 ความมั่นใจในการทำนาย: {analysis.prediction_confidence:.1%}")
        
        # แสดงสรุปคำแนะนำ
        if analysis.betting_recommendations.get("recommended_team") != "แมตช์สมดุล - เดิมพันตลาดรอง":
            recommended = analysis.betting_recommendations.get("recommended_team")
            edge = analysis.betting_recommendations.get("edge_difference", 0)
            kelly = analysis.betting_recommendations.get("kelly_percentage", 0)
            
            print(f"\n🎯 ทีมที่แนะนำ: {recommended}")
            print(f"📊 Edge: {edge:.3f}")
            print(f"💰 Kelly Criterion: {kelly:.1%} ของ bankroll")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("🔄 กรุณาลองใหม่อีกครั้ง")

if __name__ == "__main__":
    asyncio.run(main())
