#!/usr/bin/env python3
"""
แสดงการทำนาย - แสดงการทำนายแมตช์ CS2 วันนี้พร้อมการวิเคราะห์การเดิมพันเชิงลึก
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

from core.upcoming_matches_predictor import get_predictor
from core.advanced_betting_analyzer import get_betting_analyzer
from core.real_odds_scraper import update_matches_with_real_odds
from app.utils.logger import setup_logger


def print_header():
    """พิมพ์หัวข้อพร้อมเวลาปัจจุบัน"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print("🎯 ระบบทำนายแมตช์ CS2 - BLAST Open London 2025")
    print(f"📅 สร้างเมื่อ: {current_time}")
    print("⚡ Created by KoJao")
    print("=" * 80)


def print_match_card(i: int, match: Dict[str, Any]) -> None:
    """พิมพ์การ์ดแมตช์พร้อมการวิเคราะห์การเดิมพัน"""
    print(f"\n+-- แมตช์ {i} {'-' * (65 - len(str(i)))}+")
    print(f"| [ทีม] {match['team1']} vs {match['team2']:<40} |")
    print(f"| [เวลา] {match['scheduled_time']:<48} |")
    print(f"| [ราคา] {match['odds']:<48} |")
    
    if match['predicted_winner'] != 'No prediction yet':
        confidence_val = float(match['confidence'].rstrip('%'))
        confidence_indicator = "[สูง]" if confidence_val > 70 else "[กลาง]" if confidence_val > 55 else "[ต่ำ]"
        print(f"| {confidence_indicator} การทำนาย: {match['predicted_winner']} ({match['confidence']}) {'':>15} |")
        print(f"| [ข้อมูล] {match['reasoning'][:45]:<45} |")
        if len(match['reasoning']) > 45:
            remaining = match['reasoning'][45:]
            while remaining:
                chunk = remaining[:45]
                remaining = remaining[45:]
                print(f"|         {chunk:<45} |")
    else:
        print(f"| [รอ] การทำนาย: กำลังสร้าง...{'':<30} |")
    
    # แสดงคำแนะนำการเดิมพันถ้ามี
    if 'betting_recommendations' in match and match['betting_recommendations']:
        print(f"| {'':>66} |")
        print(f"| [เดิมพัน] คำแนะนำยอดนิยม: {'':>32} |")
        for j, rec in enumerate(match['betting_recommendations'][:3], 1):
            bet_line = f"{j}. {rec['selection']} @{rec['odds']} ({rec['confidence_level']})"
            print(f"| {bet_line:<64} |")
            ev_line = f"   EV: {rec['expected_value']:.1%} | เสี่ยง: {rec['stake_recommendation']:.1%}"
            print(f"| {ev_line:<64} |")
    
    print(f"+{'-' * 66}+")


def print_summary_section(summary: Dict[str, Any]) -> None:
    """พิมพ์ส่วนสรุป"""
    print("\n" + "=" * 80)
    print("📈 สรุปการทำนาย")
    print("=" * 80)
    
    print(f"📊 แมตช์ทั้งหมดวันนี้: {summary['total_upcoming_matches']}")
    print(f"🎯 การทำนายที่สร้าง: {summary['total_predictions_made']}")
    print(f"⏳ แมตช์ที่รอผล: {summary['pending_matches']}")
    
    if summary['completed_matches'] > 0:
        print(f"✅ แมตช์ที่เสร็จแล้ว: {summary['completed_matches']}")
        print(f"🎯 อัตราความแม่นยำ: {summary['accuracy_rate']}")
        print(f"✅ การทำนายที่ถูก: {summary['correct_predictions']}")
        
        if summary['recent_results']:
            print("\n🏆 ผลล่าสุด:")
            for result in summary['recent_results'][-3:]:
                status = "[ชนะ]" if result.get('prediction_correct') else "[แพ้]"
                print(f"   {status} {result['teams']} - ผู้ชนะ: {result['winner']}")


async def print_betting_summary(matches: List[Dict[str, Any]]) -> None:
    """พิมพ์สรุปการเดิมพันแบบครอบคลุม"""
    print("\n" + "=" * 80)
    print("💰 สรุปการวิเคราะห์การเดิมพันเชิงลึก")
    print("=" * 80)
    
    all_recommendations = []
    for match in matches:
        if 'betting_recommendations' in match:
            all_recommendations.extend(match['betting_recommendations'])
    
    if not all_recommendations:
        print("[ข้อมูล] ไม่พบโอกาสการเดิมพันที่มีกำไร")
        return
    
    # เรียงตาม expected value
    all_recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
    
    print(f"[สถิติ] โอกาสการเดิมพันที่วิเคราะห์: {len(all_recommendations)}")
    
    # แสดง 3 คำแนะนำยอดนิยมจากทุกแมตช์
    print("\n[เดิมพันดี] โอกาสที่มี Expected Value สูงสุด:")
    for i, rec in enumerate(all_recommendations[:3], 1):
        print(f"  {i}. {rec['selection']} @{rec['odds']} - EV: {rec['expected_value']:.1%} ({rec['confidence_level']})")
        print(f"     ความเสี่ยง: {rec['risk_level']} | แนะนำเดิมพัน: {rec['stake_recommendation']:.1%} ของเงินทุน")
        print(f"     เหตุผล: {rec['reasoning'][:60]}...")
        print()
    
    # แยกประเภทการเดิมพัน
    bet_types = {}
    for rec in all_recommendations:
        bet_type = rec['selection'].split()[0] if 'Over' in rec['selection'] or 'Under' in rec['selection'] else 'แมตช์/แฮนดิแคป'
        bet_types[bet_type] = bet_types.get(bet_type, 0) + 1
    
    print("[แยกประเภท] โอกาสตามประเภทการเดิมพัน:")
    for bet_type, count in bet_types.items():
        print(f"  {bet_type}: {count} โอกาส")
    
    # สรุประดับความเสี่ยง
    risk_levels = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    for rec in all_recommendations:
        risk_levels[rec['risk_level']] += 1
    
    print("\n[ความเสี่ยง] การกระจายความเสี่ยง:")
    risk_thai = {'LOW': 'ต่ำ', 'MEDIUM': 'กลาง', 'HIGH': 'สูง'}
    for risk, count in risk_levels.items():
        print(f"  ความเสี่ยง{risk_thai[risk]}: {count} การเดิมพัน")


def print_footer() -> None:
    """พิมพ์ส่วนท้ายพร้อมคำแนะนำ"""
    print("\n" + "=" * 80)
    print("📝 ขั้นตอนต่อไป")
    print("=" * 80)
    print("🔄 รีเฟรชการทำนาย: python scripts/show_predictions_thai.py")
    print("📊 รายงานผลแมตช์: python scripts/report_result.py")
    print("📈 ดูการวิเคราะห์รายละเอียด: python scripts/daily_report_generator.py")
    print("⚠️ คำเตือน: การพนันมีความเสี่ยง เดิมพันเท่าที่สามารถเสียได้เท่านั้น")
    print("⚡ Created by KoJao - ระบบวิเคราะห์ CS2 ชั้นนำ")
    print("=" * 80 + "\n")


async def save_predictions_snapshot(matches: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """บันทึกสแนปช็อตการทำนายปัจจุบัน"""
    try:
        snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = snapshot_dir / f"predictions_snapshot_thai_{timestamp}.json"
        
        # สกัดข้อมูลการเดิมพันสำหรับสแนปช็อต
        betting_summary = {
            "total_opportunities": sum(len(m.get('betting_recommendations', [])) for m in matches),
            "high_confidence_bets": sum(1 for m in matches for r in m.get('betting_recommendations', []) if r.get('confidence_level') == 'HIGH'),
            "avg_expected_value": sum(r.get('expected_value', 0) for m in matches for r in m.get('betting_recommendations', [])) / max(1, sum(len(m.get('betting_recommendations', [])) for m in matches))
        }
        
        # ทำความสะอาดข้อมูลแมตช์สำหรับ JSON serialization
        clean_matches = []
        for match in matches:
            clean_match = match.copy()
            # ลบ real_odds_data ที่ไม่สามารถ serialize ได้
            if 'real_odds_data' in clean_match:
                del clean_match['real_odds_data']
            clean_matches.append(clean_match)
        
        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "matches": clean_matches,
            "summary": summary,
            "betting_analysis": betting_summary,
            "created_by": "KoJao",
            "language": "thai"
        }
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
        print(f"[บันทึก] บันทึกสแนปช็อต: {snapshot_file.name}")
    except Exception as e:
        print(f"[คำเตือน] ไม่สามารถบันทึกสแนปช็อต - {e}")


async def main():
    """ฟังก์ชันหลักพร้อมการแสดงการทำนายและการวิเคราะห์การเดิมพันแบบครอบคลุม"""
    # ตั้งค่า logging
    logger = setup_logger("show_predictions_thai")
    
    try:
        # เริ่มต้นระบบ
        print("[เริ่ม] กำลังเริ่มต้นระบบทำนาย CS2...")
        predictor = get_predictor()
        betting_analyzer = get_betting_analyzer()
        
        # พิมพ์หัวข้อ
        print_header()
        
        # สร้างการทำนายพร้อมแสดงความคืบหน้า
        print("\n[AI] กำลังสร้างการทำนาย...")
        try:
            await predictor.generate_predictions()
            print("[สำเร็จ] สร้างการทำนายเสร็จสิ้น!")
        except Exception as e:
            print(f"[ข้อผิดพลาด] เกิดข้อผิดพลาดในการสร้างการทำนาย: {e}")
            logger.error(f"การสร้างการทำนายล้มเหลว: {e}")
            return
        
        # ดึงแมตช์พร้อมการทำนาย
        try:
            matches = await predictor.get_upcoming_matches_with_predictions()
            if not matches:
                print("\n[ข้อมูล] ไม่พบแมตช์ที่จะมาถึงสำหรับวันนี้")
                return
        except Exception as e:
            print(f"[ข้อผิดพลาด] เกิดข้อผิดพลาดในการดึงแมตช์: {e}")
            logger.error(f"การดึงแมตช์ล้มเหลว: {e}")
            return
        
        # อัปเดตแมตช์ด้วยข้อมูลราคาจริง
        print("\n[ราคา] กำลังดึงข้อมูลราคาแบบเรียลไทม์...")
        try:
            matches = await update_matches_with_real_odds(matches)
            updated_count = sum(1 for m in matches if m.get('odds_updated', False))
            print(f"[สำเร็จ] อัปเดต {updated_count}/{len(matches)} แมตช์ด้วยราคาจริง!")
        except Exception as e:
            print(f"[คำเตือน] ไม่สามารถอัปเดตด้วยราคาจริง: {e}")
            logger.warning(f"การอัปเดตราคาจริงล้มเหลว: {e}")
        
        # สร้างการวิเคราะห์การเดิมพันสำหรับแต่ละแมตช์
        print("\n[เดิมพัน] กำลังวิเคราะห์โอกาสการเดิมพัน...")
        try:
            for match in matches:
                match_id = f"blast_london_{match['team1'].lower()}_{match['team2'].lower()}"
                betting_recs = await betting_analyzer.analyze_betting_opportunities(
                    match_id, match['team1'], match['team2']
                )
                match['betting_recommendations'] = [{
                    'selection': rec.selection,
                    'odds': rec.odds,
                    'expected_value': rec.expected_value,
                    'confidence_level': rec.confidence_level,
                    'stake_recommendation': rec.stake_recommendation,
                    'reasoning': rec.reasoning,
                    'risk_level': rec.risk_level
                } for rec in betting_recs]
            print("[สำเร็จ] การวิเคราะห์การเดิมพันเสร็จสิ้น!")
        except Exception as e:
            print(f"[คำเตือน] การวิเคราะห์การเดิมพันล้มเหลว: {e}")
            logger.warning(f"การวิเคราะห์การเดิมพันล้มเหลว: {e}")
            # ดำเนินการต่อโดยไม่มีการวิเคราะห์การเดิมพัน
            for match in matches:
                match['betting_recommendations'] = []
        
        # แสดงแมตช์พร้อมการวิเคราะห์การเดิมพัน
        print(f"\n[แมตช์] แมตช์วันนี้พร้อมราคาจริงและการวิเคราะห์การเดิมพัน ({len(matches)} แมตช์)")
        for i, match in enumerate(matches, 1):
            print_match_card(i, match)
        
        # แสดงสรุปการเดิมพัน
        await print_betting_summary(matches)
        
        # ดึงและแสดงสรุป
        try:
            summary = await predictor.get_prediction_summary()
            print_summary_section(summary)
        except Exception as e:
            print(f"[คำเตือน] ไม่สามารถสร้างสรุป - {e}")
            logger.warning(f"การสร้างสรุปล้มเหลว: {e}")
            summary = {}
        
        # บันทึกสแนปช็อตที่ปรับปรุงแล้ว
        await save_predictions_snapshot(matches, summary)
        
        # พิมพ์ส่วนท้าย
        print_footer()
        
        logger.info(f"แสดงการทำนายและการวิเคราะห์การเดิมพันสำหรับ {len(matches)} แมตช์เสร็จสิ้น")
        
    except KeyboardInterrupt:
        print("\n\n[หยุด] การดำเนินการถูกยกเลิกโดยผู้ใช้")
    except Exception as e:
        print(f"\n[ข้อผิดพลาด] เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดใน main: {e}")
        raise


if __name__ == "__main__":
    print("🚀 เริ่มต้นระบบทำนาย CS2 ภาษาไทย - Created by KoJao")
    asyncio.run(main())
