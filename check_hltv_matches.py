#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ตรวจสอบแมตช์ CS2 วันนี้จาก HLTV
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent))

from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper

async def main():
    """ตรวจสอบแมตช์ CS2 วันนี้"""
    print("🔥 แมตช์ CS2 วันนี้ จาก HLTV:")
    print("=" * 80)
    
    try:
        async with EnhancedHLTVScraper() as scraper:
            matches = await scraper.get_upcoming_matches(limit=20)
            
            if not matches:
                print("❌ ไม่พบแมตช์ที่กำลังจะมาถึง")
                return
            
            for i, match in enumerate(matches, 1):
                team1 = match.get("team1", "TBD")
                team2 = match.get("team2", "TBD") 
                time = match.get("time", "TBD")
                event = match.get("event", "Unknown Event")
                
                print(f"{i:2d}. {team1:>15} vs {team2:<15} | {time:>10} | {event}")
            
            print("=" * 80)
            print(f"📊 รวม {len(matches)} แมตช์")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    asyncio.run(main())
