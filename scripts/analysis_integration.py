#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Integration - รวมระบบวิเคราะห์กับ Prediction Tracker
เชื่อมต่อการวิเคราะห์ทั้งหมดเข้ากับระบบติดตามผล
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from prediction_tracker import PredictionTracker, PredictionRecord, create_match_prediction, update_match_result

# Windows console encoding fix
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class AnalysisIntegration:
    def __init__(self):
        self.tracker = PredictionTracker()
        
    def record_vp_gl_predictions(self):
        """บันทึกการทำนาย VP vs GL"""
        
        # Map 1 - Dust2 (ทำนายถูกแล้ว)
        map1_id = create_match_prediction(
            match="VP vs GL - Map 1 Dust2",
            tournament="BLAST Open London 2025",
            predicted_winner="GamerLegion",
            confidence=0.74,
            edge=31.0,
            reasoning=[
                "GL มี win rate 74% บน Dust2",
                "VP มี win rate เพียง 43% บน Dust2", 
                "Edge 31% เอียงไป GL อย่างชัดเจน",
                "Dust2 เป็นแมพที่ GL เก่งที่สุด"
            ]
        )
        
        # อัปเดตผล Map 1
        update_match_result(map1_id, "GamerLegion", 100.0)  # สมมติกำไร 100 บาท
        
        # Map 2 - Ancient (ทำนายใหม่)
        map2_id = create_match_prediction(
            match="VP vs GL - Map 2 Ancient",
            tournament="BLAST Open London 2025", 
            predicted_winner="Virtus.pro",
            confidence=0.71,
            edge=26.0,
            reasoning=[
                "VP มี win rate 71% บน Ancient",
                "GL มี win rate เพียง 45% บน Ancient",
                "Edge 26% เอียงไป VP",
                "electroNic เก่ง split execute บน Ancient",
                "Ancient เป็นแมพที่ VP เก่งที่สุด"
            ]
        )
        
        # Map 3 - Train (ทำนายเผื่อไว้)
        map3_id = create_match_prediction(
            match="VP vs GL - Map 3 Train",
            tournament="BLAST Open London 2025",
            predicted_winner="GamerLegion", 
            confidence=0.52,
            edge=4.0,
            reasoning=[
                "GL มี win rate 62% บน Train",
                "VP มี win rate 58% บน Train",
                "Edge เพียง 4% ใกล้เคียงมาก",
                "GL มีโมเมนตัมจาก Map 1",
                "Train เป็นแมพกลาง ขึ้นอยู่กับจิตวิทยา"
            ]
        )
        
        # Series Winner
        series_id = create_match_prediction(
            match="VP vs GL - Series Winner",
            tournament="BLAST Open London 2025",
            predicted_winner="GamerLegion",
            confidence=0.55,
            edge=10.0,
            reasoning=[
                "GL นำ 1-0 จาก Map 1",
                "GL มีโมเมนตัมและความมั่นใจ",
                "หาก VP ชนะ Ancient จะไป Train 50-50",
                "หาก GL ชนะ Ancient จะจบ 2-0",
                "โอกาสรวม GL เอาชนะ VP ได้"
            ]
        )
        
        return {
            "map1_id": map1_id,
            "map2_id": map2_id, 
            "map3_id": map3_id,
            "series_id": series_id
        }
        
    def record_blast_predictions(self):
        """บันทึกการทำนายแมตช์อื่นๆ ใน BLAST"""
        
        predictions = []
        
        # FaZe vs ECSTATIC
        faze_id = create_match_prediction(
            match="FaZe vs ECSTATIC",
            tournament="BLAST Open London 2025",
            predicted_winner="FaZe",
            confidence=0.90,
            edge=59.7,
            reasoning=[
                "FaZe อันดับ 9 vs ECSTATIC อันดับ 36",
                "ความแตกต่างระดับชัดเจน",
                "FaZe มีประสบการณ์และทีมเวิร์กดีกว่า",
                "ECSTATIC เป็นทีมใหม่ยังไม่แน่นอน",
                "Kelly Criterion แนะนำ 59.7%"
            ]
        )
        predictions.append(("faze_ecstatic", faze_id))
        
        # NAVI vs fnatic
        navi_id = create_match_prediction(
            match="NAVI vs fnatic", 
            tournament="BLAST Open London 2025",
            predicted_winner="Natus Vincere",
            confidence=0.90,
            edge=53.0,
            reasoning=[
                "NAVI อันดับ 6 vs fnatic อันดับ 34",
                "NAVI มี s1mple และ electronic",
                "fnatic อยู่ในช่วงฟื้นฟูทีม",
                "NAVI เป็นเต็งแรงที่ปลอดภัย",
                "Kelly Criterion แนะนำ 53.0%"
            ]
        )
        predictions.append(("navi_fnatic", navi_id))
        
        return predictions
        
    def update_live_results(self, match_results: Dict[str, str]):
        """อัปเดตผลแมตช์แบบ Live"""
        
        for prediction_id, result in match_results.items():
            if "win" in result.lower():
                winner = result.split(" win")[0]
                profit = 50.0 if "correct" in result.lower() else -50.0
                update_match_result(prediction_id, winner, profit)
                
    def analyze_prediction_patterns(self):
        """วิเคราะห์รูปแบบการทำนาย"""
        
        insights = self.tracker.get_learning_insights()
        metrics = self.tracker.get_analysis_metrics()
        
        print("🧠 การวิเคราะห์รูปแบบการทำนาย:")
        print("="*60)
        
        print("📊 รูปแบบที่ประสบความสำเร็จ:")
        for pattern in insights["top_patterns"][:3]:
            success_rate = pattern["success_rate"] * 100
            print(f"   ✅ {pattern['description']}")
            print(f"      Success Rate: {success_rate:.1f}% ({pattern['sample_size']} ครั้ง)")
            
        print("\n❌ รูปแบบที่ต้องปรับปรุง:")
        for mistake in insights["recent_mistakes"][:3]:
            print(f"   • {mistake['match']}")
            print(f"     คาด: {mistake['predicted']} | จริง: {mistake['actual']}")
            
        print(f"\n🎯 Win Rate ปัจจุบัน: {metrics.win_rate:.1f}%")
        if metrics.win_rate < 80:
            print("⚠️ ต้องปรับปรุงให้ถึง 80%+")
        else:
            print("🏆 เป้าหมาย 80%+ บรรลุแล้ว!")
            
    def generate_improvement_plan(self):
        """สร้างแผนการปรับปรุง"""
        
        metrics = self.tracker.get_analysis_metrics()
        insights = self.tracker.get_learning_insights()
        
        print("\n📋 แผนการปรับปรุงการวิเคราะห์:")
        print("="*60)
        
        if metrics.win_rate < 80:
            print("🎯 เป้าหมายหลัก: เพิ่ม Win Rate จาก {:.1f}% เป็น 80%+".format(metrics.win_rate))
            print("\n📚 กลยุทธ์การปรับปรุง:")
            print("   1. เน้นใช้รูปแบบที่ประสบความสำเร็จ")
            print("   2. หลีกเลี่ยงรูปแบบที่มี Success Rate ต่ำ")
            print("   3. เพิ่มข้อมูลสถิติที่แม่นยำมากขึ้น")
            print("   4. วิเคราะห์ momentum และจิตวิทยาทีม")
            print("   5. ติดตาม meta game และการเปลี่ยนแปลง")
            
        print("\n🔄 การเรียนรู้อย่างต่อเนื่อง:")
        print("   • อัปเดตผลแมตช์ทุกครั้งที่แข่งจบ")
        print("   • วิเคราะห์สาเหตุการทำนายผิด")
        print("   • ปรับปรุงโมเดลการคำนวณ Edge")
        print("   • เพิ่มข้อมูล live stats และ in-game factors")
        
        print("\n⚡ การปรับปรุงเร่งด่วน:")
        for area in metrics.improvement_areas[:3]:
            print(f"   🔧 {area}")
            
    def run_full_integration(self):
        """รันการรวมระบบทั้งหมด"""
        
        print("🚀 เริ่มการรวมระบบวิเคราะห์กับ Prediction Tracker")
        print("="*80)
        
        # บันทึกการทำนาย VP vs GL
        print("📝 บันทึกการทำนาย VP vs GL...")
        vp_gl_predictions = self.record_vp_gl_predictions()
        print(f"✅ บันทึกแล้ว {len(vp_gl_predictions)} การทำนาย")
        
        # บันทึกการทำนายแมตช์อื่น
        print("\n📝 บันทึกการทำนายแมตช์อื่นๆ...")
        blast_predictions = self.record_blast_predictions()
        print(f"✅ บันทึกแล้ว {len(blast_predictions)} การทำนาย")
        
        # สร้างรายงานประสิทธิภาพ
        print("\n📊 สร้างรายงานประสิทธิภาพ...")
        self.tracker.generate_performance_report()
        
        # วิเคราะห์รูปแบบ
        print("\n🧠 วิเคราะห์รูปแบบการทำนาย...")
        self.analyze_prediction_patterns()
        
        # สร้างแผนปรับปรุง
        self.generate_improvement_plan()
        
        print("\n✅ การรวมระบบเสร็จสมบูรณ์!")
        print("📁 ข้อมูลถูกเก็บไว้ที่: data/prediction_tracker.db")
        print("💾 สำรองข้อมูลที่: data/prediction_history.json")
        
        return {
            "vp_gl_predictions": vp_gl_predictions,
            "blast_predictions": blast_predictions,
            "status": "completed"
        }

if __name__ == "__main__":
    integration = AnalysisIntegration()
    integration.run_full_integration()
