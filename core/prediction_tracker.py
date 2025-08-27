#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Tracker - ระบบติดตามและเรียนรู้จากผลการวิเคราะห์
เป้าหมาย: Win Rate 80%+ และเรียนรู้จากความผิดพลาด
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3

# Windows console encoding fix
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

@dataclass
class PredictionRecord:
    """บันทึกการทำนาย"""
    id: str
    timestamp: str
    match: str
    tournament: str
    prediction_type: str  # "match_winner", "map_winner", "series_score", "player_performance"
    predicted_outcome: str
    actual_outcome: Optional[str]
    confidence: float  # 0.0 - 1.0
    edge_percentage: float
    kelly_fraction: float
    reasoning: List[str]
    data_sources: List[str]
    is_correct: Optional[bool]
    profit_loss: Optional[float]
    lessons_learned: List[str]

@dataclass
class AnalysisMetrics:
    """เมตริกการวิเคราะห์"""
    total_predictions: int
    correct_predictions: int
    win_rate: float
    total_profit: float
    roi: float
    avg_confidence: float
    best_prediction_type: str
    worst_prediction_type: str
    improvement_areas: List[str]

class PredictionTracker:
    def __init__(self):
        self.db_path = "data/prediction_tracker.db"
        self.json_backup_path = "data/prediction_history.json"
        self.init_database()
        
    def init_database(self):
        """สร้างฐานข้อมูล"""
        os.makedirs("data", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                match TEXT,
                tournament TEXT,
                prediction_type TEXT,
                predicted_outcome TEXT,
                actual_outcome TEXT,
                confidence REAL,
                edge_percentage REAL,
                kelly_fraction REAL,
                reasoning TEXT,
                data_sources TEXT,
                is_correct INTEGER,
                profit_loss REAL,
                lessons_learned TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_description TEXT,
                success_rate REAL,
                sample_size INTEGER,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_prediction(self, prediction: PredictionRecord):
        """เพิ่มการทำนายใหม่"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.id,
            prediction.timestamp,
            prediction.match,
            prediction.tournament,
            prediction.prediction_type,
            prediction.predicted_outcome,
            prediction.actual_outcome,
            prediction.confidence,
            prediction.edge_percentage,
            prediction.kelly_fraction,
            json.dumps(prediction.reasoning, ensure_ascii=False),
            json.dumps(prediction.data_sources, ensure_ascii=False),
            prediction.is_correct,
            prediction.profit_loss,
            json.dumps(prediction.lessons_learned, ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
        
        # Backup to JSON
        self.backup_to_json()
        
    def update_prediction_result(self, prediction_id: str, actual_outcome: str, 
                               profit_loss: float, lessons_learned: List[str] = None):
        """อัปเดตผลการทำนาย"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get original prediction
        cursor.execute('SELECT predicted_outcome FROM predictions WHERE id = ?', (prediction_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"❌ ไม่พบการทำนาย ID: {prediction_id}")
            conn.close()
            return
            
        predicted_outcome = result[0]
        is_correct = 1 if predicted_outcome.lower() == actual_outcome.lower() else 0
        
        lessons = lessons_learned or []
        if not is_correct:
            lessons.append(f"ทำนายผิด: คาด {predicted_outcome} แต่ได้ {actual_outcome}")
            
        cursor.execute('''
            UPDATE predictions 
            SET actual_outcome = ?, is_correct = ?, profit_loss = ?, lessons_learned = ?
            WHERE id = ?
        ''', (actual_outcome, is_correct, profit_loss, json.dumps(lessons, ensure_ascii=False), prediction_id))
        
        conn.commit()
        conn.close()
        
        # Learn from this result
        self.learn_from_prediction(prediction_id)
        self.backup_to_json()
        
    def learn_from_prediction(self, prediction_id: str):
        """เรียนรู้จากการทำนาย"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_type, is_correct, edge_percentage, confidence, reasoning
            FROM predictions WHERE id = ?
        ''', (prediction_id,))
        
        result = cursor.fetchone()
        if not result:
            conn.close()
            return
            
        pred_type, is_correct, edge, confidence, reasoning_json = result
        reasoning = json.loads(reasoning_json) if reasoning_json else []
        
        # Update learning patterns
        for reason in reasoning:
            cursor.execute('''
                SELECT success_rate, sample_size FROM learning_patterns 
                WHERE pattern_type = ? AND pattern_description = ?
            ''', (pred_type, reason))
            
            pattern_result = cursor.fetchone()
            
            if pattern_result:
                old_success_rate, old_sample_size = pattern_result
                new_sample_size = old_sample_size + 1
                new_success_rate = ((old_success_rate * old_sample_size) + is_correct) / new_sample_size
                
                cursor.execute('''
                    UPDATE learning_patterns 
                    SET success_rate = ?, sample_size = ?, last_updated = ?
                    WHERE pattern_type = ? AND pattern_description = ?
                ''', (new_success_rate, new_sample_size, datetime.now().isoformat(), pred_type, reason))
            else:
                cursor.execute('''
                    INSERT INTO learning_patterns (pattern_type, pattern_description, success_rate, sample_size, last_updated)
                    VALUES (?, ?, ?, 1, ?)
                ''', (pred_type, reason, float(is_correct), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
    def get_analysis_metrics(self) -> AnalysisMetrics:
        """คำนวณเมตริกการวิเคราะห์"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE actual_outcome IS NOT NULL')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_correct = 1')
        correct_predictions = cursor.fetchone()[0]
        
        win_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        cursor.execute('SELECT SUM(profit_loss) FROM predictions WHERE profit_loss IS NOT NULL')
        total_profit = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(confidence) FROM predictions WHERE confidence IS NOT NULL')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Best/worst prediction types
        cursor.execute('''
            SELECT prediction_type, 
                   COUNT(*) as total,
                   SUM(is_correct) as correct,
                   (SUM(is_correct) * 100.0 / COUNT(*)) as win_rate
            FROM predictions 
            WHERE actual_outcome IS NOT NULL
            GROUP BY prediction_type
            ORDER BY win_rate DESC
        ''')
        
        type_stats = cursor.fetchall()
        best_type = type_stats[0][0] if type_stats else "N/A"
        worst_type = type_stats[-1][0] if type_stats else "N/A"
        
        # Improvement areas
        cursor.execute('''
            SELECT pattern_description, success_rate, sample_size
            FROM learning_patterns
            WHERE success_rate < 0.6 AND sample_size >= 3
            ORDER BY success_rate ASC
            LIMIT 5
        ''')
        
        weak_patterns = cursor.fetchall()
        improvement_areas = [f"{pattern[0]} ({pattern[1]:.1%} จาก {pattern[2]} ครั้ง)" 
                           for pattern in weak_patterns]
        
        conn.close()
        
        return AnalysisMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            win_rate=win_rate,
            total_profit=total_profit,
            roi=(total_profit / 10000 * 100) if total_profit != 0 else 0,  # Assume 10k bankroll
            avg_confidence=avg_confidence,
            best_prediction_type=best_type,
            worst_prediction_type=worst_type,
            improvement_areas=improvement_areas
        )
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """ดึงข้อมูลการเรียนรู้"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Top performing patterns
        cursor.execute('''
            SELECT pattern_type, pattern_description, success_rate, sample_size
            FROM learning_patterns
            WHERE sample_size >= 3
            ORDER BY success_rate DESC
            LIMIT 10
        ''')
        
        top_patterns = cursor.fetchall()
        
        # Recent mistakes
        cursor.execute('''
            SELECT match, predicted_outcome, actual_outcome, lessons_learned, timestamp
            FROM predictions
            WHERE is_correct = 0 AND actual_outcome IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 5
        ''')
        
        recent_mistakes = cursor.fetchall()
        
        conn.close()
        
        return {
            "top_patterns": [
                {
                    "type": p[0],
                    "description": p[1],
                    "success_rate": p[2],
                    "sample_size": p[3]
                } for p in top_patterns
            ],
            "recent_mistakes": [
                {
                    "match": m[0],
                    "predicted": m[1],
                    "actual": m[2],
                    "lessons": json.loads(m[3]) if m[3] else [],
                    "timestamp": m[4]
                } for m in recent_mistakes
            ]
        }
        
    def backup_to_json(self):
        """สำรองข้อมูลเป็น JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions')
        predictions = cursor.fetchall()
        
        cursor.execute('SELECT * FROM learning_patterns')
        patterns = cursor.fetchall()
        
        conn.close()
        
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "learning_patterns": patterns
        }
        
        with open(self.json_backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
    def generate_performance_report(self):
        """สร้างรายงานประสิทธิภาพ"""
        metrics = self.get_analysis_metrics()
        insights = self.get_learning_insights()
        
        print("="*80)
        print("📊 รายงานประสิทธิภาพการวิเคราะห์")
        print("="*80)
        print(f"⏰ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("🎯 เมตริกหลัก:")
        print(f"   📈 การทำนายทั้งหมด: {metrics.total_predictions}")
        print(f"   ✅ ทำนายถูก: {metrics.correct_predictions}")
        print(f"   🏆 Win Rate: {metrics.win_rate:.1f}% {'🟢' if metrics.win_rate >= 80 else '🟡' if metrics.win_rate >= 60 else '🔴'}")
        print(f"   💰 กำไรรวม: {metrics.total_profit:,.0f} บาท")
        print(f"   📊 ROI: {metrics.roi:.1f}%")
        print(f"   🎲 ความมั่นใจเฉลี่ย: {metrics.avg_confidence:.1%}")
        print()
        
        print("🏅 ประเภทการทำนาย:")
        print(f"   🥇 ดีที่สุด: {metrics.best_prediction_type}")
        print(f"   🥉 แย่ที่สุด: {metrics.worst_prediction_type}")
        print()
        
        if metrics.improvement_areas:
            print("⚠️ จุดที่ต้องปรับปรุง:")
            for i, area in enumerate(metrics.improvement_areas, 1):
                print(f"   {i}. {area}")
            print()
        
        print("🧠 รูปแบบที่ประสบความสำเร็จ:")
        for pattern in insights["top_patterns"][:5]:
            print(f"   ✅ {pattern['description']}")
            print(f"      Success Rate: {pattern['success_rate']:.1%} ({pattern['sample_size']} ตัวอย่าง)")
        print()
        
        if insights["recent_mistakes"]:
            print("❌ ความผิดพลาดล่าสุด:")
            for mistake in insights["recent_mistakes"]:
                print(f"   • {mistake['match']}: คาด {mistake['predicted']} แต่ได้ {mistake['actual']}")
                if mistake['lessons']:
                    for lesson in mistake['lessons']:
                        print(f"     📝 {lesson}")
            print()
        
        # Recommendations
        print("💡 คำแนะนำการปรับปรุง:")
        if metrics.win_rate < 80:
            print("   🎯 เป้าหมาย: เพิ่ม Win Rate ให้ถึง 80%+")
            print("   📚 เน้นเรียนรู้จากรูปแบบที่ประสบความสำเร็จ")
            print("   ⚠️ หลีกเลี่ยงรูปแบบที่มี Success Rate ต่ำ")
        else:
            print("   🏆 ยอดเยี่ยม! Win Rate เกิน 80% แล้ว")
            print("   🚀 เน้นเพิ่ม ROI และลดความเสี่ยง")
            
        print("   🔄 อัปเดตข้อมูลผลการแข่งอย่างสม่ำเสมอ")
        print("   📊 วิเคราะห์ pattern ใหม่ๆ ที่เกิดขึ้น")
        print("="*80)

# Helper functions for easy integration
def create_match_prediction(match: str, tournament: str, predicted_winner: str, 
                          confidence: float, edge: float, reasoning: List[str]) -> str:
    """สร้างการทำนายแมตช์"""
    tracker = PredictionTracker()
    
    prediction_id = f"{match.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    prediction = PredictionRecord(
        id=prediction_id,
        timestamp=datetime.now().isoformat(),
        match=match,
        tournament=tournament,
        prediction_type="match_winner",
        predicted_outcome=predicted_winner,
        actual_outcome=None,
        confidence=confidence,
        edge_percentage=edge,
        kelly_fraction=edge / 100 * confidence,  # Simplified Kelly
        reasoning=reasoning,
        data_sources=["HLTV", "Team Stats", "Recent Form"],
        is_correct=None,
        profit_loss=None,
        lessons_learned=[]
    )
    
    tracker.add_prediction(prediction)
    return prediction_id

def update_match_result(prediction_id: str, actual_winner: str, profit_loss: float):
    """อัปเดตผลแมตช์"""
    tracker = PredictionTracker()
    tracker.update_prediction_result(prediction_id, actual_winner, profit_loss)

if __name__ == "__main__":
    tracker = PredictionTracker()
    tracker.generate_performance_report()
