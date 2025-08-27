#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Match Analysis Engine - World-Class CS2 Match Analysis System
ระบบวิเคราะห์แมตช์ CS2 ระดับโลก ที่สามารถวิเคราะห์คู่ใดก็ได้โดยอัตโนมัติ
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# เพิ่ม path สำหรับ import
sys.path.append(str(Path(__file__).parent.parent))

from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper
from core.enhanced_team_analyzer import EnhancedTeamAnalyzer
from core.deep_betting_analyzer import DeepBettingAnalyzer

class AnalysisDepth(Enum):
    """ระดับความลึกของการวิเคราะห์"""
    BASIC = "basic"
    ADVANCED = "advanced"
    WORLD_CLASS = "world_class"
    PROFESSIONAL = "professional"

@dataclass
class PlayerPsychology:
    """การวิเคราะห์จิตวิทยาผู้เล่น"""
    pressure_resistance: float  # ความทนต่อความกดดัน (0-1)
    clutch_mentality: float     # ความสามารถในสถานการณ์วิกฤต
    consistency_factor: float   # ความสม่ำเสมอ
    confidence_level: float     # ระดับความมั่นใจ
    recent_form_trend: str      # แนวโน้มฟอร์มล่าสุด

@dataclass
class TacticalPattern:
    """รูปแบบการเล่นเชิงยุทธศาสตร์"""
    default_setup: str          # การจัดวางพื้นฐาน
    retake_efficiency: float    # ประสิทธิภาพการ retake
    adaptation_ability: float   # ความสามารถในการปรับตัว
    rotation_speed: float       # ความเร็วในการหมุนตำแหน่ง

@dataclass
class WorldClassAnalysis:
    """การวิเคราะห์ระดับโลก"""
    team1_psychology: PlayerPsychology
    team2_psychology: PlayerPsychology
    team1_tactics: TacticalPattern
    team2_tactics: TacticalPattern
    prediction_confidence: float
    betting_recommendations: Dict[str, Any]

class UniversalMatchAnalyzer:
    """ระบบวิเคราะห์แมตช์สากล - ระดับโลก"""
    
    def __init__(self):
        self.team_analyzer = EnhancedTeamAnalyzer()
        self.betting_analyzer = DeepBettingAnalyzer()
        
        # ฐานข้อมูลทีมระดับโลก
        self.world_class_teams = {
            "Natus Vincere": {"tier": "S", "region": "CIS", "style": "aggressive"},
            "FaZe Clan": {"tier": "S", "region": "EU", "style": "individual_skill"},
            "Astralis": {"tier": "A", "region": "EU", "style": "tactical"},
            "Vitality": {"tier": "S", "region": "EU", "style": "star_player"},
            "G2 Esports": {"tier": "A", "region": "EU", "style": "versatile"},
            "ENCE": {"tier": "A", "region": "EU", "style": "disciplined"},
            "Cloud9": {"tier": "A", "region": "NA", "style": "momentum_based"},
            "Liquid": {"tier": "A", "region": "NA", "style": "structured"},
            "Virtus.pro": {"tier": "B", "region": "CIS", "style": "unpredictable"},
            "GamerLegion": {"tier": "B", "region": "EU", "style": "teamwork"},
            "ECSTATIC": {"tier": "C", "region": "EU", "style": "upset_potential"},
            "fnatic": {"tier": "B", "region": "EU", "style": "experience"},
            "M80": {"tier": "C", "region": "NA", "style": "aggressive"}
        }

    async def analyze_any_match(self, team1: str, team2: str, 
                               depth: AnalysisDepth = AnalysisDepth.WORLD_CLASS) -> WorldClassAnalysis:
        """วิเคราะห์แมตช์ใดก็ได้ในระดับที่กำหนด"""
        
        print(f"🔬 เริ่มการวิเคราะห์ระดับ {depth.value}: {team1} vs {team2}")
        print("=" * 80)
        
        # ดึงข้อมูลทีมแบบ real-time
        team1_data = await self._get_comprehensive_team_data(team1)
        team2_data = await self._get_comprehensive_team_data(team2)
        
        # วิเคราะห์จิตวิทยาผู้เล่น
        team1_psychology = self._analyze_player_psychology(team1, team1_data)
        team2_psychology = self._analyze_player_psychology(team2, team2_data)
        
        # วิเคราะห์รูปแบบยุทธศาสตร์
        team1_tactics = self._analyze_tactical_patterns(team1, team1_data)
        team2_tactics = self._analyze_tactical_patterns(team2, team2_data)
        
        # คำนวณความมั่นใจและคำแนะนำ
        prediction_confidence = self._calculate_prediction_confidence(team1_data, team2_data)
        betting_recommendations = await self._generate_betting_recommendations(
            team1, team2, team1_psychology, team2_psychology, team1_tactics, team2_tactics
        )
        
        analysis = WorldClassAnalysis(
            team1_psychology=team1_psychology,
            team2_psychology=team2_psychology,
            team1_tactics=team1_tactics,
            team2_tactics=team2_tactics,
            prediction_confidence=prediction_confidence,
            betting_recommendations=betting_recommendations
        )
        
        # แสดงผลการวิเคราะห์
        await self._display_world_class_analysis(team1, team2, analysis)
        
        return analysis

    async def _get_comprehensive_team_data(self, team_name: str) -> Dict[str, Any]:
        """ดึงข้อมูลทีมแบบครอบคลุม"""
        try:
            # ใช้ข้อมูลจาก enhanced team analyzer
            team_data = await self.team_analyzer.get_team_analysis(team_name)
            
            # เพิ่มข้อมูลเชิงลึก
            enhanced_data = {
                **team_data,
                "tier_info": self.world_class_teams.get(team_name, {"tier": "Unknown"}),
                "tactical_style": self._determine_tactical_style(team_name),
                "recent_form": self._calculate_recent_form(team_data.get("recent_matches", [])),
            }
            
            return enhanced_data
            
        except Exception as e:
            print(f"⚠️ ใช้ข้อมูลสำรองสำหรับ {team_name}")
            return self._get_fallback_team_data(team_name)

    def _analyze_player_psychology(self, team_name: str, team_data: Dict) -> PlayerPsychology:
        """วิเคราะห์จิตวิทยาผู้เล่นและทีม"""
        
        recent_results = team_data.get("recent_matches", [])
        pressure_resistance = self._calculate_pressure_resistance(recent_results)
        clutch_mentality = self._calculate_clutch_mentality(team_data.get("player_stats", {}))
        consistency_factor = self._calculate_consistency(recent_results)
        confidence_level = self._calculate_confidence_level(team_data)
        recent_form_trend = self._determine_form_trend(recent_results)
        
        return PlayerPsychology(
            pressure_resistance=pressure_resistance,
            clutch_mentality=clutch_mentality,
            consistency_factor=consistency_factor,
            confidence_level=confidence_level,
            recent_form_trend=recent_form_trend
        )

    def _analyze_tactical_patterns(self, team_name: str, team_data: Dict) -> TacticalPattern:
        """วิเคราะห์รูปแบบยุทธศาสตร์"""
        
        style = team_data.get("tactical_style", "balanced")
        map_stats = team_data.get("map_pool", {})
        
        return TacticalPattern(
            default_setup=self._determine_default_setup(style),
            retake_efficiency=self._calculate_retake_efficiency(map_stats),
            adaptation_ability=self._calculate_adaptation_ability(team_data),
            rotation_speed=self._calculate_rotation_speed(team_data)
        )

    async def _display_world_class_analysis(self, team1: str, team2: str, 
                                          analysis: WorldClassAnalysis):
        """แสดงผลการวิเคราะห์ระดับโลก"""
        
        print(f"\n🏆 การวิเคราะห์ระดับโลก: {team1} vs {team2}")
        print("=" * 80)
        
        # จิตวิทยาผู้เล่น
        print(f"\n🧠 การวิเคราะห์จิตวิทยา")
        print(f"┌─ {team1}")
        print(f"│  💪 ความทนต่อความกดดัน: {analysis.team1_psychology.pressure_resistance:.2f}")
        print(f"│  🎯 ความสามารถ clutch: {analysis.team1_psychology.clutch_mentality:.2f}")
        print(f"│  📊 ความสม่ำเสมอ: {analysis.team1_psychology.consistency_factor:.2f}")
        print(f"│  🔥 ระดับความมั่นใจ: {analysis.team1_psychology.confidence_level:.2f}")
        print(f"└─ แนวโน้มฟอร์ม: {analysis.team1_psychology.recent_form_trend}")
        
        print(f"\n┌─ {team2}")
        print(f"│  💪 ความทนต่อความกดดัน: {analysis.team2_psychology.pressure_resistance:.2f}")
        print(f"│  🎯 ความสามารถ clutch: {analysis.team2_psychology.clutch_mentality:.2f}")
        print(f"│  📊 ความสม่ำเสมอ: {analysis.team2_psychology.consistency_factor:.2f}")
        print(f"│  🔥 ระดับความมั่นใจ: {analysis.team2_psychology.confidence_level:.2f}")
        print(f"└─ แนวโน้มฟอร์ม: {analysis.team2_psychology.recent_form_trend}")
        
        # ยุทธศาสตร์
        print(f"\n⚔️ การวิเคราะห์ยุทธศาสตร์")
        print(f"┌─ {team1}: {analysis.team1_tactics.default_setup}")
        print(f"│  🛡️ ประสิทธิภาพ retake: {analysis.team1_tactics.retake_efficiency:.2f}")
        print(f"│  🔄 ความเร็วการหมุน: {analysis.team1_tactics.rotation_speed:.2f}")
        print(f"└─ ความสามารถปรับตัว: {analysis.team1_tactics.adaptation_ability:.2f}")
        
        print(f"\n┌─ {team2}: {analysis.team2_tactics.default_setup}")
        print(f"│  🛡️ ประสิทธิภาพ retake: {analysis.team2_tactics.retake_efficiency:.2f}")
        print(f"│  🔄 ความเร็วการหมุน: {analysis.team2_tactics.rotation_speed:.2f}")
        print(f"└─ ความสามารถปรับตัว: {analysis.team2_tactics.adaptation_ability:.2f}")
        
        # ความมั่นใจในการทำนาย
        confidence_emoji = "🎯" if analysis.prediction_confidence > 0.8 else "📊" if analysis.prediction_confidence > 0.6 else "❓"
        print(f"\n{confidence_emoji} ความมั่นใจในการทำนาย: {analysis.prediction_confidence:.2f}")
        
        # คำแนะนำการเดิมพัน
        print(f"\n💰 คำแนะนำการเดิมพันระดับโลก")
        print("=" * 50)
        for key, value in analysis.betting_recommendations.items():
            if isinstance(value, (int, float)):
                print(f"• {key}: {value:.3f}")
            else:
                print(f"• {key}: {value}")

    async def _generate_betting_recommendations(self, team1: str, team2: str,
                                             t1_psych: PlayerPsychology, t2_psych: PlayerPsychology,
                                             t1_tactics: TacticalPattern, t2_tactics: TacticalPattern) -> Dict[str, Any]:
        """สร้างคำแนะนำการเดิมพันจากการวิเคราะห์"""
        
        # คำนวณ edge จากการวิเคราะห์
        team1_edge = self._calculate_team_edge(t1_psych, t1_tactics)
        team2_edge = self._calculate_team_edge(t2_psych, t2_tactics)
        
        recommendations = {}
        
        if team1_edge > team2_edge + 0.1:
            recommendations["recommended_team"] = team1
            recommendations["edge_difference"] = team1_edge - team2_edge
            recommendations["kelly_percentage"] = min((team1_edge - team2_edge) * 10, 0.05)
        elif team2_edge > team1_edge + 0.1:
            recommendations["recommended_team"] = team2
            recommendations["edge_difference"] = team2_edge - team1_edge
            recommendations["kelly_percentage"] = min((team2_edge - team1_edge) * 10, 0.05)
        else:
            recommendations["recommended_team"] = "แมตช์สมดุล - เดิมพันตลาดรอง"
            recommendations["edge_difference"] = 0
            recommendations["kelly_percentage"] = 0
        
        # แนะนำตลาดเดิมพัน
        markets = []
        if abs(team1_edge - team2_edge) > 0.15:
            markets.append("Match Winner")
        if max(t1_psych.clutch_mentality, t2_psych.clutch_mentality) > 0.8:
            markets.append("First Map Winner")
        if min(t1_tactics.adaptation_ability, t2_tactics.adaptation_ability) > 0.7:
            markets.append("Total Maps Over 2.5")
        
        recommendations["recommended_markets"] = markets
        
        return recommendations

    # Helper methods
    def _calculate_pressure_resistance(self, recent_matches: List) -> float:
        if not recent_matches:
            return 0.5
        important_matches = [m for m in recent_matches if m.get("importance", 0) > 0.7]
        if not important_matches:
            return 0.5
        wins_under_pressure = sum(1 for m in important_matches if m.get("result") == "win")
        return wins_under_pressure / len(important_matches)

    def _calculate_clutch_mentality(self, player_stats: Dict) -> float:
        if not player_stats:
            return 0.5
        clutch_stats = [p.get("clutch_rating", 0.5) for p in player_stats.values()]
        return sum(clutch_stats) / len(clutch_stats) if clutch_stats else 0.5

    def _calculate_consistency(self, recent_matches: List) -> float:
        if not recent_matches:
            return 0.5
        ratings = [m.get("team_rating", 0.5) for m in recent_matches]
        if not ratings:
            return 0.5
        avg_rating = sum(ratings) / len(ratings)
        variance = sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
        return max(0, 1 - variance)

    def _calculate_confidence_level(self, team_data: Dict) -> float:
        recent_form = team_data.get("recent_form", 0.5)
        ranking = team_data.get("ranking", 50)
        ranking_factor = max(0, (50 - ranking) / 50)
        return (recent_form + ranking_factor) / 2

    def _determine_form_trend(self, recent_matches: List) -> str:
        if not recent_matches:
            return "unknown"
        recent_results = [m.get("result") for m in recent_matches[-5:]]
        wins = sum(1 for r in recent_results if r == "win")
        if wins >= 4:
            return "excellent"
        elif wins >= 3:
            return "good"
        elif wins >= 2:
            return "average"
        else:
            return "poor"

    def _determine_tactical_style(self, team_name: str) -> str:
        return self.world_class_teams.get(team_name, {}).get("style", "balanced")

    def _calculate_recent_form(self, matches: List) -> float:
        if not matches:
            return 0.5
        wins = sum(1 for m in matches if m.get("result") == "win")
        return wins / len(matches)

    def _determine_default_setup(self, style: str) -> str:
        setups = {
            "aggressive": "Fast Execute Setup",
            "tactical": "Structured Setup", 
            "individual_skill": "Skill-based Setup",
            "balanced": "Adaptive Setup"
        }
        return setups.get(style, "Standard Setup")

    def _calculate_retake_efficiency(self, map_stats: Dict) -> float:
        # คำนวณจากสถิติแมพ
        return 0.7  # placeholder

    def _calculate_adaptation_ability(self, team_data: Dict) -> float:
        # คำนวณจากประวัติการปรับตัว
        return 0.6  # placeholder

    def _calculate_rotation_speed(self, team_data: Dict) -> float:
        # คำนวณจากสถิติการหมุนตำแหน่ง
        return 0.65  # placeholder

    def _calculate_prediction_confidence(self, team1_data: Dict, team2_data: Dict) -> float:
        # คำนวณความมั่นใจจากคุณภาพข้อมูล
        data_quality = (len(team1_data) + len(team2_data)) / 20
        return min(data_quality, 0.95)

    def _calculate_team_edge(self, psychology: PlayerPsychology, tactics: TacticalPattern) -> float:
        return (
            psychology.confidence_level * 0.3 +
            psychology.pressure_resistance * 0.2 +
            psychology.clutch_mentality * 0.2 +
            tactics.adaptation_ability * 0.15 +
            tactics.retake_efficiency * 0.15
        )

    def _get_fallback_team_data(self, team_name: str) -> Dict[str, Any]:
        return {
            "name": team_name,
            "ranking": 50,
            "recent_matches": [],
            "player_stats": {},
            "map_pool": {},
            "tier_info": self.world_class_teams.get(team_name, {"tier": "Unknown"}),
            "tactical_style": "balanced",
            "recent_form": 0.5
        }

# ฟังก์ชันสำหรับเรียกใช้
async def analyze_match(team1: str, team2: str, depth: str = "world_class"):
    """ฟังก์ชันหลักสำหรับวิเคราะห์แมตช์"""
    analyzer = UniversalMatchAnalyzer()
    depth_enum = AnalysisDepth(depth)
    return await analyzer.analyze_any_match(team1, team2, depth_enum)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        team1, team2 = sys.argv[1], sys.argv[2]
        depth = sys.argv[3] if len(sys.argv) > 3 else "world_class"
        asyncio.run(analyze_match(team1, team2, depth))
    else:
        print("Usage: python universal_match_analyzer.py <team1> <team2> [depth]")
