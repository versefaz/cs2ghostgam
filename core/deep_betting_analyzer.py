#!/usr/bin/env python3
"""
Deep Betting Analyzer - วิเคราะห์การเดิมพันเชิงลึกด้วยข้อมูลจริง
Created by KoJao
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from enum import Enum

from .enhanced_team_analyzer import get_enhanced_analyzer, TeamAnalysis

class BetType(Enum):
    MATCH_WINNER = "match_winner"
    HANDICAP = "handicap"
    TOTAL_MAPS = "total_maps"
    EXACT_SCORE = "exact_score"

@dataclass
class BettingOpportunity:
    """โอกาสการเดิมพัน"""
    bet_type: BetType
    selection: str
    odds: float
    implied_probability: float
    true_probability: float
    expected_value: float
    confidence_level: str
    risk_level: str
    stake_recommendation: float
    detailed_reasoning: str
    supporting_stats: Dict

class DeepBettingAnalyzer:
    """ตัววิเคราะห์การเดิมพันเชิงลึก"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.team_analyzer = get_enhanced_analyzer()
        
        # ราคาจริงจากรูปภาพ Vitality vs M80
        self.real_odds = {
            "Vitality_vs_M80": {
                "match_winner": {"Vitality": 1.01, "M80": 15.28},
                "handicap": {"Vitality_-1.5": 1.19, "M80_+1.5": 4.43},
                "total_maps": {"over_2.5": 4.74, "under_2.5": 1.17},
                "exact_score": {"2-0": 1.19, "2-1": 5.21, "1-2": 20.39, "0-2": 25.0}
            }
        }
    
    async def analyze_match_deep(self, team1: str, team2: str) -> Dict:
        """วิเคราะห์แมตช์เชิงลึกพร้อมคำแนะนำการเดิมพัน"""
        
        # วิเคราะห์ทีม
        comparison = await self.team_analyzer.compare_teams(team1, team2)
        team1_analysis = comparison["team1_analysis"]
        team2_analysis = comparison["team2_analysis"]
        
        # คำนวณความน่าจะเป็นที่แท้จริง
        true_probabilities = self._calculate_true_probabilities(team1_analysis, team2_analysis)
        
        # วิเคราะห์โอกาสการเดิมพัน
        betting_opportunities = await self._analyze_betting_opportunities(
            team1, team2, team1_analysis, team2_analysis, true_probabilities
        )
        
        return {
            "team1_analysis": team1_analysis,
            "team2_analysis": team2_analysis,
            "predicted_winner": comparison["predicted_winner"],
            "win_probability": true_probabilities["team1_win"] if comparison["predicted_winner"] == team1 else true_probabilities["team2_win"],
            "betting_opportunities": betting_opportunities,
            "key_factors": self._identify_key_factors(team1_analysis, team2_analysis)
        }
    
    def _calculate_true_probabilities(self, team1: TeamAnalysis, team2: TeamAnalysis) -> Dict[str, float]:
        """คำนวณความน่าจะเป็นที่แท้จริง"""
        
        # คำนวณจากสถิติผู้เล่น
        team1_avg_rating = sum(p.rating for p in team1.players) / len(team1.players)
        team2_avg_rating = sum(p.rating for p in team2.players) / len(team2.players)
        
        # คำนวณจากฟอร์ม
        form_weights = {"excellent": 1.3, "good": 1.1, "average": 1.0, "poor": 0.8}
        team1_form = form_weights[team1.recent_form]
        team2_form = form_weights[team2.recent_form]
        
        # คำนวณจาก ranking
        ranking_factor1 = (50 - team1.current_ranking) / 50
        ranking_factor2 = (50 - team2.current_ranking) / 50
        
        # รวมคะแนน
        team1_score = team1_avg_rating * team1_form + ranking_factor1
        team2_score = team2_avg_rating * team2_form + ranking_factor2
        
        total_score = team1_score + team2_score
        team1_win_prob = team1_score / total_score
        team2_win_prob = team2_score / total_score
        
        # คำนวณสกอร์
        strength_diff = abs(team1_score - team2_score)
        if team1_win_prob > team2_win_prob:
            prob_2_0 = 0.4 + (strength_diff * 0.3)
            prob_2_1 = 0.6 - (strength_diff * 0.3)
        else:
            prob_2_0 = 0.05
            prob_2_1 = 0.15
        
        return {
            "team1_win": team1_win_prob,
            "team2_win": team2_win_prob,
            "score_2_0": max(0.05, prob_2_0),
            "score_2_1": max(0.1, prob_2_1),
            "over_2_5": prob_2_1,
            "under_2_5": prob_2_0
        }
    
    async def _analyze_betting_opportunities(self, team1: str, team2: str, 
                                           team1_analysis: TeamAnalysis, 
                                           team2_analysis: TeamAnalysis,
                                           true_probs: Dict) -> List[BettingOpportunity]:
        """วิเคราะห์โอกาสการเดิมพัน"""
        
        opportunities = []
        match_key = f"{team1}_vs_{team2}"
        
        if match_key not in self.real_odds:
            return opportunities
        
        odds_data = self.real_odds[match_key]
        
        # 1. Match Winner Analysis
        for team in [team1, team2]:
            if team in odds_data["match_winner"]:
                odds = odds_data["match_winner"][team]
                implied_prob = 1 / odds
                true_prob = true_probs["team1_win"] if team == team1 else true_probs["team2_win"]
                
                if true_prob > implied_prob:
                    ev = (true_prob * odds) - 1
                    
                    reasoning = self._generate_match_winner_reasoning(
                        team, team1_analysis if team == team1 else team2_analysis,
                        team2_analysis if team == team1 else team1_analysis, ev
                    )
                    
                    opportunities.append(BettingOpportunity(
                        bet_type=BetType.MATCH_WINNER,
                        selection=team,
                        odds=odds,
                        implied_probability=implied_prob,
                        true_probability=true_prob,
                        expected_value=ev,
                        confidence_level=self._get_confidence_level(ev),
                        risk_level=self._get_risk_level(odds),
                        stake_recommendation=self._calculate_kelly_stake(odds, true_prob),
                        detailed_reasoning=reasoning,
                        supporting_stats=self._get_supporting_stats(team1_analysis if team == team1 else team2_analysis)
                    ))
        
        # 2. Handicap Analysis
        if "handicap" in odds_data:
            for selection, odds in odds_data["handicap"].items():
                implied_prob = 1 / odds
                
                if "-1.5" in selection:
                    true_prob = true_probs["score_2_0"]
                else:
                    true_prob = 1 - true_probs["score_2_0"]
                
                if true_prob > implied_prob:
                    ev = (true_prob * odds) - 1
                    reasoning = f"{selection}: Expected Value {ev:.1%} จากการวิเคราะห์ความแข็งแกร่งของทีม"
                    
                    opportunities.append(BettingOpportunity(
                        bet_type=BetType.HANDICAP,
                        selection=selection,
                        odds=odds,
                        implied_probability=implied_prob,
                        true_probability=true_prob,
                        expected_value=ev,
                        confidence_level=self._get_confidence_level(ev),
                        risk_level=self._get_risk_level(odds),
                        stake_recommendation=self._calculate_kelly_stake(odds, true_prob),
                        detailed_reasoning=reasoning,
                        supporting_stats={}
                    ))
        
        # 3. Total Maps Analysis
        if "total_maps" in odds_data:
            for selection, odds in odds_data["total_maps"].items():
                implied_prob = 1 / odds
                true_prob = true_probs["over_2_5"] if "over" in selection else true_probs["under_2_5"]
                
                if true_prob > implied_prob:
                    ev = (true_prob * odds) - 1
                    reasoning = f"{selection}: Expected Value {ev:.1%} จากการวิเคราะห์ระยะเวลาแมตช์"
                    
                    opportunities.append(BettingOpportunity(
                        bet_type=BetType.TOTAL_MAPS,
                        selection=selection,
                        odds=odds,
                        implied_probability=implied_prob,
                        true_probability=true_prob,
                        expected_value=ev,
                        confidence_level=self._get_confidence_level(ev),
                        risk_level=self._get_risk_level(odds),
                        stake_recommendation=self._calculate_kelly_stake(odds, true_prob),
                        detailed_reasoning=reasoning,
                        supporting_stats={}
                    ))
        
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        return opportunities[:5]
    
    def _generate_match_winner_reasoning(self, team: str, team_analysis: TeamAnalysis, 
                                       opponent_analysis: TeamAnalysis, ev: float) -> str:
        """สร้างเหตุผลเชิงลึกสำหรับการเดิมพันผู้ชนะแมตช์"""
        
        reasons = []
        
        # วิเคราะห์ผู้เล่นดาวเด่น
        star_player = max(team_analysis.players, key=lambda p: p.rating)
        opponent_star = max(opponent_analysis.players, key=lambda p: p.rating)
        
        if star_player.rating > opponent_star.rating + 0.15:
            reasons.append(f"🌟 {star_player.name} (Rating {star_player.rating}) เหนือกว่า {opponent_star.name} ({opponent_star.rating}) อย่างชัดเจน")
        
        # วิเคราะห์ฟอร์มและ win streak
        if team_analysis.recent_form == "excellent" and team_analysis.win_streak >= 5:
            reasons.append(f"🔥 {team} ฟอร์มร้อนแรง ชนะติดต่อกัน {team_analysis.win_streak} แมตช์")
        elif team_analysis.recent_form == "excellent":
            reasons.append(f"⚡ {team} อยู่ในฟอร์มดีเยี่ยม มีความมั่นคงสูง")
        
        # วิเคราะห์ ranking และความแข็งแกร่ง
        ranking_diff = opponent_analysis.current_ranking - team_analysis.current_ranking
        if ranking_diff > 20:
            reasons.append(f"🏆 อันดับ {team_analysis.current_ranking} เหนือกว่าคู่แข่ง {ranking_diff} อันดับ")
        
        # วิเคราะห์ map pool
        team_avg_winrate = sum(m.win_rate for m in team_analysis.map_pool) / len(team_analysis.map_pool)
        opponent_avg_winrate = sum(m.win_rate for m in opponent_analysis.map_pool) / len(opponent_analysis.map_pool)
        if team_avg_winrate > opponent_avg_winrate + 15:
            reasons.append(f"🗺️ Win rate เฉลี่ย {team_avg_winrate:.1f}% สูงกว่าคู่แข่ง {team_avg_winrate - opponent_avg_winrate:.1f}%")
        
        # วิเคราะห์ clutch และ pressure performance
        team_clutch = sum(p.clutch_success_rate for p in team_analysis.players) / len(team_analysis.players)
        opponent_clutch = sum(p.clutch_success_rate for p in opponent_analysis.players) / len(opponent_analysis.players)
        if team_clutch > opponent_clutch + 5:
            reasons.append(f"🎯 Clutch success rate {team_clutch:.1f}% เหนือกว่าคู่แข่ง")
        
        if team_analysis.pressure_performance > opponent_analysis.pressure_performance + 15:
            reasons.append(f"💪 เล่นภายใต้แรงกดดันได้ดีกว่า ({team_analysis.pressure_performance:.1f}% vs {opponent_analysis.pressure_performance:.1f}%)")
        
        # สรุป Expected Value และคำแนะนำ
        if ev > 3.0:
            reasons.append(f"💎 Expected Value {ev:.1%} สูงมหาศาล คุ้มค่าการลงทุนสูงสุด")
        elif ev > 0.5:
            reasons.append(f"💰 Expected Value {ev:.1%} สูงมาก คุ้มค่าการลงทุน")
        elif ev > 0.2:
            reasons.append(f"✅ Expected Value {ev:.1%} ดี คุ้มค่าการลงทุน")
        
        return " | ".join(reasons[:4])
    
    def _identify_key_factors(self, team1: TeamAnalysis, team2: TeamAnalysis) -> List[str]:
        """ระบุปัจจัยสำคัญ"""
        factors = []
        
        # ความแตกต่างของ rating
        team1_avg = sum(p.rating for p in team1.players) / len(team1.players)
        team2_avg = sum(p.rating for p in team2.players) / len(team2.players)
        
        if abs(team1_avg - team2_avg) > 0.2:
            stronger = team1.team_name if team1_avg > team2_avg else team2.team_name
            factors.append(f"{stronger} มีทักษะรายบุคคลเหนือกว่าอย่างชัดเจน")
        
        # ฟอร์ม
        if team1.recent_form != team2.recent_form:
            better_form = team1.team_name if team1.recent_form == "excellent" else team2.team_name
            factors.append(f"{better_form} อยู่ในฟอร์มที่ดีกว่า")
        
        return factors[:3]
    
    def _get_confidence_level(self, ev: float) -> str:
        if ev > 0.3: return "HIGH"
        elif ev > 0.15: return "MEDIUM"
        else: return "LOW"
    
    def _get_risk_level(self, odds: float) -> str:
        if odds > 5.0: return "HIGH"
        elif odds > 2.5: return "MEDIUM"
        else: return "LOW"
    
    def _calculate_kelly_stake(self, odds: float, true_prob: float) -> float:
        b = odds - 1
        p = true_prob
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0, min(0.1, kelly))
    
    def _get_supporting_stats(self, team_analysis: TeamAnalysis) -> Dict:
        return {
            "avg_rating": sum(p.rating for p in team_analysis.players) / len(team_analysis.players),
            "star_player": max(team_analysis.players, key=lambda p: p.rating).name,
            "recent_form": team_analysis.recent_form,
            "ranking": team_analysis.current_ranking
        }

# Global instance
_deep_analyzer = None

def get_deep_betting_analyzer() -> DeepBettingAnalyzer:
    global _deep_analyzer
    if _deep_analyzer is None:
        _deep_analyzer = DeepBettingAnalyzer()
    return _deep_analyzer
