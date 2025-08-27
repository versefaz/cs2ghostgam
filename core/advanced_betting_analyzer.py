#!/usr/bin/env python3
"""
Advanced Betting Analyzer - Deep CS2 Match Analysis System
Provides comprehensive betting recommendations with handicap and over/under analysis
"""

import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of bets available"""
    MATCH_WINNER = "match_winner"
    HANDICAP = "handicap"
    OVER_UNDER_ROUNDS = "over_under_rounds"
    FIRST_MAP = "first_map"
    TOTAL_MAPS = "total_maps"


@dataclass
class OddsData:
    """Enhanced odds data structure"""
    bookmaker: str
    match_winner_team1: float
    match_winner_team2: float
    
    # Handicap betting (rounds advantage/disadvantage)
    handicap_team1_minus_1_5: Optional[float] = None
    handicap_team1_plus_1_5: Optional[float] = None
    handicap_team2_minus_1_5: Optional[float] = None
    handicap_team2_plus_1_5: Optional[float] = None
    
    # Over/Under rounds betting
    over_26_5_rounds: Optional[float] = None
    under_26_5_rounds: Optional[float] = None
    over_24_5_rounds: Optional[float] = None
    under_24_5_rounds: Optional[float] = None
    
    # Map betting
    first_map_team1: Optional[float] = None
    first_map_team2: Optional[float] = None
    total_maps_over_2_5: Optional[float] = None
    total_maps_under_2_5: Optional[float] = None


@dataclass
class BettingRecommendation:
    """Betting recommendation with analysis"""
    bet_type: BetType
    selection: str
    odds: float
    win_probability: float
    expected_value: float
    confidence_level: str  # HIGH, MEDIUM, LOW
    reasoning: str
    risk_level: str  # LOW, MEDIUM, HIGH
    stake_recommendation: float  # Percentage of bankroll


@dataclass
class TeamStats:
    """Team statistical analysis"""
    team_name: str
    avg_rounds_per_map: float
    win_rate_last_10: float
    handicap_performance: float
    over_under_tendency: str  # "OVER", "UNDER", "NEUTRAL"
    map_win_rate: float
    recent_form_score: float


class AdvancedBettingAnalyzer:
    """Advanced betting analysis engine"""
    
    def __init__(self, data_dir: str = "data/betting_analysis"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Load historical data and team stats
        self.team_stats = self._load_team_stats()
        self.historical_odds = self._load_historical_odds()
    
    def _load_team_stats(self) -> Dict[str, TeamStats]:
        """Load team statistical data"""
        # In production, this would load from database
        return {
            "Vitality": TeamStats(
                team_name="Vitality",
                avg_rounds_per_map=14.2,
                win_rate_last_10=0.85,
                handicap_performance=0.75,
                over_under_tendency="OVER",
                map_win_rate=0.78,
                recent_form_score=9.2
            ),
            "Natus Vincere": TeamStats(
                team_name="Natus Vincere",
                avg_rounds_per_map=13.8,
                win_rate_last_10=0.80,
                handicap_performance=0.70,
                over_under_tendency="OVER",
                map_win_rate=0.75,
                recent_form_score=8.8
            ),
            "FaZe": TeamStats(
                team_name="FaZe",
                avg_rounds_per_map=14.5,
                win_rate_last_10=0.75,
                handicap_performance=0.68,
                over_under_tendency="OVER",
                map_win_rate=0.72,
                recent_form_score=8.5
            ),
            "Virtus.pro": TeamStats(
                team_name="Virtus.pro",
                avg_rounds_per_map=13.2,
                win_rate_last_10=0.65,
                handicap_performance=0.60,
                over_under_tendency="UNDER",
                map_win_rate=0.68,
                recent_form_score=7.5
            ),
            "fnatic": TeamStats(
                team_name="fnatic",
                avg_rounds_per_map=12.8,
                win_rate_last_10=0.60,
                handicap_performance=0.55,
                over_under_tendency="UNDER",
                map_win_rate=0.65,
                recent_form_score=7.0
            ),
            "GamerLegion": TeamStats(
                team_name="GamerLegion",
                avg_rounds_per_map=12.5,
                win_rate_last_10=0.55,
                handicap_performance=0.50,
                over_under_tendency="UNDER",
                map_win_rate=0.60,
                recent_form_score=6.8
            ),
            "ECSTATIC": TeamStats(
                team_name="ECSTATIC",
                avg_rounds_per_map=12.0,
                win_rate_last_10=0.45,
                handicap_performance=0.45,
                over_under_tendency="UNDER",
                map_win_rate=0.55,
                recent_form_score=6.2
            ),
            "M80": TeamStats(
                team_name="M80",
                avg_rounds_per_map=11.8,
                win_rate_last_10=0.40,
                handicap_performance=0.40,
                over_under_tendency="UNDER",
                map_win_rate=0.50,
                recent_form_score=5.8
            )
        }
    
    def _load_historical_odds(self) -> List[Dict]:
        """Load historical odds data"""
        # Mock historical data - in production, load from database
        return []
    
    async def get_enhanced_odds_data(self, match_id: str, team1: str, team2: str) -> List[OddsData]:
        """Get comprehensive odds data from multiple bookmakers"""
        
        # Mock enhanced odds data - in production, scrape from bookmakers
        bookmakers_odds = []
        
        # Pinnacle odds (typically sharp)
        pinnacle_odds = OddsData(
            bookmaker="Pinnacle",
            match_winner_team1=self._get_match_winner_odds(team1, team2)[0],
            match_winner_team2=self._get_match_winner_odds(team1, team2)[1],
            handicap_team1_minus_1_5=self._calculate_handicap_odds(team1, team2, -1.5)[0],
            handicap_team1_plus_1_5=self._calculate_handicap_odds(team1, team2, 1.5)[0],
            handicap_team2_minus_1_5=self._calculate_handicap_odds(team1, team2, -1.5)[1],
            handicap_team2_plus_1_5=self._calculate_handicap_odds(team1, team2, 1.5)[1],
            over_26_5_rounds=self._calculate_over_under_odds(team1, team2, 26.5)[0],
            under_26_5_rounds=self._calculate_over_under_odds(team1, team2, 26.5)[1],
            over_24_5_rounds=self._calculate_over_under_odds(team1, team2, 24.5)[0],
            under_24_5_rounds=self._calculate_over_under_odds(team1, team2, 24.5)[1],
            first_map_team1=self._calculate_first_map_odds(team1, team2)[0],
            first_map_team2=self._calculate_first_map_odds(team1, team2)[1],
            total_maps_over_2_5=1.85,
            total_maps_under_2_5=1.95
        )
        bookmakers_odds.append(pinnacle_odds)
        
        return bookmakers_odds
    
    def _get_match_winner_odds(self, team1: str, team2: str, margin: float = 0.03) -> Tuple[float, float]:
        """Calculate match winner odds based on team strength"""
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return (2.0, 2.0)  # Default even odds
        
        # Calculate implied probability based on team strength
        team1_strength = (team1_stats.win_rate_last_10 * 0.4 + 
                         team1_stats.recent_form_score / 10 * 0.3 + 
                         team1_stats.map_win_rate * 0.3)
        
        team2_strength = (team2_stats.win_rate_last_10 * 0.4 + 
                         team2_stats.recent_form_score / 10 * 0.3 + 
                         team2_stats.map_win_rate * 0.3)
        
        total_strength = team1_strength + team2_strength
        team1_prob = team1_strength / total_strength
        team2_prob = team2_strength / total_strength
        
        # Add bookmaker margin
        team1_odds = round(1 / (team1_prob - margin), 2)
        team2_odds = round(1 / (team2_prob - margin), 2)
        
        return (team1_odds, team2_odds)
    
    def _calculate_handicap_odds(self, team1: str, team2: str, handicap: float, margin: float = 0.04) -> Tuple[float, float]:
        """Calculate handicap betting odds"""
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return (1.90, 1.90)
        
        # Adjust probability based on handicap and team performance
        base_prob = team1_stats.handicap_performance if handicap > 0 else team2_stats.handicap_performance
        handicap_adjustment = abs(handicap) * 0.15  # Each 1.5 rounds worth ~15% probability shift
        
        if handicap > 0:  # Team1 getting rounds
            team1_prob = min(0.85, base_prob + handicap_adjustment)
        else:  # Team1 giving rounds
            team1_prob = max(0.15, base_prob - handicap_adjustment)
        
        team2_prob = 1 - team1_prob
        
        team1_odds = round(1 / (team1_prob - margin), 2)
        team2_odds = round(1 / (team2_prob - margin), 2)
        
        return (team1_odds, team2_odds)
    
    def _calculate_over_under_odds(self, team1: str, team2: str, line: float, margin: float = 0.04) -> Tuple[float, float]:
        """Calculate over/under rounds odds"""
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return (1.90, 1.90)
        
        # Calculate expected total rounds
        avg_rounds = (team1_stats.avg_rounds_per_map + team2_stats.avg_rounds_per_map) / 2
        
        # Adjust based on team tendencies
        if team1_stats.over_under_tendency == "OVER" and team2_stats.over_under_tendency == "OVER":
            avg_rounds += 1.5
        elif team1_stats.over_under_tendency == "UNDER" and team2_stats.over_under_tendency == "UNDER":
            avg_rounds -= 1.5
        
        # Calculate probability based on distance from line
        distance_from_line = avg_rounds - line
        over_prob = 0.5 + (distance_from_line * 0.08)  # Each round difference = 8% probability shift
        over_prob = max(0.15, min(0.85, over_prob))
        under_prob = 1 - over_prob
        
        over_odds = round(1 / (over_prob - margin), 2)
        under_odds = round(1 / (under_prob - margin), 2)
        
        return (over_odds, under_odds)
    
    def _calculate_first_map_odds(self, team1: str, team2: str, margin: float = 0.05) -> Tuple[float, float]:
        """Calculate first map winner odds"""
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return (1.95, 1.95)
        
        # First map often more volatile, adjust probabilities
        team1_prob = team1_stats.map_win_rate * 0.9  # Slight reduction for volatility
        team2_prob = team2_stats.map_win_rate * 0.9
        
        total_prob = team1_prob + team2_prob
        team1_prob = team1_prob / total_prob
        team2_prob = team2_prob / total_prob
        
        team1_odds = round(1 / (team1_prob - margin), 2)
        team2_odds = round(1 / (team2_prob - margin), 2)
        
        return (team1_odds, team2_odds)
    
    async def analyze_betting_opportunities(self, match_id: str, team1: str, team2: str) -> List[BettingRecommendation]:
        """Comprehensive betting analysis with recommendations"""
        
        # Get odds data
        odds_data = await self.get_enhanced_odds_data(match_id, team1, team2)
        
        recommendations = []
        
        # Analyze each bet type
        for odds in odds_data:
            if odds.bookmaker == "Pinnacle":  # Use sharp bookmaker for analysis
                
                # Match Winner Analysis
                match_winner_rec = await self._analyze_match_winner(team1, team2, odds)
                if match_winner_rec:
                    recommendations.append(match_winner_rec)
                
                # Handicap Analysis
                handicap_recs = await self._analyze_handicap_bets(team1, team2, odds)
                recommendations.extend(handicap_recs)
                
                # Over/Under Analysis
                over_under_recs = await self._analyze_over_under_bets(team1, team2, odds)
                recommendations.extend(over_under_recs)
                
                # First Map Analysis
                first_map_rec = await self._analyze_first_map(team1, team2, odds)
                if first_map_rec:
                    recommendations.append(first_map_rec)
        
        # Sort by expected value
        recommendations.sort(key=lambda x: x.expected_value, reverse=True)
        
        # Save analysis
        await self._save_betting_analysis(match_id, recommendations)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _analyze_match_winner(self, team1: str, team2: str, odds: OddsData) -> Optional[BettingRecommendation]:
        """Analyze match winner betting opportunity"""
        
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return None
        
        # Calculate true probability
        team1_true_prob = (team1_stats.win_rate_last_10 * 0.5 + 
                          team1_stats.recent_form_score / 10 * 0.3 + 
                          team1_stats.map_win_rate * 0.2)
        
        team2_true_prob = (team2_stats.win_rate_last_10 * 0.5 + 
                          team2_stats.recent_form_score / 10 * 0.3 + 
                          team2_stats.map_win_rate * 0.2)
        
        # Normalize probabilities
        total_prob = team1_true_prob + team2_true_prob
        team1_true_prob = team1_true_prob / total_prob
        team2_true_prob = team2_true_prob / total_prob
        
        # Calculate expected values
        team1_ev = (team1_true_prob * odds.match_winner_team1) - 1
        team2_ev = (team2_true_prob * odds.match_winner_team2) - 1
        
        # Find best bet
        if team1_ev > 0.05 and team1_ev > team2_ev:
            return BettingRecommendation(
                bet_type=BetType.MATCH_WINNER,
                selection=team1,
                odds=odds.match_winner_team1,
                win_probability=team1_true_prob,
                expected_value=team1_ev,
                confidence_level="HIGH" if team1_ev > 0.15 else "MEDIUM" if team1_ev > 0.08 else "LOW",
                reasoning=f"{team1} shows {team1_ev:.1%} positive expected value based on recent form and historical performance",
                risk_level="MEDIUM",
                stake_recommendation=min(0.05, team1_ev * 0.25)  # Kelly criterion approximation
            )
        elif team2_ev > 0.05 and team2_ev > team1_ev:
            return BettingRecommendation(
                bet_type=BetType.MATCH_WINNER,
                selection=team2,
                odds=odds.match_winner_team2,
                win_probability=team2_true_prob,
                expected_value=team2_ev,
                confidence_level="HIGH" if team2_ev > 0.15 else "MEDIUM" if team2_ev > 0.08 else "LOW",
                reasoning=f"{team2} shows {team2_ev:.1%} positive expected value based on recent form and historical performance",
                risk_level="MEDIUM",
                stake_recommendation=min(0.05, team2_ev * 0.25)
            )
        
        return None
    
    async def _analyze_handicap_bets(self, team1: str, team2: str, odds: OddsData) -> List[BettingRecommendation]:
        """Analyze handicap betting opportunities"""
        
        recommendations = []
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return recommendations
        
        # Analyze -1.5 handicap for stronger team
        stronger_team = team1 if team1_stats.recent_form_score > team2_stats.recent_form_score else team2
        
        if stronger_team == team1 and odds.handicap_team1_minus_1_5:
            handicap_prob = team1_stats.handicap_performance * 0.85  # Reduce for difficulty
            ev = (handicap_prob * odds.handicap_team1_minus_1_5) - 1
            
            if ev > 0.03:
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.HANDICAP,
                    selection=f"{team1} -1.5",
                    odds=odds.handicap_team1_minus_1_5,
                    win_probability=handicap_prob,
                    expected_value=ev,
                    confidence_level="HIGH" if ev > 0.12 else "MEDIUM" if ev > 0.06 else "LOW",
                    reasoning=f"{team1} has strong handicap performance ({team1_stats.handicap_performance:.1%}) and form advantage",
                    risk_level="HIGH",
                    stake_recommendation=min(0.03, ev * 0.2)
                ))
        
        return recommendations
    
    async def _analyze_over_under_bets(self, team1: str, team2: str, odds: OddsData) -> List[BettingRecommendation]:
        """Analyze over/under rounds betting opportunities"""
        
        recommendations = []
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats:
            return recommendations
        
        # Calculate expected rounds
        expected_rounds = (team1_stats.avg_rounds_per_map + team2_stats.avg_rounds_per_map) / 2
        
        # Adjust for team tendencies
        over_tendency_bonus = 0
        if team1_stats.over_under_tendency == "OVER" and team2_stats.over_under_tendency == "OVER":
            over_tendency_bonus = 2.0
        elif team1_stats.over_under_tendency == "UNDER" and team2_stats.over_under_tendency == "UNDER":
            over_tendency_bonus = -2.0
        
        expected_rounds += over_tendency_bonus
        
        # Analyze Over 26.5
        if odds.over_26_5_rounds:
            over_26_5_prob = max(0.1, min(0.9, 0.5 + ((expected_rounds - 26.5) * 0.08)))
            over_26_5_ev = (over_26_5_prob * odds.over_26_5_rounds) - 1
            
            if over_26_5_ev > 0.04:
                recommendations.append(BettingRecommendation(
                    bet_type=BetType.OVER_UNDER_ROUNDS,
                    selection="Over 26.5 rounds",
                    odds=odds.over_26_5_rounds,
                    win_probability=over_26_5_prob,
                    expected_value=over_26_5_ev,
                    confidence_level="HIGH" if over_26_5_ev > 0.12 else "MEDIUM" if over_26_5_ev > 0.07 else "LOW",
                    reasoning=f"Expected {expected_rounds:.1f} rounds, both teams tend towards {team1_stats.over_under_tendency}/{team2_stats.over_under_tendency}",
                    risk_level="LOW",
                    stake_recommendation=min(0.06, over_26_5_ev * 0.3)
                ))
        
        return recommendations
    
    async def _analyze_first_map(self, team1: str, team2: str, odds: OddsData) -> Optional[BettingRecommendation]:
        """Analyze first map betting opportunity"""
        
        team1_stats = self.team_stats.get(team1)
        team2_stats = self.team_stats.get(team2)
        
        if not team1_stats or not team2_stats or not odds.first_map_team1:
            return None
        
        # First map is more volatile, adjust probabilities
        team1_first_map_prob = team1_stats.map_win_rate * 0.95
        team2_first_map_prob = team2_stats.map_win_rate * 0.95
        
        # Normalize
        total_prob = team1_first_map_prob + team2_first_map_prob
        team1_first_map_prob = team1_first_map_prob / total_prob
        
        # Calculate expected values
        team1_ev = (team1_first_map_prob * odds.first_map_team1) - 1
        
        # Return best opportunity
        if team1_ev > 0.06:
            return BettingRecommendation(
                bet_type=BetType.FIRST_MAP,
                selection=f"{team1} first map",
                odds=odds.first_map_team1,
                win_probability=team1_first_map_prob,
                expected_value=team1_ev,
                confidence_level="MEDIUM" if team1_ev > 0.10 else "LOW",
                reasoning=f"{team1} has strong map win rate ({team1_stats.map_win_rate:.1%}) and good recent form",
                risk_level="HIGH",
                stake_recommendation=min(0.02, team1_ev * 0.15)
            )
        
        return None
    
    async def _save_betting_analysis(self, match_id: str, recommendations: List[BettingRecommendation]):
        """Save betting analysis to file"""
        
        analysis_file = self.data_dir / f"betting_analysis_{match_id}.json"
        
        analysis_data = {
            "match_id": match_id,
            "analysis_time": datetime.now().isoformat(),
            "recommendations": [{
                "bet_type": rec.bet_type.value,
                "selection": rec.selection,
                "odds": rec.odds,
                "win_probability": rec.win_probability,
                "expected_value": rec.expected_value,
                "confidence_level": rec.confidence_level,
                "reasoning": rec.reasoning,
                "risk_level": rec.risk_level,
                "stake_recommendation": rec.stake_recommendation
            } for rec in recommendations]
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)


# Global analyzer instance
_analyzer_instance = None

def get_betting_analyzer() -> AdvancedBettingAnalyzer:
    """Get global betting analyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AdvancedBettingAnalyzer()
    return _analyzer_instance
