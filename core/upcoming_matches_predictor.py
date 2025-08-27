#!/usr/bin/env python3
"""
Upcoming Matches Predictor - Real-time CS2 Match Prediction System
Focuses on upcoming matches only with manual result reporting
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UpcomingMatch:
    """Data structure for upcoming CS2 matches"""
    match_id: str
    team1: str
    team2: str
    scheduled_time: datetime
    event_name: str
    odds_team1: float
    odds_team2: float
    predicted_winner: Optional[str] = None
    confidence: Optional[float] = None
    prediction_reasoning: Optional[str] = None
    status: str = "upcoming"  # upcoming, live, completed, cancelled


@dataclass
class MatchPrediction:
    """Prediction data for a match"""
    match_id: str
    predicted_winner: str
    confidence: float
    reasoning: str
    prediction_time: datetime
    model_features: Optional[Dict[str, Any]] = None


class UpcomingMatchesPredictor:
    """Predict outcomes for upcoming CS2 matches only"""
    
    def __init__(self, data_dir: str = "data/matches"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Load today's matches
        self.upcoming_matches: Dict[str, UpcomingMatch] = {}
        self.predictions: Dict[str, MatchPrediction] = {}
        
        # Initialize with today's BLAST Open London matches
        self._load_todays_matches()
    
    def _load_todays_matches(self):
        """Load today's actual matches from BLAST Open London 2025"""
        today = datetime.now().date()
        
        # Today's actual matches from the image
        matches_data = [
            {
                "match_id": "blast_london_001",
                "team1": "Vitality",
                "team2": "M80", 
                "time": "17:00",
                "odds_team1": 1.05,
                "odds_team2": 8.95
            },
            {
                "match_id": "blast_london_002", 
                "team1": "GamerLegion",
                "team2": "Virtus.pro",
                "time": "19:30",
                "odds_team1": 2.07,
                "odds_team2": 1.78
            },
            {
                "match_id": "blast_london_003",
                "team1": "FaZe",
                "team2": "ECSTATIC", 
                "time": "22:00",
                "odds_team1": 1.34,
                "odds_team2": 3.25
            },
            {
                "match_id": "blast_london_004",
                "team1": "Natus Vincere",
                "team2": "fnatic",
                "time": "00:30",
                "odds_team1": 1.26,
                "odds_team2": 3.82
            }
        ]
        
        for match_data in matches_data:
            # Parse time for today
            time_str = match_data["time"]
            hour, minute = map(int, time_str.split(":"))
            
            # Handle next day for 00:30 match
            match_date = today
            if hour == 0:  # Next day
                match_date = today + timedelta(days=1)
            
            scheduled_time = datetime.combine(match_date, datetime.min.time().replace(hour=hour, minute=minute))
            
            match = UpcomingMatch(
                match_id=match_data["match_id"],
                team1=match_data["team1"],
                team2=match_data["team2"],
                scheduled_time=scheduled_time,
                event_name="BLAST Open London 2025",
                odds_team1=match_data["odds_team1"],
                odds_team2=match_data["odds_team2"]
            )
            
            self.upcoming_matches[match.match_id] = match
        
        self.logger.info(f"Loaded {len(self.upcoming_matches)} upcoming matches for today")
    
    async def generate_predictions(self) -> Dict[str, MatchPrediction]:
        """Generate predictions for all upcoming matches"""
        predictions = {}
        
        for match_id, match in self.upcoming_matches.items():
            if match.status != "upcoming":
                continue
                
            prediction = await self._predict_match(match)
            predictions[match_id] = prediction
            
            # Update match with prediction
            match.predicted_winner = prediction.predicted_winner
            match.confidence = prediction.confidence
            match.prediction_reasoning = prediction.reasoning
        
        self.predictions.update(predictions)
        await self._save_predictions()
        
        return predictions
    
    async def _predict_match(self, match: UpcomingMatch) -> MatchPrediction:
        """Generate prediction for a single match using team analysis"""
        
        # Team strength analysis based on odds and historical performance
        team1_strength = await self._analyze_team_strength(match.team1, match.odds_team1)
        team2_strength = await self._analyze_team_strength(match.team2, match.odds_team2)
        
        # Determine predicted winner
        if team1_strength > team2_strength:
            predicted_winner = match.team1
            confidence = min(0.95, team1_strength / (team1_strength + team2_strength))
        else:
            predicted_winner = match.team2
            confidence = min(0.95, team2_strength / (team1_strength + team2_strength))
        
        # Generate reasoning
        reasoning = await self._generate_prediction_reasoning(match, team1_strength, team2_strength)
        
        prediction = MatchPrediction(
            match_id=match.match_id,
            predicted_winner=predicted_winner,
            confidence=confidence,
            reasoning=reasoning,
            prediction_time=datetime.now(),
            model_features={
                "team1_strength": team1_strength,
                "team2_strength": team2_strength,
                "odds_factor": match.odds_team1 / match.odds_team2,
                "event_tier": "tier1"  # BLAST is tier 1
            }
        )
        
        return prediction
    
    async def _analyze_team_strength(self, team_name: str, odds: float) -> float:
        """Analyze team strength based on various factors"""
        
        # Base strength from odds (lower odds = higher strength)
        odds_strength = 1.0 / odds if odds > 0 else 0.5
        
        # Team tier analysis
        tier1_teams = ["Vitality", "Natus Vincere", "FaZe", "Astralis", "G2"]
        tier2_teams = ["Virtus.pro", "fnatic", "ENCE", "Cloud9"]
        tier3_teams = ["GamerLegion", "ECSTATIC", "M80"]
        
        if team_name in tier1_teams:
            tier_bonus = 0.3
        elif team_name in tier2_teams:
            tier_bonus = 0.1
        elif team_name in tier3_teams:
            tier_bonus = -0.1
        else:
            tier_bonus = 0.0
        
        # Recent form analysis (simplified)
        recent_form = 0.1  # Assume neutral form
        
        total_strength = odds_strength + tier_bonus + recent_form
        return max(0.1, min(2.0, total_strength))
    
    async def _generate_prediction_reasoning(self, match: UpcomingMatch, 
                                           team1_strength: float, team2_strength: float) -> str:
        """Generate human-readable reasoning for the prediction"""
        
        stronger_team = match.team1 if team1_strength > team2_strength else match.team2
        weaker_team = match.team2 if team1_strength > team2_strength else match.team1
        stronger_odds = match.odds_team1 if team1_strength > team2_strength else match.odds_team2
        
        reasoning_parts = []
        
        # Odds analysis
        if stronger_odds < 1.5:
            reasoning_parts.append(f"{stronger_team} heavily favored by bookmakers (odds {stronger_odds:.2f})")
        elif stronger_odds < 2.0:
            reasoning_parts.append(f"{stronger_team} favored by bookmakers (odds {stronger_odds:.2f})")
        else:
            reasoning_parts.append(f"Close match according to odds ({match.odds_team1:.2f} vs {match.odds_team2:.2f})")
        
        # Team tier analysis
        tier1_teams = ["Vitality", "Natus Vincere", "FaZe"]
        if stronger_team in tier1_teams:
            reasoning_parts.append(f"{stronger_team} is a top-tier team with consistent performance")
        
        # Event context
        reasoning_parts.append("BLAST Open London is a high-stakes tournament")
        
        return ". ".join(reasoning_parts) + "."
    
    async def get_upcoming_matches_with_predictions(self) -> List[Dict[str, Any]]:
        """Get all upcoming matches with their predictions"""
        matches_with_predictions = []
        
        for match in self.upcoming_matches.values():
            if match.status != "upcoming":
                continue
            
            match_data = {
                "match_id": match.match_id,
                "team1": match.team1,
                "team2": match.team2,
                "scheduled_time": match.scheduled_time.strftime("%H:%M"),
                "event": match.event_name,
                "odds": f"{match.odds_team1:.2f} / {match.odds_team2:.2f}",
                "predicted_winner": match.predicted_winner,
                "confidence": f"{match.confidence:.1%}" if match.confidence else "N/A",
                "reasoning": match.prediction_reasoning or "No prediction yet"
            }
            matches_with_predictions.append(match_data)
        
        # Sort by scheduled time
        matches_with_predictions.sort(key=lambda x: x["scheduled_time"])
        return matches_with_predictions
    
    async def report_match_result(self, match_id: str, winner: str, 
                                user_name: str = "User") -> Dict[str, Any]:
        """Allow user to report match result manually"""
        
        if match_id not in self.upcoming_matches:
            return {"error": f"Match {match_id} not found"}
        
        match = self.upcoming_matches[match_id]
        
        if match.status == "completed":
            return {"error": f"Match {match_id} already completed"}
        
        # Validate winner
        if winner not in [match.team1, match.team2]:
            return {"error": f"Winner must be either {match.team1} or {match.team2}"}
        
        # Update match status
        match.status = "completed"
        
        # Calculate prediction accuracy if we had a prediction
        prediction_correct = None
        if match.predicted_winner:
            prediction_correct = (match.predicted_winner == winner)
        
        # Create result record
        result = {
            "match_id": match_id,
            "teams": f"{match.team1} vs {match.team2}",
            "winner": winner,
            "reported_by": user_name,
            "reported_time": datetime.now().isoformat(),
            "predicted_winner": match.predicted_winner,
            "prediction_correct": prediction_correct,
            "confidence": match.confidence
        }
        
        # Save result
        await self._save_match_result(result)
        
        self.logger.info(f"Match result reported: {match.team1} vs {match.team2} - Winner: {winner}")
        
        return {
            "success": True,
            "message": f"Result reported successfully: {winner} won",
            "prediction_correct": prediction_correct,
            "result": result
        }
    
    async def _save_predictions(self):
        """Save predictions to file"""
        predictions_file = self.data_dir / "predictions.json"
        
        predictions_data = {}
        for match_id, prediction in self.predictions.items():
            predictions_data[match_id] = {
                "match_id": prediction.match_id,
                "predicted_winner": prediction.predicted_winner,
                "confidence": prediction.confidence,
                "reasoning": prediction.reasoning,
                "prediction_time": prediction.prediction_time.isoformat()
            }
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    
    async def _save_match_result(self, result: Dict[str, Any]):
        """Save match result to file"""
        results_file = self.data_dir / "results.json"
        
        # Load existing results
        results = []
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        # Add new result
        results.append(result)
        
        # Save updated results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    async def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions and results"""
        
        # Load results
        results_file = self.data_dir / "results.json"
        results = []
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        # Calculate statistics
        total_predictions = len([m for m in self.upcoming_matches.values() if m.predicted_winner])
        completed_matches = len(results)
        correct_predictions = len([r for r in results if r.get('prediction_correct')])
        
        accuracy = (correct_predictions / completed_matches * 100) if completed_matches > 0 else 0
        
        return {
            "total_upcoming_matches": len(self.upcoming_matches),
            "total_predictions_made": total_predictions,
            "completed_matches": completed_matches,
            "correct_predictions": correct_predictions,
            "accuracy_rate": f"{accuracy:.1f}%",
            "pending_matches": len([m for m in self.upcoming_matches.values() if m.status == "upcoming"]),
            "recent_results": results[-5:] if results else []
        }


# Global predictor instance
_predictor_instance = None

def get_predictor() -> UpcomingMatchesPredictor:
    """Get global predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = UpcomingMatchesPredictor()
    return _predictor_instance
