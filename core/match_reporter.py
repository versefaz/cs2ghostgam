#!/usr/bin/env python3
"""
CS2 Match Reporter - Comprehensive Match Analysis and Reporting System
Generates detailed reports on match predictions, outcomes, and betting performance
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
class MatchReport:
    """Comprehensive match report data structure"""
    match_id: str
    timestamp: datetime
    team1: str
    team2: str
    predicted_winner: Optional[str]
    actual_winner: Optional[str]
    confidence: float
    odds_team1: float
    odds_team2: float
    prediction_correct: Optional[bool]
    profit_loss: Optional[float]
    bet_amount: Optional[float]
    roi: Optional[float]
    match_status: str  # upcoming, live, completed
    event_name: Optional[str] = None
    map_predictions: Optional[Dict[str, Any]] = None
    market_analysis: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy_rate: float = 0.0
    total_profit_loss: float = 0.0
    total_bet_amount: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    average_confidence: float = 0.0
    best_streak: int = 0
    current_streak: int = 0
    last_updated: Optional[datetime] = None


class MatchReporter:
    """Advanced match reporting and analytics system"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for active reports
        self.active_matches: Dict[str, MatchReport] = {}
        self.completed_reports: List[MatchReport] = []
        self.performance_metrics = PerformanceMetrics()
        
        # Load existing reports
        self._load_existing_reports()
    
    def _load_existing_reports(self):
        """Load existing reports from disk"""
        try:
            reports_file = self.reports_dir / "match_reports.json"
            if reports_file.exists():
                with open(reports_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load completed reports
                for report_data in data.get('completed_reports', []):
                    report_data['timestamp'] = datetime.fromisoformat(report_data['timestamp'])
                    if report_data.get('last_updated'):
                        report_data['last_updated'] = datetime.fromisoformat(report_data['last_updated'])
                    self.completed_reports.append(MatchReport(**report_data))
                
                # Load performance metrics
                metrics_data = data.get('performance_metrics', {})
                if metrics_data.get('last_updated'):
                    metrics_data['last_updated'] = datetime.fromisoformat(metrics_data['last_updated'])
                self.performance_metrics = PerformanceMetrics(**metrics_data)
                
                self.logger.info(f"Loaded {len(self.completed_reports)} existing match reports")
        except Exception as e:
            self.logger.warning(f"Could not load existing reports: {e}")
    
    def _save_reports(self):
        """Save reports to disk"""
        try:
            reports_file = self.reports_dir / "match_reports.json"
            
            # Convert to serializable format
            completed_data = []
            for report in self.completed_reports:
                report_dict = asdict(report)
                report_dict['timestamp'] = report.timestamp.isoformat()
                completed_data.append(report_dict)
            
            metrics_dict = asdict(self.performance_metrics)
            if self.performance_metrics.last_updated:
                metrics_dict['last_updated'] = self.performance_metrics.last_updated.isoformat()
            
            data = {
                'completed_reports': completed_data,
                'performance_metrics': metrics_dict,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            with open(reports_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.debug("Reports saved to disk")
        except Exception as e:
            self.logger.error(f"Failed to save reports: {e}")
    
    async def create_match_report(self, match_data: Dict[str, Any], 
                                prediction_data: Optional[Dict[str, Any]] = None,
                                odds_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new match report"""
        try:
            match_id = match_data.get('match_id', f"match_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            
            # Extract match information
            team1 = match_data.get('team1', 'Team A')
            team2 = match_data.get('team2', 'Team B')
            
            # Extract prediction information
            predicted_winner = None
            confidence = 0.0
            if prediction_data:
                predicted_winner = prediction_data.get('predicted_winner')
                confidence = prediction_data.get('confidence', 0.0)
            
            # Extract odds information
            odds_team1 = 2.0
            odds_team2 = 2.0
            if odds_data:
                odds_team1 = odds_data.get('team1_odds', 2.0)
                odds_team2 = odds_data.get('team2_odds', 2.0)
            
            # Create match report
            report = MatchReport(
                match_id=match_id,
                timestamp=datetime.utcnow(),
                team1=team1,
                team2=team2,
                predicted_winner=predicted_winner,
                actual_winner=None,
                confidence=confidence,
                odds_team1=odds_team1,
                odds_team2=odds_team2,
                prediction_correct=None,
                profit_loss=None,
                bet_amount=None,
                roi=None,
                match_status='upcoming',
                event_name=match_data.get('event', 'CS2 Match'),
                map_predictions=prediction_data.get('map_predictions') if prediction_data else None,
                market_analysis=odds_data.get('market_analysis') if odds_data else None
            )
            
            self.active_matches[match_id] = report
            self.logger.info(f"Created match report: {team1} vs {team2} (ID: {match_id})")
            
            return match_id
            
        except Exception as e:
            self.logger.error(f"Failed to create match report: {e}")
            raise
    
    async def update_match_status(self, match_id: str, status: str, 
                                actual_winner: Optional[str] = None):
        """Update match status and outcome"""
        try:
            if match_id not in self.active_matches:
                self.logger.warning(f"Match {match_id} not found in active matches")
                return
            
            report = self.active_matches[match_id]
            report.match_status = status
            
            if actual_winner:
                report.actual_winner = actual_winner
                
                # Calculate prediction accuracy
                if report.predicted_winner:
                    report.prediction_correct = (report.predicted_winner == actual_winner)
                    
                    # Calculate profit/loss (simplified)
                    if report.prediction_correct:
                        winning_odds = report.odds_team1 if actual_winner == report.team1 else report.odds_team2
                        report.bet_amount = 100.0  # Standard bet amount
                        report.profit_loss = report.bet_amount * (winning_odds - 1)
                        report.roi = (report.profit_loss / report.bet_amount) * 100
                    else:
                        report.bet_amount = 100.0
                        report.profit_loss = -report.bet_amount
                        report.roi = -100.0
            
            if status == 'completed':
                # Move to completed reports
                self.completed_reports.append(report)
                del self.active_matches[match_id]
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                self.logger.info(f"Match completed: {report.team1} vs {report.team2}, Winner: {actual_winner}")
            
            # Save reports
            self._save_reports()
            
        except Exception as e:
            self.logger.error(f"Failed to update match status: {e}")
    
    async def _update_performance_metrics(self):
        """Update overall performance metrics"""
        try:
            if not self.completed_reports:
                return
            
            # Calculate metrics from completed reports
            total_predictions = len([r for r in self.completed_reports if r.predicted_winner])
            correct_predictions = len([r for r in self.completed_reports if r.prediction_correct])
            
            accuracy_rate = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
            
            total_profit_loss = sum(r.profit_loss for r in self.completed_reports if r.profit_loss)
            total_bet_amount = sum(r.bet_amount for r in self.completed_reports if r.bet_amount)
            
            roi = (total_profit_loss / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
            win_rate = accuracy_rate  # Same as accuracy for now
            
            average_confidence = sum(r.confidence for r in self.completed_reports if r.confidence) / len(self.completed_reports)
            
            # Calculate streaks
            current_streak = 0
            best_streak = 0
            temp_streak = 0
            
            for report in reversed(self.completed_reports):
                if report.prediction_correct is True:
                    temp_streak += 1
                    if current_streak == 0:  # Still in current streak
                        current_streak = temp_streak
                else:
                    if current_streak == 0:  # Current streak broken
                        current_streak = 0
                    temp_streak = 0
                
                best_streak = max(best_streak, temp_streak)
            
            # Update metrics
            self.performance_metrics = PerformanceMetrics(
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                accuracy_rate=accuracy_rate,
                total_profit_loss=total_profit_loss,
                total_bet_amount=total_bet_amount,
                roi=roi,
                win_rate=win_rate,
                average_confidence=average_confidence,
                best_streak=best_streak,
                current_streak=current_streak,
                last_updated=datetime.utcnow()
            )
            
            self.logger.info(f"Performance metrics updated: {accuracy_rate:.1f}% accuracy, {roi:.1f}% ROI")
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        try:
            today = datetime.utcnow().date()
            today_reports = [r for r in self.completed_reports 
                           if r.timestamp.date() == today]
            
            # Daily statistics
            daily_stats = {
                'date': today.isoformat(),
                'matches_analyzed': len(today_reports),
                'predictions_made': len([r for r in today_reports if r.predicted_winner]),
                'correct_predictions': len([r for r in today_reports if r.prediction_correct]),
                'daily_accuracy': 0.0,
                'daily_profit_loss': 0.0,
                'daily_roi': 0.0,
                'top_matches': []
            }
            
            if today_reports:
                correct_today = len([r for r in today_reports if r.prediction_correct])
                predictions_today = len([r for r in today_reports if r.predicted_winner])
                
                daily_stats['daily_accuracy'] = (correct_today / predictions_today * 100) if predictions_today > 0 else 0.0
                daily_stats['daily_profit_loss'] = sum(r.profit_loss for r in today_reports if r.profit_loss)
                
                # Top matches by confidence
                top_matches = sorted([r for r in today_reports if r.confidence], 
                                   key=lambda x: x.confidence, reverse=True)[:5]
                
                daily_stats['top_matches'] = [
                    {
                        'teams': f"{r.team1} vs {r.team2}",
                        'predicted_winner': r.predicted_winner,
                        'actual_winner': r.actual_winner,
                        'confidence': r.confidence,
                        'correct': r.prediction_correct,
                        'profit_loss': r.profit_loss
                    }
                    for r in top_matches
                ]
            
            # Overall performance summary
            overall_stats = asdict(self.performance_metrics)
            if self.performance_metrics.last_updated:
                overall_stats['last_updated'] = self.performance_metrics.last_updated.isoformat()
            
            report = {
                'daily_performance': daily_stats,
                'overall_performance': overall_stats,
                'active_matches': len(self.active_matches),
                'report_generated_at': datetime.utcnow().isoformat()
            }
            
            # Save daily report
            daily_report_file = self.reports_dir / f"daily_report_{today.strftime('%Y%m%d')}.json"
            with open(daily_report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Daily report generated: {daily_stats['matches_analyzed']} matches, "
                           f"{daily_stats['daily_accuracy']:.1f}% accuracy")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")
            return {}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            return {
                'total_predictions': self.performance_metrics.total_predictions,
                'accuracy_rate': f"{self.performance_metrics.accuracy_rate:.1f}%",
                'roi': f"{self.performance_metrics.roi:.1f}%",
                'profit_loss': f"${self.performance_metrics.total_profit_loss:.2f}",
                'current_streak': self.performance_metrics.current_streak,
                'best_streak': self.performance_metrics.best_streak,
                'active_matches': len(self.active_matches),
                'last_updated': self.performance_metrics.last_updated.isoformat() if self.performance_metrics.last_updated else None
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    async def cleanup_old_reports(self, days_to_keep: int = 30):
        """Clean up old reports to prevent storage bloat"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Remove old completed reports
            old_count = len(self.completed_reports)
            self.completed_reports = [r for r in self.completed_reports 
                                    if r.timestamp > cutoff_date]
            
            removed_count = old_count - len(self.completed_reports)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old reports (older than {days_to_keep} days)")
                self._save_reports()
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old reports: {e}")


# Global reporter instance
_reporter_instance = None

def get_reporter() -> MatchReporter:
    """Get global reporter instance"""
    global _reporter_instance
    if _reporter_instance is None:
        _reporter_instance = MatchReporter()
    return _reporter_instance
