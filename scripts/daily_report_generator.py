#!/usr/bin/env python3
"""
Daily Report Generator - Day-by-Day CS2 Betting Performance Analysis
Generates comprehensive daily breakdown reports with detailed analytics
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.match_reporter import get_reporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generate detailed day-by-day performance reports"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.reporter = get_reporter()
    
    async def generate_daily_breakdown(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate day-by-day breakdown for the last N days"""
        try:
            # Get all completed reports
            completed_reports = self.reporter.completed_reports
            
            if not completed_reports:
                logger.info("No completed reports found")
                return {}
            
            # Group reports by date
            daily_data = defaultdict(list)
            
            for report in completed_reports:
                date_key = report.timestamp.date().isoformat()
                daily_data[date_key].append(report)
            
            # Generate daily breakdown
            daily_breakdown = {}
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            current_date = start_date
            while current_date <= end_date:
                date_key = current_date.isoformat()
                day_reports = daily_data.get(date_key, [])
                
                daily_stats = await self._calculate_daily_stats(day_reports, current_date)
                daily_breakdown[date_key] = daily_stats
                
                current_date += timedelta(days=1)
            
            return daily_breakdown
            
        except Exception as e:
            logger.error(f"Failed to generate daily breakdown: {e}")
            return {}
    
    async def _calculate_daily_stats(self, day_reports: List, date: datetime.date) -> Dict[str, Any]:
        """Calculate statistics for a single day"""
        if not day_reports:
            return {
                'date': date.isoformat(),
                'day_name': date.strftime('%A'),
                'matches_count': 0,
                'predictions_made': 0,
                'correct_predictions': 0,
                'accuracy_rate': 0.0,
                'profit_loss': 0.0,
                'roi': 0.0,
                'average_confidence': 0.0,
                'best_match': None,
                'worst_match': None,
                'matches': []
            }
        
        # Calculate basic stats
        predictions_made = len([r for r in day_reports if r.predicted_winner])
        correct_predictions = len([r for r in day_reports if r.prediction_correct])
        accuracy_rate = (correct_predictions / predictions_made * 100) if predictions_made > 0 else 0.0
        
        total_profit_loss = sum(r.profit_loss for r in day_reports if r.profit_loss is not None)
        total_bet_amount = sum(r.bet_amount for r in day_reports if r.bet_amount is not None)
        roi = (total_profit_loss / total_bet_amount * 100) if total_bet_amount > 0 else 0.0
        
        average_confidence = sum(r.confidence for r in day_reports if r.confidence) / len(day_reports) if day_reports else 0.0
        
        # Find best and worst matches
        profitable_matches = [r for r in day_reports if r.profit_loss and r.profit_loss > 0]
        losing_matches = [r for r in day_reports if r.profit_loss and r.profit_loss < 0]
        
        best_match = max(profitable_matches, key=lambda x: x.profit_loss) if profitable_matches else None
        worst_match = min(losing_matches, key=lambda x: x.profit_loss) if losing_matches else None
        
        # Format match details
        matches = []
        for report in day_reports:
            match_info = {
                'match_id': report.match_id,
                'teams': f"{report.team1} vs {report.team2}",
                'predicted_winner': report.predicted_winner,
                'actual_winner': report.actual_winner,
                'confidence': report.confidence,
                'correct': report.prediction_correct,
                'profit_loss': report.profit_loss,
                'roi': report.roi,
                'odds': f"{report.odds_team1:.2f} / {report.odds_team2:.2f}",
                'event': report.event_name
            }
            matches.append(match_info)
        
        return {
            'date': date.isoformat(),
            'day_name': date.strftime('%A'),
            'matches_count': len(day_reports),
            'predictions_made': predictions_made,
            'correct_predictions': correct_predictions,
            'accuracy_rate': round(accuracy_rate, 1),
            'profit_loss': round(total_profit_loss, 2),
            'roi': round(roi, 1),
            'average_confidence': round(average_confidence, 2),
            'best_match': {
                'teams': f"{best_match.team1} vs {best_match.team2}",
                'profit': best_match.profit_loss,
                'confidence': best_match.confidence
            } if best_match else None,
            'worst_match': {
                'teams': f"{worst_match.team1} vs {worst_match.team2}",
                'loss': worst_match.profit_loss,
                'confidence': worst_match.confidence
            } if worst_match else None,
            'matches': matches
        }
    
    async def generate_formatted_daily_report(self, days_back: int = 7) -> str:
        """Generate a formatted daily report for console output"""
        daily_breakdown = await self.generate_daily_breakdown(days_back)
        
        if not daily_breakdown:
            return "No data available for daily report"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CS2 BETTING SYSTEM - DAILY PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Report Period: Last {days_back} days")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        total_matches = sum(day['matches_count'] for day in daily_breakdown.values())
        total_predictions = sum(day['predictions_made'] for day in daily_breakdown.values())
        total_correct = sum(day['correct_predictions'] for day in daily_breakdown.values())
        total_profit = sum(day['profit_loss'] for day in daily_breakdown.values())
        
        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        
        report_lines.append("PERIOD SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Matches: {total_matches}")
        report_lines.append(f"Total Predictions: {total_predictions}")
        report_lines.append(f"Overall Accuracy: {overall_accuracy:.1f}%")
        report_lines.append(f"Total Profit/Loss: ${total_profit:.2f}")
        report_lines.append("")
        
        # Daily breakdown
        report_lines.append("DAILY BREAKDOWN")
        report_lines.append("-" * 40)
        
        # Sort by date (most recent first)
        sorted_days = sorted(daily_breakdown.items(), key=lambda x: x[0], reverse=True)
        
        for date_str, day_data in sorted_days:
            if day_data['matches_count'] == 0:
                continue  # Skip days with no matches
                
            report_lines.append(f"\n{day_data['day_name']}, {date_str}")
            report_lines.append(f"   Matches: {day_data['matches_count']} | Accuracy: {day_data['accuracy_rate']:.1f}% | P/L: ${day_data['profit_loss']:.2f}")
            
            if day_data['best_match']:
                report_lines.append(f"   [BEST] {day_data['best_match']['teams']} (+${day_data['best_match']['profit']:.2f})")
            
            if day_data['worst_match']:
                report_lines.append(f"   [WORST] {day_data['worst_match']['teams']} (${day_data['worst_match']['loss']:.2f})")
            
            # Show individual matches for the day
            for match in day_data['matches']:
                status_icon = "[WIN]" if match['correct'] else "[LOSS]"
                profit_str = f"+${match['profit_loss']:.2f}" if match['profit_loss'] > 0 else f"${match['profit_loss']:.2f}"
                report_lines.append(f"     {status_icon} {match['teams']} | {match['confidence']:.0%} confidence | {profit_str}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    async def save_daily_report_json(self, days_back: int = 30) -> str:
        """Save detailed daily breakdown to JSON file"""
        daily_breakdown = await self.generate_daily_breakdown(days_back)
        
        report_data = {
            'report_type': 'daily_breakdown',
            'generated_at': datetime.now().isoformat(),
            'period_days': days_back,
            'daily_data': daily_breakdown
        }
        
        filename = f"daily_breakdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Daily breakdown report saved: {filepath}")
        return str(filepath)
    
    async def get_weekly_summary(self) -> Dict[str, Any]:
        """Get weekly performance summary"""
        daily_breakdown = await self.generate_daily_breakdown(days_back=7)
        
        if not daily_breakdown:
            return {}
        
        # Calculate weekly stats
        week_data = [day for day in daily_breakdown.values() if day['matches_count'] > 0]
        
        if not week_data:
            return {}
        
        total_matches = sum(day['matches_count'] for day in week_data)
        total_predictions = sum(day['predictions_made'] for day in week_data)
        total_correct = sum(day['correct_predictions'] for day in week_data)
        total_profit = sum(day['profit_loss'] for day in week_data)
        
        best_day = max(week_data, key=lambda x: x['profit_loss']) if week_data else None
        worst_day = min(week_data, key=lambda x: x['profit_loss']) if week_data else None
        
        return {
            'period': 'Last 7 days',
            'active_days': len(week_data),
            'total_matches': total_matches,
            'total_predictions': total_predictions,
            'overall_accuracy': (total_correct / total_predictions * 100) if total_predictions > 0 else 0,
            'total_profit_loss': total_profit,
            'average_daily_profit': total_profit / len(week_data) if week_data else 0,
            'best_day': {
                'date': best_day['date'],
                'day_name': best_day['day_name'],
                'profit': best_day['profit_loss'],
                'accuracy': best_day['accuracy_rate']
            } if best_day else None,
            'worst_day': {
                'date': worst_day['date'],
                'day_name': worst_day['day_name'],
                'loss': worst_day['profit_loss'],
                'accuracy': worst_day['accuracy_rate']
            } if worst_day else None
        }


async def main():
    """Main function to generate and display daily reports"""
    try:
        generator = DailyReportGenerator()
        
        # Generate formatted daily report
        print("Generating daily performance report...")
        daily_report = await generator.generate_formatted_daily_report(days_back=7)
        print(daily_report)
        
        # Generate weekly summary
        weekly_summary = await generator.get_weekly_summary()
        if weekly_summary:
            print("\nWEEKLY SUMMARY")
            print("-" * 40)
            print(f"Active Days: {weekly_summary['active_days']}/7")
            print(f"Total Matches: {weekly_summary['total_matches']}")
            print(f"Overall Accuracy: {weekly_summary['overall_accuracy']:.1f}%")
            print(f"Total P/L: ${weekly_summary['total_profit_loss']:.2f}")
            print(f"Avg Daily P/L: ${weekly_summary['average_daily_profit']:.2f}")
            
            if weekly_summary['best_day']:
                print(f"Best Day: {weekly_summary['best_day']['day_name']} (+${weekly_summary['best_day']['profit']:.2f})")
            
            if weekly_summary['worst_day']:
                print(f"Worst Day: {weekly_summary['worst_day']['day_name']} (${weekly_summary['worst_day']['loss']:.2f})")
        
        # Save detailed JSON report
        json_file = await generator.save_daily_report_json(days_back=30)
        print(f"\nDetailed report saved: {json_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
