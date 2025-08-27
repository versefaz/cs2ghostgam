#!/usr/bin/env python3
"""
Show Predictions - Display today's match predictions with enhanced formatting
Provides comprehensive match analysis and real-time prediction updates
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
from app.utils.logger import setup_logger


def print_header():
    """Print formatted header with current time"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print("[*] CS2 MATCH PREDICTIONS - BLAST Open London 2025")
    print(f"[+] Generated: {current_time}")
    print("=" * 80)


def print_match_card(i: int, match: Dict[str, Any]) -> None:
    """Print a beautifully formatted match card with betting analysis"""
    print(f"\n+-- Match {i} {'-' * (65 - len(str(i)))}+")
    print(f"| [VS] {match['team1']} vs {match['team2']:<40} |")
    print(f"| [TIME] {match['scheduled_time']:<48} |")
    print(f"| [ODDS] {match['odds']:<48} |")
    
    if match['predicted_winner'] != 'No prediction yet':
        confidence_val = float(match['confidence'].rstrip('%'))
        confidence_indicator = "[HIGH]" if confidence_val > 70 else "[MED]" if confidence_val > 55 else "[LOW]"
        print(f"| {confidence_indicator} PREDICTION: {match['predicted_winner']} ({match['confidence']}) {'':>15} |")
        print(f"| [INFO] {match['reasoning'][:45]:<45} |")
        if len(match['reasoning']) > 45:
            remaining = match['reasoning'][45:]
            while remaining:
                chunk = remaining[:45]
                remaining = remaining[45:]
                print(f"|        {chunk:<45} |")
    else:
        print(f"| [WAIT] PREDICTION: Generating...{'':<30} |")
    
    # Display betting recommendations if available
    if 'betting_recommendations' in match and match['betting_recommendations']:
        print(f"| {'':>66} |")
        print(f"| [BETTING] TOP RECOMMENDATIONS: {'':>32} |")
        for j, rec in enumerate(match['betting_recommendations'][:3], 1):
            bet_line = f"{j}. {rec['selection']} @{rec['odds']} ({rec['confidence_level']})"
            print(f"| {bet_line:<64} |")
            ev_line = f"   EV: {rec['expected_value']:.1%} | Stake: {rec['stake_recommendation']:.1%}"
            print(f"| {ev_line:<64} |")
    
    print(f"+{'-' * 66}+")


def print_summary_section(summary: Dict[str, Any]) -> None:
    """Print formatted summary section"""
    print("\n" + "=" * 80)
    print("[SUMMARY] PREDICTION OVERVIEW")
    print("=" * 80)
    
    print(f"[STATS] Total matches today: {summary['total_upcoming_matches']}")
    print(f"[STATS] Predictions made: {summary['total_predictions_made']}")
    print(f"[STATS] Pending matches: {summary['pending_matches']}")
    
    if summary['completed_matches'] > 0:
        print(f"[DONE] Completed matches: {summary['completed_matches']}")
        print(f"[PERF] Accuracy rate: {summary['accuracy_rate']}")
        print(f"[PERF] Correct predictions: {summary['correct_predictions']}")
        
        if summary['recent_results']:
            print("\n[RESULTS] Recent Results:")
            for result in summary['recent_results'][-3:]:
                status = "[WIN]" if result.get('prediction_correct') else "[LOSS]"
                print(f"   {status} {result['teams']} - Winner: {result['winner']}")


async def print_betting_summary(matches: List[Dict[str, Any]]) -> None:
    """Print comprehensive betting summary"""
    print("\n" + "=" * 80)
    print("[BETTING] DEEP ANALYSIS SUMMARY")
    print("=" * 80)
    
    all_recommendations = []
    for match in matches:
        if 'betting_recommendations' in match:
            all_recommendations.extend(match['betting_recommendations'])
    
    if not all_recommendations:
        print("[INFO] No profitable betting opportunities found")
        return
    
    # Sort by expected value
    all_recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
    
    print(f"[STATS] Total betting opportunities analyzed: {len(all_recommendations)}")
    
    # Show top 3 recommendations across all matches
    print("\n[TOP BETS] Highest Expected Value Opportunities:")
    for i, rec in enumerate(all_recommendations[:3], 1):
        print(f"  {i}. {rec['selection']} @{rec['odds']} - EV: {rec['expected_value']:.1%} ({rec['confidence_level']})")
        print(f"     Risk: {rec['risk_level']} | Recommended Stake: {rec['stake_recommendation']:.1%} of bankroll")
        print(f"     Reason: {rec['reasoning'][:60]}...")
        print()
    
    # Betting type breakdown
    bet_types = {}
    for rec in all_recommendations:
        bet_type = rec['selection'].split()[0] if 'Over' in rec['selection'] or 'Under' in rec['selection'] else 'Match/Handicap'
        bet_types[bet_type] = bet_types.get(bet_type, 0) + 1
    
    print("[BREAKDOWN] Opportunities by Bet Type:")
    for bet_type, count in bet_types.items():
        print(f"  {bet_type}: {count} opportunities")
    
    # Risk level summary
    risk_levels = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
    for rec in all_recommendations:
        risk_levels[rec['risk_level']] += 1
    
    print("\n[RISK] Risk Distribution:")
    for risk, count in risk_levels.items():
        print(f"  {risk} Risk: {count} bets")


def print_footer() -> None:
    """Print formatted footer with instructions"""
    print("\n" + "=" * 80)
    print("[ACTIONS] NEXT STEPS")
    print("=" * 80)
    print("[CMD] To refresh predictions: python scripts/show_predictions.py")
    print("[CMD] To report match results: python scripts/report_result.py")
    print("[CMD] To view detailed analytics: python scripts/daily_report_generator.py")
    print("[WARN] BETTING DISCLAIMER: Gambling involves risk. Only bet what you can afford to lose.")
    print("=" * 80 + "\n")


async def save_predictions_snapshot(matches: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Save current predictions to a snapshot file"""
    try:
        snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = snapshot_dir / f"predictions_snapshot_{timestamp}.json"
        
        # Extract betting data for snapshot
        betting_summary = {
            "total_opportunities": sum(len(m.get('betting_recommendations', [])) for m in matches),
            "high_confidence_bets": sum(1 for m in matches for r in m.get('betting_recommendations', []) if r.get('confidence_level') == 'HIGH'),
            "avg_expected_value": sum(r.get('expected_value', 0) for m in matches for r in m.get('betting_recommendations', [])) / max(1, sum(len(m.get('betting_recommendations', [])) for m in matches))
        }
        
        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "matches": matches,
            "summary": summary,
            "betting_analysis": betting_summary
        }
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
        print(f"[SAVE] Snapshot saved: {snapshot_file.name}")
    except Exception as e:
        print(f"[WARN] Could not save snapshot - {e}")


async def main():
    """Enhanced main function with comprehensive prediction and betting analysis display"""
    # Setup logging
    logger = setup_logger("show_predictions")
    
    try:
        # Initialize systems
        print("[INIT] Initializing CS2 prediction system...")
        predictor = get_predictor()
        betting_analyzer = get_betting_analyzer()
        
        # Print header
        print_header()
        
        # Generate predictions with progress indication
        print("\n[AI] Generating predictions...")
        try:
            await predictor.generate_predictions()
            print("[OK] Predictions generated successfully!")
        except Exception as e:
            print(f"[ERROR] Error generating predictions: {e}")
            logger.error(f"Prediction generation failed: {e}")
            return
        
        # Get matches with predictions
        try:
            matches = await predictor.get_upcoming_matches_with_predictions()
            if not matches:
                print("\n[INFO] No upcoming matches found for today.")
                return
        except Exception as e:
            print(f"[ERROR] Error retrieving matches: {e}")
            logger.error(f"Failed to retrieve matches: {e}")
            return
        
        # Generate betting analysis for each match
        print("\n[BETTING] Analyzing betting opportunities...")
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
            print("[OK] Betting analysis completed!")
        except Exception as e:
            print(f"[WARN] Betting analysis failed: {e}")
            logger.warning(f"Betting analysis failed: {e}")
            # Continue without betting analysis
            for match in matches:
                match['betting_recommendations'] = []
        
        # Display matches with betting analysis
        print(f"\n[MATCHES] Today's Matches with Betting Analysis ({len(matches)} total)")
        for i, match in enumerate(matches, 1):
            print_match_card(i, match)
        
        # Display betting summary
        await print_betting_summary(matches)
        
        # Get and display summary
        try:
            summary = await predictor.get_prediction_summary()
            print_summary_section(summary)
        except Exception as e:
            print(f"[WARN] Could not generate summary - {e}")
            logger.warning(f"Summary generation failed: {e}")
            summary = {}
        
        # Save enhanced snapshot
        await save_predictions_snapshot(matches, summary)
        
        # Print footer
        print_footer()
        
        logger.info(f"Successfully displayed predictions and betting analysis for {len(matches)} matches")
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Operation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
