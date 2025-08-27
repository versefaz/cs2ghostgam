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
from app.utils.logger import setup_logger


def print_header():
    """Print formatted header with current time"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print("[*] CS2 MATCH PREDICTIONS - BLAST Open London 2025")
    print(f"[+] Generated: {current_time}")
    print("=" * 80)


def print_match_card(i: int, match: Dict[str, Any]) -> None:
    """Print a beautifully formatted match card"""
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


def print_footer() -> None:
    """Print formatted footer with instructions"""
    print("\n" + "=" * 80)
    print("[ACTIONS] NEXT STEPS")
    print("=" * 80)
    print("[CMD] To refresh predictions: python scripts/show_predictions.py")
    print("[CMD] To report match results: python scripts/report_result.py")
    print("[CMD] To view detailed analytics: python scripts/daily_report_generator.py")
    print("=" * 80 + "\n")


async def save_predictions_snapshot(matches: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    """Save current predictions to a snapshot file"""
    try:
        snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = snapshot_dir / f"predictions_snapshot_{timestamp}.json"
        
        snapshot_data = {
            "timestamp": datetime.now().isoformat(),
            "matches": matches,
            "summary": summary
        }
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
        print(f"[SAVE] Snapshot saved: {snapshot_file.name}")
    except Exception as e:
        print(f"[WARN] Could not save snapshot - {e}")


async def main():
    """Enhanced main function with comprehensive prediction display"""
    # Setup logging
    logger = setup_logger("show_predictions")
    
    try:
        # Initialize predictor
        print("[INIT] Initializing CS2 prediction system...")
        predictor = get_predictor()
        
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
        
        # Display matches
        print(f"\n[MATCHES] Today's Matches ({len(matches)} total)")
        for i, match in enumerate(matches, 1):
            print_match_card(i, match)
        
        # Get and display summary
        try:
            summary = await predictor.get_prediction_summary()
            print_summary_section(summary)
        except Exception as e:
            print(f"[WARN] Could not generate summary - {e}")
            logger.warning(f"Summary generation failed: {e}")
            summary = {}
        
        # Save snapshot
        await save_predictions_snapshot(matches, summary)
        
        # Print footer
        print_footer()
        
        logger.info(f"Successfully displayed predictions for {len(matches)} matches")
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Operation cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
