#!/usr/bin/env python3
"""
Match Predictor CLI - Interactive interface for CS2 match predictions and result reporting
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.upcoming_matches_predictor import get_predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatchPredictorCLI:
    """Command-line interface for match predictions"""
    
    def __init__(self):
        self.predictor = get_predictor()
    
    async def run(self):
        """Main CLI loop"""
        print("=" * 60)
        print("CS2 MATCH PREDICTOR - BLAST Open London 2025")
        print("=" * 60)
        
        while True:
            try:
                print("\nOptions:")
                print("1. Show upcoming matches with predictions")
                print("2. Generate new predictions")
                print("3. Report match result")
                print("4. Show prediction summary")
                print("5. Exit")
                
                choice = input("\nSelect option (1-5): ").strip()
                
                if choice == "1":
                    await self.show_upcoming_matches()
                elif choice == "2":
                    await self.generate_predictions()
                elif choice == "3":
                    await self.report_result()
                elif choice == "4":
                    await self.show_summary()
                elif choice == "5":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid option. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def show_upcoming_matches(self):
        """Display upcoming matches with predictions"""
        print("\n" + "=" * 60)
        print("UPCOMING MATCHES - BLAST Open London 2025")
        print("=" * 60)
        
        matches = await self.predictor.get_upcoming_matches_with_predictions()
        
        if not matches:
            print("No upcoming matches found.")
            return
        
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match['team1']} vs {match['team2']}")
            print(f"   Time: {match['scheduled_time']}")
            print(f"   Event: {match['event']}")
            print(f"   Odds: {match['odds']}")
            
            if match['predicted_winner']:
                print(f"   PREDICTION: {match['predicted_winner']} ({match['confidence']})")
                print(f"   Reasoning: {match['reasoning']}")
            else:
                print("   PREDICTION: Not generated yet")
    
    async def generate_predictions(self):
        """Generate predictions for all upcoming matches"""
        print("\nGenerating predictions for upcoming matches...")
        
        predictions = await self.predictor.generate_predictions()
        
        if not predictions:
            print("No matches to predict.")
            return
        
        print(f"\nGenerated {len(predictions)} predictions:")
        
        for match_id, prediction in predictions.items():
            match = self.predictor.upcoming_matches[match_id]
            print(f"\n{match.team1} vs {match.team2}")
            print(f"PREDICTED WINNER: {prediction.predicted_winner}")
            print(f"CONFIDENCE: {prediction.confidence:.1%}")
            print(f"REASONING: {prediction.reasoning}")
    
    async def report_result(self):
        """Allow user to report match results"""
        print("\n" + "=" * 60)
        print("REPORT MATCH RESULT")
        print("=" * 60)
        
        # Show available matches
        matches = []
        for match in self.predictor.upcoming_matches.values():
            if match.status == "upcoming":
                matches.append(match)
        
        if not matches:
            print("No upcoming matches to report results for.")
            return
        
        print("\nAvailable matches:")
        for i, match in enumerate(matches, 1):
            print(f"{i}. {match.team1} vs {match.team2} ({match.scheduled_time.strftime('%H:%M')})")
        
        try:
            choice = int(input(f"\nSelect match (1-{len(matches)}): ")) - 1
            if choice < 0 or choice >= len(matches):
                print("Invalid selection.")
                return
            
            selected_match = matches[choice]
            
            print(f"\nSelected: {selected_match.team1} vs {selected_match.team2}")
            print("Who won?")
            print(f"1. {selected_match.team1}")
            print(f"2. {selected_match.team2}")
            
            winner_choice = int(input("Select winner (1-2): "))
            if winner_choice == 1:
                winner = selected_match.team1
            elif winner_choice == 2:
                winner = selected_match.team2
            else:
                print("Invalid selection.")
                return
            
            user_name = input("Your name (optional): ").strip() or "Anonymous"
            
            result = await self.predictor.report_match_result(
                selected_match.match_id, winner, user_name
            )
            
            if result.get("success"):
                print(f"\n✓ {result['message']}")
                if result.get("prediction_correct") is not None:
                    if result["prediction_correct"]:
                        print("✓ Prediction was CORRECT!")
                    else:
                        print("✗ Prediction was INCORRECT.")
            else:
                print(f"Error: {result.get('error')}")
                
        except (ValueError, IndexError):
            print("Invalid input.")
    
    async def show_summary(self):
        """Show prediction summary and statistics"""
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        
        summary = await self.predictor.get_prediction_summary()
        
        print(f"Total upcoming matches: {summary['total_upcoming_matches']}")
        print(f"Predictions made: {summary['total_predictions_made']}")
        print(f"Completed matches: {summary['completed_matches']}")
        print(f"Correct predictions: {summary['correct_predictions']}")
        print(f"Accuracy rate: {summary['accuracy_rate']}")
        print(f"Pending matches: {summary['pending_matches']}")
        
        if summary['recent_results']:
            print("\nRecent results:")
            for result in summary['recent_results']:
                status = "✓" if result.get('prediction_correct') else "✗" if result.get('prediction_correct') is False else "?"
                print(f"  {status} {result['teams']} - Winner: {result['winner']}")


async def main():
    """Main entry point"""
    cli = MatchPredictorCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
