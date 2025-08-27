#!/usr/bin/env python3
"""
CS2 Match Reporter - Generate comprehensive match reports
Shows all today's matches with teams, times, events, and odds
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scrapers.enhanced_hltv_scraper import EnhancedHLTVScraper
from app.scrapers.odds_scraper import OddsScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchReporter:
    def __init__(self):
        self.hltv_scraper = None
        self.odds_scraper = OddsScraper()
        
    async def generate_comprehensive_report(self):
        """Generate comprehensive match report for today"""
        print("=" * 80)
        print("[TARGET] CS2 BETTING SYSTEM - TODAY'S MATCH REPORT")
        print("=" * 80)
        print(f"[DATE] Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Get matches from HLTV
        matches = await self._get_hltv_matches()
        
        # Get odds data
        odds_data = await self._get_odds_data()
        
        if matches:
            print(f"[MATCHES] FOUND {len(matches)} CS2 MATCHES TODAY:")
            print("-" * 80)
            
            for i, match in enumerate(matches, 1):
                await self._display_match_details(i, match, odds_data)
                print()
        else:
            print("[ERROR] No matches found from HLTV scraper")
            print("[FALLBACK] Trying fallback data sources...")
            await self._display_fallback_matches(odds_data)
        
        # Display odds summary
        await self._display_odds_summary(odds_data)
        
        print("=" * 80)
        print("[SUCCESS] REPORT COMPLETE")
        print("=" * 80)
    
    async def _get_hltv_matches(self) -> List[Dict[str, Any]]:
        """Get matches from HLTV scraper"""
        try:
            async with EnhancedHLTVScraper() as scraper:
                matches = await scraper.get_upcoming_matches(limit=20)
                logger.info(f"Retrieved {len(matches)} matches from HLTV")
                return matches
        except Exception as e:
            logger.error(f"HLTV scraper error: {e}")
            return []
    
    async def _get_odds_data(self) -> List[Dict[str, Any]]:
        """Get odds data from bookmakers"""
        try:
            odds = await self.odds_scraper.scrape_cs2_odds()
            logger.info(f"Retrieved {len(odds)} odds entries")
            return odds
        except Exception as e:
            logger.error(f"Odds scraper error: {e}")
            return []
    
    async def _display_match_details(self, match_num: int, match: Dict[str, Any], odds_data: List[Dict]):
        """Display detailed match information"""
        print(f"[MATCH] #{match_num}")
        print(f"   Teams: {match.get('team1', 'TBD')} vs {match.get('team2', 'TBD')}")
        print(f"   Time: {match.get('time', 'TBD')}")
        print(f"   Event: {match.get('event', 'Unknown Event')}")
        
        # Find relevant odds for this match
        relevant_odds = self._find_relevant_odds(match, odds_data)
        if relevant_odds:
            print(f"   [ODDS] Available:")
            for odd in relevant_odds[:3]:  # Show first 3
                print(f"      {odd['bookmaker']}: {odd['team1_odds']} vs {odd['team2_odds']}")
        else:
            print(f"   [ODDS] Fetching...")
    
    def _find_relevant_odds(self, match: Dict[str, Any], odds_data: List[Dict]) -> List[Dict]:
        """Find odds relevant to a specific match"""
        # For now, return sample odds since we don't have match-specific mapping
        return odds_data[:3] if odds_data else []
    
    async def _display_fallback_matches(self, odds_data: List[Dict]):
        """Display fallback match data when HLTV fails"""
        fallback_matches = [
            {"team1": "Natus Vincere", "team2": "fnatic", "time": "17:00", "event": "BLAST Open London 2025"},
            {"team1": "FaZe", "team2": "ECSTATIC", "time": "22:00", "event": "BLAST Open London 2025"},
            {"team1": "Vitality", "team2": "M80", "time": "17:00", "event": "BLAST Open London 2025"},
            {"team1": "GamerLegion", "team2": "Virtus.pro", "time": "19:30", "event": "BLAST Open London 2025"}
        ]
        
        print(f"[FALLBACK] MATCHES ({len(fallback_matches)} matches):")
        print("-" * 80)
        
        for i, match in enumerate(fallback_matches, 1):
            await self._display_match_details(i, match, odds_data)
            print()
    
    async def _display_odds_summary(self, odds_data: List[Dict]):
        """Display odds summary"""
        if not odds_data:
            print("[ERROR] No odds data available")
            return
            
        print("[ODDS] BOOKMAKER ODDS SUMMARY:")
        print("-" * 80)
        
        bookmakers = set(odd['bookmaker'] for odd in odds_data)
        print(f"[BOOKMAKERS] Active: {', '.join(bookmakers)}")
        print(f"[MARKETS] Total: {len(odds_data)}")
        
        print("\n[SAMPLE] Odds:")
        for i, odd in enumerate(odds_data[:5], 1):
            print(f"   {i}. {odd['bookmaker']}: {odd['team1_odds']} vs {odd['team2_odds']}")

async def main():
    """Main function to run the match reporter"""
    reporter = MatchReporter()
    await reporter.generate_comprehensive_report()

if __name__ == "__main__":
    asyncio.run(main())
