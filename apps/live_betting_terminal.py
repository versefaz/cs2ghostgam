#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Betting Terminal - แอปเดิมพันระดับโลกแบบ Real-time
Created by KoJao - Professional CS2 Betting Analytics
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List
import logging

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.live_odds_fetcher import LiveOddsFetcher, get_live_odds_fetcher
from core.enhanced_team_analyzer import get_enhanced_analyzer
from core.deep_betting_analyzer import get_deep_betting_analyzer
from app.utils.logger import setup_logger

class LiveBettingTerminal:
    """แอปเดิมพันสดระดับโลก"""
    
    def __init__(self):
        self.logger = setup_logger("live_betting_terminal")
        self.odds_fetcher = get_live_odds_fetcher()
        self.team_analyzer = get_enhanced_analyzer()
        self.betting_analyzer = get_deep_betting_analyzer()
        
        # BLAST Open London 2025 matches
        self.today_matches = [
            ("Virtus.pro", "GamerLegion", "20:30"),
            ("ECSTATIC", "FaZe", "23:00"),
            ("NAVI", "Fnatic", "01:30")
        ]
    
    def print_header(self):
        """แสดงหัวข้อแอป"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀" * 50)
        print("💎 LIVE BETTING TERMINAL - WORLD CLASS CS2 ANALYTICS 💎")
        print("⚡ Created by KoJao - Professional Betting System ⚡")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🎯 BLAST Open London 2025")
        print("🚀" * 50)
        print()
    
    def print_live_odds_table(self, match_odds: Dict):
        """แสดงตารางราคาสดแบบมืออาชีพ"""
        
        for match_key, best_odds_list in match_odds.items():
            team1, team2 = match_key.replace("_vs_", " vs ").split(" vs ")
            
            print(f"🔥 {team1} vs {team2} - LIVE ODDS")
            print("┌" + "─" * 120 + "┐")
            print("│ Bet Type          │ Selection               │ Best Odds │ Bookmaker │ EV     │ Risk │ All Odds            │")
            print("├" + "─" * 120 + "┤")
            
            if not best_odds_list:
                print("│ ❌ No live odds available - Check again in a few minutes                                                   │")
                print("└" + "─" * 120 + "┘")
                continue
            
            for best_odd in best_odds_list[:8]:  # แสดงแค่ 8 อันดับแรก
                # สีสำหรับ EV
                ev_color = "🟢" if best_odd.expected_value > 10 else "🟡" if best_odd.expected_value > 5 else "🔴"
                
                # รวมราคาจากเว็บอื่น
                other_odds = ", ".join([f"{o.bookmaker}:{o.odds:.2f}" for o in best_odd.all_odds[:3]])
                if len(best_odd.all_odds) > 3:
                    other_odds += f" +{len(best_odd.all_odds)-3} more"
                
                print(f"│ {best_odd.bet_type:<17} │ {best_odd.selection:<23} │ {best_odd.best_odds:<9.2f} │ {best_odd.best_bookmaker:<9} │ {ev_color}{best_odd.expected_value:+5.1f}% │ {best_odd.risk_level:<4} │ {other_odds:<19} │")
            
            print("└" + "─" * 120 + "┘")
            print()
    
    def print_betting_recommendations(self, match_odds: Dict):
        """แสดงคำแนะนำการเดิมพันจากราคาจริง"""
        
        print("💰 PROFESSIONAL BETTING RECOMMENDATIONS")
        print("=" * 120)
        
        all_recommendations = []
        
        for match_key, best_odds_list in match_odds.items():
            team1, team2 = match_key.replace("_vs_", " vs ").split(" vs ")
            
            # หาโอกาสดีที่สุด
            top_opportunities = [odd for odd in best_odds_list if odd.expected_value > 5][:3]
            
            for i, opp in enumerate(top_opportunities, 1):
                stake = self._calculate_stake(opp.expected_value, opp.risk_level)
                
                all_recommendations.append({
                    'match': f"{team1} vs {team2}",
                    'rank': len(all_recommendations) + 1,
                    'bet_type': opp.bet_type,
                    'selection': opp.selection,
                    'odds': opp.best_odds,
                    'bookmaker': opp.best_bookmaker,
                    'expected_value': opp.expected_value,
                    'risk_level': opp.risk_level,
                    'stake': stake,
                    'reasoning': self._generate_reasoning(opp, team1, team2)
                })
        
        # เรียงตาม EV
        all_recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # แสดงแค่ 5 อันดับแรก
        for rec in all_recommendations[:5]:
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            
            print(f"🏆 RANK #{rec['rank']}: {rec['match']}")
            print(f"┌─ {rec['bet_type']}: {rec['selection']} ─────────────────────────────────────────────────────┐")
            print(f"│ 💵 Best Odds: {rec['odds']:.2f} @ {rec['bookmaker']:<12} │ 📊 EV: {rec['expected_value']:+.1f}% │ {risk_emoji[rec['risk_level']]} {rec['risk_level']} Risk │")
            print(f"│ 💰 Recommended Stake: {rec['stake']} of bankroll                                           │")
            print(f"│ 🧠 Analysis: {rec['reasoning']:<65} │")
            print("└─────────────────────────────────────────────────────────────────────────────────────────┘")
            print()
    
    def _calculate_stake(self, expected_value: float, risk_level: str) -> str:
        """คำนวณขนาดเดิมพันที่แนะนำ"""
        base_stake = expected_value / 100 * 0.5  # Kelly Criterion แบบ conservative
        
        if risk_level == "LOW":
            multiplier = 1.2
        elif risk_level == "MEDIUM":
            multiplier = 1.0
        else:  # HIGH
            multiplier = 0.7
        
        final_stake = base_stake * multiplier
        final_stake = max(0.01, min(0.05, final_stake))  # จำกัดระหว่าง 1-5%
        
        return f"{final_stake:.1%}"
    
    def _generate_reasoning(self, opportunity, team1: str, team2: str) -> str:
        """สร้างเหตุผลการเดิมพัน"""
        
        reasoning_map = {
            ("Virtus.pro", "GamerLegion"): {
                "Match Winner": "GL has better recent form (3 wins vs 1), close rankings",
                "Handicap": "Very close matchup, handicap provides safety margin",
                "Total Maps": "Both teams strong on different maps, likely goes 3 maps"
            },
            ("ECSTATIC", "FaZe"): {
                "Match Winner": "jcobbb debut creates uncertainty, ECSTATIC upset potential",
                "Handicap": "FaZe roster changes, ECSTATIC strong on Vertigo/Overpass",
                "Total Maps": "New FaZe lineup may struggle initially"
            },
            ("NAVI", "Fnatic"): {
                "Match Winner": "NAVI much stronger (#6 vs #34), should dominate",
                "Handicap": "Large skill gap, NAVI should win comfortably",
                "Total Maps": "Likely quick 2-0 for NAVI"
            }
        }
        
        match_key = (team1, team2)
        if match_key in reasoning_map and opportunity.bet_type in reasoning_map[match_key]:
            return reasoning_map[match_key][opportunity.bet_type]
        
        return f"Value bet based on {opportunity.expected_value:.1f}% EV"
    
    def print_portfolio_summary(self, recommendations: List):
        """แสดงสรุปพอร์ตการเดิมพัน"""
        
        total_stake = sum(float(rec['stake'].rstrip('%')) for rec in recommendations[:5]) / 100
        total_ev = sum(rec['expected_value'] for rec in recommendations[:5]) / len(recommendations[:5])
        
        print("📊 PORTFOLIO SUMMARY")
        print("┌" + "─" * 80 + "┐")
        print(f"│ Total Recommended Stake: {total_stake:.1%} of bankroll                           │")
        print(f"│ Average Expected Value: {total_ev:+.1f}%                                        │")
        print(f"│ Number of Bets: {len(recommendations[:5])}                                                    │")
        print(f"│ Risk Distribution: Diversified across {len(set(r['match'] for r in recommendations[:5]))} matches                     │")
        print("└" + "─" * 80 + "┘")
        print()
        
        print("⚠️  RISK MANAGEMENT RULES:")
        print("• Never bet more than 25% of total bankroll in one day")
        print("• Stop betting if down 10% for the day")
        print("• Always verify odds before placing bets")
        print("• Monitor line movements and bet early for best value")
        print()
    
    async def run_live_analysis(self):
        """รันการวิเคราะห์สด"""
        
        self.print_header()
        
        print("🔄 FETCHING LIVE ODDS FROM MULTIPLE BOOKMAKERS...")
        print("⏳ Please wait while we scan Pinnacle, GG.bet, Betway and others...")
        print()
        
        try:
            # ดึงราคาสด
            async with self.odds_fetcher as fetcher:
                match_teams = [(m[0], m[1]) for m in self.today_matches]
                live_odds = await fetcher.fetch_live_odds(match_teams)
            
            if not live_odds:
                print("❌ Unable to fetch live odds at this time")
                print("🔄 This could be due to:")
                print("   • Matches not yet available for betting")
                print("   • Bookmaker websites blocking requests")
                print("   • Network connectivity issues")
                print()
                print("💡 Try again in 30 minutes or check odds manually")
                return
            
            # แสดงราคาสด
            self.print_live_odds_table(live_odds)
            
            # แสดงคำแนะนำ
            self.print_betting_recommendations(live_odds)
            
            # สรุปพอร์ต
            all_recs = []
            for odds_list in live_odds.values():
                for odd in odds_list:
                    if odd.expected_value > 5:
                        all_recs.append({
                            'expected_value': odd.expected_value,
                            'stake': self._calculate_stake(odd.expected_value, odd.risk_level),
                            'match': f"{odd.bet_type} - {odd.selection}"
                        })
            
            self.print_portfolio_summary(all_recs)
            
            # แสดงเวลาอัปเดตล่าสุด
            print(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("🔄 Refresh every 5-10 minutes for latest odds")
            
        except Exception as e:
            self.logger.error(f"Error in live analysis: {e}")
            print(f"❌ Error occurred: {e}")
            print("🔄 Please try again or check your internet connection")
    
    async def run_continuous_monitoring(self):
        """รันการติดตามต่อเนื่อง"""
        
        while True:
            await self.run_live_analysis()
            
            print("\n" + "="*50)
            print("⏰ Waiting 5 minutes for next update...")
            print("Press Ctrl+C to stop monitoring")
            print("="*50)
            
            try:
                await asyncio.sleep(300)  # 5 minutes
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                break

async def main():
    """เริ่มต้นแอป"""
    
    terminal = LiveBettingTerminal()
    
    print("🎯 LIVE BETTING TERMINAL")
    print("1. Single Analysis (one-time)")
    print("2. Continuous Monitoring (every 5 minutes)")
    print()
    
    try:
        choice = input("Select option (1 or 2): ").strip()
        
        if choice == "1":
            await terminal.run_live_analysis()
        elif choice == "2":
            await terminal.run_continuous_monitoring()
        else:
            print("Invalid choice. Running single analysis...")
            await terminal.run_live_analysis()
    
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
