import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

# App package imports (stubs needed)
from app.database import get_db_session
from app.scrapers import HLTVScraper, VLRScraper, OddsScraper
from app.models import Match, Team, Player, PlayerPerformance
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HistoricalDataCollector:
    """Collect and store historical CS2 data"""

    def __init__(self):
        self.hltv = HLTVScraper()
        self.vlr = VLRScraper()
        self.odds = OddsScraper()
        self.stats = {
            'matches': 0,
            'teams': 0,
            'players': 0,
            'performances': 0,
            'errors': 0
        }

    async def collect_historical_matches(self, days_back: int = 30) -> List[Dict]:
        """Collect historical match data"""
        logger.info(f"📅 Collecting matches from last {days_back} days")

        all_matches: List[Dict[str, Any]] = []
        start_date = datetime.now() - timedelta(days=days_back)

        # Collect from HLTV
        try:
            hltv_matches = await self.hltv.scrape_historical_matches(
                start_date=start_date,
                end_date=datetime.now()
            )
            all_matches.extend(hltv_matches)
            logger.info(f"✅ Collected {len(hltv_matches)} matches from HLTV")
        except Exception as e:
            logger.error(f"❌ HLTV error: {e}")
            self.stats['errors'] += 1

        # Collect from VLR (if available)
        try:
            vlr_matches = await self.vlr.scrape_historical_matches(
                start_date=start_date,
                end_date=datetime.now()
            )
            all_matches.extend(vlr_matches)
            logger.info(f"✅ Collected {len(vlr_matches)} matches from VLR")
        except Exception as e:
            logger.error(f"❌ VLR error: {e}")
            self.stats['errors'] += 1

        return all_matches

    async def collect_match_details(self, match_id: str) -> Dict:
        """Collect detailed match information"""
        details: Dict[str, Any] = {}

        try:
            # Get match details
            match_info = await self.hltv.scrape_match_details(match_id)

            # Get player performances
            performances = await self.hltv.scrape_match_performances(match_id)

            # Get round history if available
            rounds = await self.hltv.scrape_round_history(match_id)

            details = {
                'info': match_info,
                'performances': performances,
                'rounds': rounds
            }

        except Exception as e:
            logger.error(f"Error collecting details for match {match_id}: {e}")
            self.stats['errors'] += 1

        return details

    async def process_and_store_matches(self, matches: List[Dict], session: AsyncSession):
        """Process and store match data in database"""
        from tqdm.asyncio import tqdm
        logger.info(f"💾 Processing {len(matches)} matches")

        for match_data in tqdm(matches, desc="Processing matches"):
            try:
                # Check if match already exists
                existing = await session.get(Match, match_data['match_id'])
                if existing:
                    continue

                # Create match record
                match = Match(
                    external_id=match_data['match_id'],
                    team1_name=match_data['team1_name'],
                    team2_name=match_data['team2_name'],
                    team1_score=match_data.get('team1_score'),
                    team2_score=match_data.get('team2_score'),
                    match_time=match_data['match_time'],
                    event_name=match_data['event_name'],
                    map_name=match_data.get('map'),
                    source='hltv',
                    status='completed'
                )
                session.add(match)
                self.stats['matches'] += 1

                # Get and store match details
                if match_data.get('match_id'):
                    details = await self.collect_match_details(match_data['match_id'])

                    # Store player performances
                    if details.get('performances'):
                        for perf in details['performances']:
                            performance = PlayerPerformance(
                                match_id=match.id,
                                player_name=perf['player_name'],
                                team_name=perf['team_name'],
                                kills=perf.get('kills', 0),
                                deaths=perf.get('deaths', 0),
                                assists=perf.get('assists', 0),
                                adr=perf.get('adr', 0),
                                kast=perf.get('kast', 0),
                                rating=perf.get('rating', 0)
                            )
                            session.add(performance)
                            self.stats['performances'] += 1

                # Commit every 50 matches
                if self.stats['matches'] % 50 == 0:
                    await session.commit()
                    logger.info(f"📊 Progress: {self.stats['matches']} matches saved")

            except Exception as e:
                logger.error(f"Error processing match: {e}")
                self.stats['errors'] += 1
                continue

        await session.commit()

    async def collect_team_rosters(self, session: AsyncSession):
        """Collect current team rosters"""
        from tqdm.asyncio import tqdm
        logger.info("👥 Collecting team rosters")

        try:
            # Get top teams
            teams = await self.hltv.scrape_top_teams()

            for team_data in tqdm(teams[:30], desc="Processing teams"):
                # Check if team exists
                existing = await session.get(Team, team_data['team_id'])
                if existing:
                    continue

                # Create team record
                team = Team(
                    external_id=team_data['team_id'],
                    name=team_data['name'],
                    ranking=team_data.get('ranking'),
                    region=team_data.get('region'),
                    created_at=datetime.utcnow()
                )
                session.add(team)
                self.stats['teams'] += 1

                # Get team roster
                roster = await self.hltv.scrape_team_roster(team_data['team_id'])

                for player_data in roster:
                    player = Player(
                        external_id=player_data['player_id'],
                        name=player_data['name'],
                        team_id=team.id,
                        role=player_data.get('role'),
                        rating=player_data.get('rating'),
                        created_at=datetime.utcnow()
                    )
                    session.add(player)
                    self.stats['players'] += 1

            await session.commit()
            logger.info(f"✅ Saved {self.stats['teams']} teams and {self.stats['players']} players")

        except Exception as e:
            logger.error(f"Error collecting rosters: {e}")
            self.stats['errors'] += 1

    async def run(self, days_back: int = 30):
        """Run historical data collection"""
        start_time = datetime.now()
        logger.info(f"""
        {'='*60}
        🚀 STARTING HISTORICAL DATA COLLECTION
        📅 Period: Last {days_back} days
        🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
        {'='*60}
        """)

        async with get_db_session() as session:
            # Collect matches
            matches = await self.collect_historical_matches(days_back)

            # Process and store
            await self.process_and_store_matches(matches, session)

            # Collect team rosters
            await self.collect_team_rosters(session)

        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"""
        {'='*60}
        ✅ HISTORICAL DATA COLLECTION COMPLETED
        {'='*60}
        📊 Summary:
          - Matches: {self.stats['matches']}
          - Teams: {self.stats['teams']}
          - Players: {self.stats['players']}
          - Performances: {self.stats['performances']}
          - Errors: {self.stats['errors']}
        ⏱️  Duration: {duration:.2f} seconds
        {'='*60}
        """)

        return self.stats

# Main execution
async def main():
    collector = HistoricalDataCollector()

    # Collect last 30 days of data
    stats = await collector.run(days_back=30)

    # Continue with real-time collection
    if stats['matches'] > 0:
        logger.info("✅ Historical data collected successfully!")
        logger.info("🔄 You can now start real-time scraping...")
    else:
        logger.warning("⚠️ No historical data collected. Check scrapers.")

if __name__ == "__main__":
    asyncio.run(main())
