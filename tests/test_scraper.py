import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

# NOTE: These imports expect an app/ package. We'll provide stubs shortly.
from app.scrapers.hltv_scraper import HLTVScraper
from app.scrapers.vlr_scraper import VLRScraper
from app.scrapers.odds_scraper import OddsScraper
from app.models import Match, Team, Player, LiveOdds
from app.database import get_db_session


class TestScraperIntegration:
    """Integration tests for all scrapers"""

    @pytest.fixture
    async def db_session(self):
        """Create test database session"""
        async with get_db_session() as session:
            yield session

    @pytest.fixture
    def hltv_scraper(self):
        """Initialize HLTV scraper"""
        return HLTVScraper()

    @pytest.fixture
    def vlr_scraper(self):
        """Initialize VLR scraper"""
        return VLRScraper()

    @pytest.fixture
    def odds_scraper(self):
        """Initialize Odds scraper"""
        return OddsScraper()

    @pytest.mark.asyncio
    async def test_hltv_match_scraping(self, hltv_scraper, db_session):
        """Test HLTV match data collection"""
        print("🔍 Testing HLTV Match Scraping...")

        # Scrape upcoming matches
        matches = await hltv_scraper.scrape_upcoming_matches()

        assert len(matches) > 0, "No matches found"

        # Validate match data structure
        for match in matches[:5]:  # Test first 5 matches
            assert match.get('match_id'), "Missing match_id"
            assert match.get('team1_name'), "Missing team1"
            assert match.get('team2_name'), "Missing team2"
            assert match.get('match_time'), "Missing match_time"
            assert match.get('event_name'), "Missing event"

            # Save to database
            db_match = Match(
                external_id=match['match_id'],
                team1_name=match['team1_name'],
                team2_name=match['team2_name'],
                match_time=match['match_time'],
                event_name=match['event_name'],
                source='hltv',
                status='upcoming'
            )
            db_session.add(db_match)

        await db_session.commit()
        print(f"✅ Successfully scraped {len(matches)} matches from HLTV")
        return True

    @pytest.mark.asyncio
    async def test_player_stats_scraping(self, hltv_scraper, db_session):
        """Test player statistics collection"""
        print("🔍 Testing Player Stats Scraping...")

        # Get top players
        players = await hltv_scraper.scrape_top_players()

        assert len(players) > 0, "No players found"

        for player in players[:10]:  # Test first 10 players
            assert player.get('player_id'), "Missing player_id"
            assert player.get('name'), "Missing player name"
            assert player.get('rating') is not None, "Missing rating"
            assert player.get('kd_ratio') is not None, "Missing K/D"

            # Validate data ranges
            assert 0.5 <= player['rating'] <= 2.0, f"Invalid rating: {player['rating']}"
            assert 0.3 <= player['kd_ratio'] <= 3.0, f"Invalid K/D: {player['kd_ratio']}"

            # Save to database
            db_player = Player(
                external_id=player['player_id'],
                name=player['name'],
                team_name=player.get('team'),
                rating=player['rating'],
                kd_ratio=player['kd_ratio'],
                adr=player.get('adr', 0),
                kast=player.get('kast', 0)
            )
            db_session.add(db_player)

        await db_session.commit()
        print(f"✅ Successfully scraped {len(players)} player stats")
        return True

    @pytest.mark.asyncio
    async def test_live_odds_scraping(self, odds_scraper, db_session):
        """Test live odds collection from multiple bookmakers"""
        print("🔍 Testing Live Odds Scraping...")

        # Get live odds for CS2 matches
        odds_data = await odds_scraper.scrape_cs2_odds()

        assert len(odds_data) > 0, "No odds found"

        for odds in odds_data[:5]:
            assert odds.get('match_id'), "Missing match_id"
            assert odds.get('bookmaker'), "Missing bookmaker"
            assert odds.get('team1_odds'), "Missing team1_odds"
            assert odds.get('team2_odds'), "Missing team2_odds"

            # Validate odds ranges
            assert 1.01 <= odds['team1_odds'] <= 100, f"Invalid odds: {odds['team1_odds']}"
            assert 1.01 <= odds['team2_odds'] <= 100, f"Invalid odds: {odds['team2_odds']}"

            # Save to database
            db_odds = LiveOdds(
                match_id=odds['match_id'],
                bookmaker=odds['bookmaker'],
                team1_odds=odds['team1_odds'],
                team2_odds=odds['team2_odds'],
                scraped_at=datetime.utcnow()
            )
            db_session.add(db_odds)

        await db_session.commit()
        print(f"✅ Successfully scraped odds from {len(set(o['bookmaker'] for o in odds_data))} bookmakers")
        return True

    @pytest.mark.asyncio
    async def test_data_consistency(self, db_session):
        """Test data consistency and relationships"""
        print("🔍 Testing Data Consistency...")

        # Check match-team relationships
        matches = await db_session.execute(
            "SELECT COUNT(*) FROM matches WHERE team1_id IS NOT NULL"
        )
        match_count = matches.scalar()

        # Check player-team relationships
        players = await db_session.execute(
            "SELECT COUNT(*) FROM players WHERE team_id IS NOT NULL"
        )
        player_count = players.scalar()

        # Check odds-match relationships
        odds = await db_session.execute(
            "SELECT COUNT(*) FROM live_odds WHERE match_id IS NOT NULL"
        )
        odds_count = odds.scalar()

        print(f"📊 Data Summary:")
        print(f"  - Matches with teams: {match_count}")
        print(f"  - Players with teams: {player_count}")
        print(f"  - Odds records: {odds_count}")

        assert match_count > 0, "No matches with team relationships"
        assert player_count > 0, "No players with team relationships"

        print("✅ Data consistency check passed")
        return True

# Run tests
async def run_all_tests():
    """Execute all scraper tests"""
    print("\n" + "="*60)
    print("🚀 STARTING SCRAPER INTEGRATION TESTS")
    print("="*60 + "\n")

    test_suite = TestScraperIntegration()

    # Initialize fixtures
    async with get_db_session() as session:
        hltv = HLTVScraper()
        vlr = VLRScraper()
        odds = OddsScraper()

        # Run tests sequentially
        try:
            await test_suite.test_hltv_match_scraping(hltv, session)
            await asyncio.sleep(2)  # Rate limiting

            await test_suite.test_player_stats_scraping(hltv, session)
            await asyncio.sleep(2)

            await test_suite.test_live_odds_scraping(odds, session)
            await asyncio.sleep(1)

            await test_suite.test_data_consistency(session)

            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED SUCCESSFULLY!")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            raise

if __name__ == "__main__":
    asyncio.run(run_all_tests())
