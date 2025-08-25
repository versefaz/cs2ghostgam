from cs2_betting_system.scrapers.live_match_scraper import LiveMatchScraper

def test_scraper_runs():
    s = LiveMatchScraper()
    matches = s.scrape_all_sources()
    assert matches is not None
