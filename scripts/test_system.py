import asyncio
import aiohttp
import json
import psycopg2
import redis

async def test_prediction_endpoint():
    test_match = {
        "match_id": "test_001",
        "team1": "NaVi",
        "team2": "FaZe",
        "team1_rating": 1800,
        "team2_rating": 1750,
        "team1_winrate": 0.65,
        "team2_winrate": 0.58,
        "odds_team1": 1.85,
        "odds_team2": 2.10,
        "best_of": 3
    }
    # Placeholder: no API in this module; ensure services are up
    print(json.dumps(test_match))

async def test_database_connection():
    try:
        conn = psycopg2.connect(host='localhost', database='cs2_predictions', user='postgres', password='your_password')
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM predictions')
        print('PostgreSQL OK, predictions count:', cur.fetchone()[0])
        conn.close()
    except Exception as e:
        print('PostgreSQL Error:', e)
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print('Redis OK, keys:', len(r.keys('prediction:*')))
    except Exception as e:
        print('Redis Error:', e)

if __name__ == '__main__':
    print('Testing system...')
    asyncio.run(test_database_connection())
    asyncio.run(test_prediction_endpoint())
    print('Done')
