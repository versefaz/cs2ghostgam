import os
import json
import time
from datetime import datetime, timezone

import redis
import psycopg
import httpx


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


# Defaults (override via env)
REDIS_HOST = env("REDIS_HOST", "localhost")
REDIS_PORT = int(env("REDIS_PORT", "6379"))
REDIS_DB = int(env("REDIS_DB", "0"))
REDIS_PASSWORD = env("REDIS_PASSWORD", "") or None

# Feature-builder DB (schema: matches(match_id, team1_id, team2_id, match_date, map_name))
PG_DSN = env(
    "PG_DSN",
    "postgresql://postgres:password@localhost:5432/cs2_betting",
)

FEATURE_BUILDER_URL = env("FEATURE_BUILDER_URL", "http://localhost:8000")


def wait_for_redis(timeout_seconds: int = 60, interval: float = 1.5) -> None:
    """Wait until Redis is reachable or timeout."""
    start = time.time()
    last_err = None
    while time.time() - start < timeout_seconds:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD, decode_responses=True)
            if r.ping():
                return
        except Exception as e:
            last_err = e
        time.sleep(interval)
    raise RuntimeError(f"Redis not ready after {timeout_seconds}s: {last_err}")


def wait_for_feature_builder(timeout_seconds: int = 60, interval: float = 1.5) -> None:
    """Wait until Feature Builder /health responds healthy or timeout."""
    start = time.time()
    last_err = None
    url = f"{FEATURE_BUILDER_URL}/health"
    while time.time() - start < timeout_seconds:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return
        except Exception as e:
            last_err = e
        time.sleep(interval)
    raise RuntimeError(f"Feature Builder not ready after {timeout_seconds}s: {last_err}")


def check_redis():
    print("== Redis Health ==")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD, decode_responses=True)
    pong = r.ping()
    info = r.info()
    print({
        "ping": pong,
        "connected_clients": info.get("connected_clients"),
        "used_memory_human": info.get("used_memory_human"),
        "uptime": info.get("uptime"),
    })

    # publish sample
    channel = "events.smoke"
    message = {
        "data": {"hello": "world"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
    }
    payload = json.dumps(message)
    subs = r.publish(channel, payload)
    r.rpush(f"queue:{channel}", payload)
    print({"publish_to": channel, "subscribers": subs, "queued": True})


def check_db_and_seed_match():
    print("\n== Postgres Health + Seed Feature-Builder Match ==")
    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            print({"select_1": cur.fetchone()[0]})

            # ensure table exists (align with feature-builder schema)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    match_id VARCHAR(50) PRIMARY KEY,
                    team1_id INTEGER,
                    team2_id INTEGER,
                    match_date TIMESTAMPTZ DEFAULT NOW(),
                    map_name VARCHAR(50)
                );
                """
            )
            conn.commit()

            # upsert demo match for calculators
            match_id = "smoke-001"
            team1_id, team2_id = 1001, 1002
            map_name = "mirage"
            match_date = datetime.now(timezone.utc)
            cur.execute(
                """
                INSERT INTO matches (match_id, team1_id, team2_id, match_date, map_name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE SET
                    team1_id = EXCLUDED.team1_id,
                    team2_id = EXCLUDED.team2_id,
                    match_date = EXCLUDED.match_date,
                    map_name = EXCLUDED.map_name;
                """,
                (match_id, team1_id, team2_id, match_date, map_name),
            )
            conn.commit()
            print({
                "upsert_match": match_id,
                "team1_id": team1_id,
                "team2_id": team2_id,
                "map": map_name,
            })

            # show counts
            cur.execute("SELECT COUNT(*) FROM matches;")
            print({"matches_count": cur.fetchone()[0]})

    return {"match_id": match_id, "team1_id": team1_id, "team2_id": team2_id, "map_name": map_name}


def call_feature_builder(match_info):
    print("\n== Call Feature Builder API ==")
    url = f"{FEATURE_BUILDER_URL}/features/match"
    payload = {
        "match_id": match_info["match_id"],
        "team1_id": match_info["team1_id"],
        "team2_id": match_info["team2_id"],
        "map_name": match_info["map_name"],
        "force_refresh": True,
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, json=payload)
        print({"status_code": resp.status_code})
        try:
            data = resp.json()
        except Exception:
            print("Response text:", resp.text[:500])
            raise
        print("features keys:", list(data.get("features", {}).keys()))
        return data


def main():
    print("E2E Smoke Test starting...")
    print("Waiting for Redis...")
    wait_for_redis()
    check_redis()
    print("Waiting for Feature Builder...")
    wait_for_feature_builder()
    match_info = check_db_and_seed_match()
    fb = call_feature_builder(match_info)
    print("\n== Summary ==")
    print("Feature Builder responded at:", fb.get("timestamp"))
    print("Feature count:", len(fb.get("features", {})))
    print("Done.")


if __name__ == "__main__":
    main()
