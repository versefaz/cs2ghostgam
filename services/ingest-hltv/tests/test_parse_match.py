import pytest


def sample_match_json():
    return {
        "id": 123,
        "team1": {"name": "Natus Vincere"},
        "team2": {"name": "G2"},
    }


def test_parse_match():
    data = sample_match_json()
    assert data["team1"]["name"] == "Natus Vincere"
