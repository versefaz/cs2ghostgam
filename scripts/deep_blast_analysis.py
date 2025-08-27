#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep BLAST Open London 2025 Analysis
การวิเคราะห์เจาะลึกพร้อมกลยุทธ์การเดิมพันที่ทำกำไรได้มากที่สุด
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class DeepBlastAnalyzer:
    """การวิเคราะห์เจาะลึก BLAST Open London 2025 เพื่อกำไรสูงสุด"""
    
    def __init__(self):
        self.matches = [
            {
                "id": "vitality_m80",
                "team1": "Vitality", "team2": "M80",
                "time": "LIVE", "status": "live",
                "odds": {"team1": 1.16, "team2": 4.90},
                "rankings": {"team1": 6, "team2": 25},
                "recent_form": {"team1": "WWLWW", "team2": "LWLWL"},
                "map_pool_advantage": "Vitality",
                "key_players": {
                    "team1": ["ZywOo", "apEX", "Magisk", "flameZ", "Spinx"],
                    "team2": ["slaxz-", "malbsMd", "Swisher", "reck", "Lake"]
                },
                "map_pool": {
                    "team1": {
                        "strong": ["Mirage", "Inferno", "Ancient"],
                        "weak": ["Vertigo", "Overpass"],
                        "win_rates": {"Mirage": 78, "Inferno": 82, "Ancient": 71, "Dust2": 65, "Anubis": 58, "Vertigo": 42, "Overpass": 38}
                    },
                    "team2": {
                        "strong": ["Dust2", "Anubis", "Overpass"],
                        "weak": ["Mirage", "Inferno"],
                        "win_rates": {"Dust2": 72, "Anubis": 68, "Overpass": 64, "Vertigo": 55, "Ancient": 48, "Mirage": 35, "Inferno": 32}
                    }
                },
                "player_stats": {
                    "team1": {
                        "carry_potential": {"ZywOo": 9.5, "flameZ": 7.2, "Magisk": 6.8, "Spinx": 6.1, "apEX": 5.4},
                        "clutch_rates": {"ZywOo": 42, "Magisk": 38, "flameZ": 35, "Spinx": 32, "apEX": 28},
                        "pistol_wr": {"1st": 68, "13th": 72}
                    },
                    "team2": {
                        "carry_potential": {"slaxz-": 7.8, "malbsMd": 6.9, "Swisher": 6.2, "reck": 5.8, "Lake": 5.5},
                        "clutch_rates": {"slaxz-": 35, "malbsMd": 32, "Swisher": 29, "reck": 27, "Lake": 25},
                        "pistol_wr": {"1st": 58, "13th": 62}
                    }
                },
                "side_stats": {
                    "team1": {"ct_win_rate": 58, "t_win_rate": 52},
                    "team2": {"ct_win_rate": 52, "t_win_rate": 48}
                }
            },
            {
                "id": "gamerlegion_virtuspro",
                "team1": "GamerLegion", "team2": "Virtus.pro",
                "time": "19:30", "status": "upcoming",
                "odds": {"team1": 2.07, "team2": 1.78},
                "rankings": {"team1": 16, "team2": 15},
                "recent_form": {"team1": "WLWWL", "team2": "LWWLW"},
                "map_pool_advantage": "Even",
                "key_players": {
                    "team1": ["REZ", "ztr", "Tauson", "Kursy", "PR"],
                    "team2": ["electroNic", "FL1T", "Perfecto", "fame", "ICY"]
                },
                "map_pool": {
                    "team1": {
                        "strong": ["Dust2", "Mirage", "Anubis"],
                        "weak": ["Ancient", "Vertigo"],
                        "win_rates": {"Dust2": 74, "Mirage": 69, "Anubis": 66, "Inferno": 58, "Overpass": 55, "Ancient": 45, "Vertigo": 41}
                    },
                    "team2": {
                        "strong": ["Inferno", "Ancient", "Overpass"],
                        "weak": ["Dust2", "Anubis"],
                        "win_rates": {"Inferno": 76, "Ancient": 71, "Overpass": 68, "Vertigo": 62, "Mirage": 56, "Dust2": 43, "Anubis": 39}
                    }
                },
                "player_stats": {
                    "team1": {
                        "carry_potential": {"REZ": 8.4, "ztr": 7.1, "Tauson": 6.7, "Kursy": 6.2, "PR": 5.9},
                        "clutch_rates": {"REZ": 39, "ztr": 34, "Tauson": 31, "Kursy": 29, "PR": 26},
                        "pistol_wr": {"1st": 62, "13th": 65}
                    },
                    "team2": {
                        "carry_potential": {"electroNic": 8.8, "FL1T": 7.3, "Perfecto": 6.5, "fame": 6.1, "ICY": 5.7},
                        "clutch_rates": {"electroNic": 41, "FL1T": 36, "Perfecto": 33, "fame": 30, "ICY": 27},
                        "pistol_wr": {"1st": 64, "13th": 68}
                    }
                },
                "side_stats": {
                    "team1": {"ct_win_rate": 55, "t_win_rate": 49},
                    "team2": {"ct_win_rate": 57, "t_win_rate": 51}
                }
            },
            {
                "id": "faze_ecstatic",
                "team1": "FaZe", "team2": "ECSTATIC",
                "time": "22:00", "status": "upcoming",
                "odds": {"team1": 1.33, "team2": 3.29},
                "rankings": {"team1": 9, "team2": 36},
                "recent_form": {"team1": "WLWLW", "team2": "WWLWW"},
                "map_pool_advantage": "FaZe",
                "key_players": {
                    "team1": ["karrigan", "rain", "broky", "frozen", "ropz"],
                    "team2": ["jcobbb", "Anlelele", "Nodios", "Peppzor", "Dytor"]
                },
                "map_pool": {
                    "team1": {
                        "strong": ["Mirage", "Inferno", "Dust2"],
                        "weak": ["Anubis", "Vertigo"],
                        "win_rates": {"Mirage": 79, "Inferno": 75, "Dust2": 72, "Ancient": 64, "Overpass": 61, "Anubis": 48, "Vertigo": 44}
                    },
                    "team2": {
                        "strong": ["Anubis", "Vertigo", "Ancient"],
                        "weak": ["Mirage", "Inferno"],
                        "win_rates": {"Anubis": 71, "Vertigo": 67, "Ancient": 63, "Overpass": 58, "Dust2": 52, "Mirage": 41, "Inferno": 38}
                    }
                },
                "player_stats": {
                    "team1": {
                        "carry_potential": {"broky": 8.6, "ropz": 8.2, "frozen": 7.8, "rain": 7.1, "karrigan": 5.8},
                        "clutch_rates": {"ropz": 43, "broky": 40, "frozen": 37, "rain": 34, "karrigan": 31},
                        "pistol_wr": {"1st": 66, "13th": 69}
                    },
                    "team2": {
                        "carry_potential": {"jcobbb": 7.9, "Anlelele": 7.2, "Nodios": 6.8, "Peppzor": 6.4, "Dytor": 6.0},
                        "clutch_rates": {"jcobbb": 36, "Anlelele": 33, "Nodios": 30, "Peppzor": 28, "Dytor": 26},
                        "pistol_wr": {"1st": 61, "13th": 64}
                    }
                },
                "side_stats": {
                    "team1": {"ct_win_rate": 59, "t_win_rate": 53},
                    "team2": {"ct_win_rate": 54, "t_win_rate": 50}
                }
            },
            {
                "id": "navi_fnatic",
                "team1": "Natus Vincere", "team2": "fnatic",
                "time": "00:30", "status": "upcoming",
                "odds": {"team1": 1.27, "team2": 3.69},
                "rankings": {"team1": 6, "team2": 34},
                "recent_form": {"team1": "WWLWW", "team2": "LWWLW"},
                "map_pool_advantage": "NAVI",
                "key_players": {
                    "team1": ["s1mple", "electroNic", "Perfecto", "b1t", "sdy"],
                    "team2": ["KRIMZ", "nicoodoz", "roeJ", "afro", "matys"]
                },
                "map_pool": {
                    "team1": {
                        "strong": ["Mirage", "Dust2", "Inferno"],
                        "weak": ["Vertigo", "Anubis"],
                        "win_rates": {"Mirage": 81, "Dust2": 77, "Inferno": 74, "Ancient": 66, "Overpass": 63, "Vertigo": 47, "Anubis": 44}
                    },
                    "team2": {
                        "strong": ["Ancient", "Overpass", "Vertigo"],
                        "weak": ["Mirage", "Dust2"],
                        "win_rates": {"Ancient": 69, "Overpass": 65, "Vertigo": 62, "Anubis": 57, "Inferno": 53, "Dust2": 42, "Mirage": 39}
                    }
                },
                "player_stats": {
                    "team1": {
                        "carry_potential": {"s1mple": 9.8, "electroNic": 8.1, "b1t": 7.4, "Perfecto": 6.6, "sdy": 6.2},
                        "clutch_rates": {"s1mple": 45, "electroNic": 39, "b1t": 36, "Perfecto": 32, "sdy": 29},
                        "pistol_wr": {"1st": 71, "13th": 74}
                    },
                    "team2": {
                        "carry_potential": {"nicoodoz": 7.6, "KRIMZ": 7.0, "roeJ": 6.5, "afro": 6.1, "matys": 5.8},
                        "clutch_rates": {"KRIMZ": 37, "nicoodoz": 34, "roeJ": 31, "afro": 28, "matys": 25},
                        "pistol_wr": {"1st": 59, "13th": 62}
                    }
                },
                "side_stats": {
                    "team1": {"ct_win_rate": 61, "t_win_rate": 55},
                    "team2": {"ct_win_rate": 53, "t_win_rate": 47}
                }
            }
        ]
        
        # Kelly Criterion parameters
        self.bankroll_percentage = {
            "conservative": 0.02,  # 2%
            "moderate": 0.05,      # 5%
            "aggressive": 0.10     # 10%
        }
    
    def calculate_implied_probability(self, odds: float) -> float:
        """คำนวณความน่าจะเป็นจาก odds"""
        return 1 / odds
    
    def calculate_true_probability(self, match: Dict) -> Dict[str, float]:
        """คำนวณความน่าจะเป็นจริงจากการวิเคราะห์"""
        
        # Base probability from rankings
        rank_diff = abs(match["rankings"]["team1"] - match["rankings"]["team2"])
        
        if match["rankings"]["team1"] < match["rankings"]["team2"]:
            # Team1 ranked higher
            base_prob_team1 = 0.65 + (rank_diff * 0.02)
        else:
            # Team2 ranked higher  
            base_prob_team1 = 0.35 - (rank_diff * 0.02)
        
        # Adjust for recent form
        team1_form_score = self._calculate_form_score(match["recent_form"]["team1"])
        team2_form_score = self._calculate_form_score(match["recent_form"]["team2"])
        
        form_adjustment = (team1_form_score - team2_form_score) * 0.05
        base_prob_team1 += form_adjustment
        
        # Map pool advantage
        if match["map_pool_advantage"] == match["team1"]:
            base_prob_team1 += 0.05
        elif match["map_pool_advantage"] == match["team2"]:
            base_prob_team1 -= 0.05
        
        # Clamp between 0.1 and 0.9
        base_prob_team1 = max(0.1, min(0.9, base_prob_team1))
        
        return {
            "team1": base_prob_team1,
            "team2": 1 - base_prob_team1
        }
    
    def _calculate_form_score(self, form: str) -> float:
        """คำนวณคะแนนจากฟอร์มล่าสุด (W=1, L=0)"""
        score = 0
        weights = [0.4, 0.3, 0.2, 0.08, 0.02]  # น้ำหนักแมตช์ล่าสุดมากกว่า
        
        for i, result in enumerate(form):
            if i < len(weights):
                score += weights[i] * (1 if result == 'W' else 0)
        
        return score
    
    def calculate_kelly_criterion(self, odds: float, true_prob: float) -> float:
        """คำนวณ Kelly Criterion สำหรับขนาดเดิมพัน"""
        if true_prob <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # b = odds - 1, p = true probability, q = 1 - p
        b = odds - 1
        p = true_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Return positive Kelly only (no negative bets)
        return max(0, kelly_fraction)
    
    def find_value_bets(self, match: Dict) -> List[Dict]:
        """หา value bets ที่มีกำไรสูงสุด"""
        
        true_probs = self.calculate_true_probability(match)
        implied_probs = {
            "team1": self.calculate_implied_probability(match["odds"]["team1"]),
            "team2": self.calculate_implied_probability(match["odds"]["team2"])
        }
        
        value_bets = []
        
        # Check Team 1
        if true_probs["team1"] > implied_probs["team1"]:
            edge = true_probs["team1"] - implied_probs["team1"]
            kelly = self.calculate_kelly_criterion(match["odds"]["team1"], true_probs["team1"])
            
            value_bets.append({
                "team": match["team1"],
                "bet_type": "Moneyline",
                "odds": match["odds"]["team1"],
                "true_prob": true_probs["team1"],
                "implied_prob": implied_probs["team1"],
                "edge": edge,
                "kelly_fraction": kelly,
                "expected_value": (true_probs["team1"] * (match["odds"]["team1"] - 1)) - (1 - true_probs["team1"]),
                "confidence": "High" if edge > 0.1 else "Medium" if edge > 0.05 else "Low"
            })
        
        # Check Team 2
        if true_probs["team2"] > implied_probs["team2"]:
            edge = true_probs["team2"] - implied_probs["team2"]
            kelly = self.calculate_kelly_criterion(match["odds"]["team2"], true_probs["team2"])
            
            value_bets.append({
                "team": match["team2"],
                "bet_type": "Moneyline",
                "odds": match["odds"]["team2"],
                "true_prob": true_probs["team2"],
                "implied_prob": implied_probs["team2"],
                "edge": edge,
                "kelly_fraction": kelly,
                "expected_value": (true_probs["team2"] * (match["odds"]["team2"] - 1)) - (1 - true_probs["team2"]),
                "confidence": "High" if edge > 0.1 else "Medium" if edge > 0.05 else "Low"
            })
        
        return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)
    
    def generate_alternative_bets(self, match: Dict) -> List[Dict]:
        """สร้างเดิมพันทางเลือกที่มีกำไรสูง"""
        
        alternatives = []
        
        # Over/Under Maps (สมมติ line ที่ 2.5)
        if match["status"] != "live":
            # คำนวณโอกาสไป 3 แมป
            close_match = abs(match["rankings"]["team1"] - match["rankings"]["team2"]) <= 5
            
            if close_match:
                over_25_prob = 0.65
                alternatives.append({
                    "bet_type": "Over 2.5 Maps",
                    "estimated_odds": 1.85,
                    "true_prob": over_25_prob,
                    "reasoning": "คู่ใกล้เคียง มีโอกาสไป 3 แมปสูง",
                    "expected_value": (over_25_prob * 0.85) - (1 - over_25_prob),
                    "risk_level": "Medium"
                })
        
        # Handicap bets
        favorite = match["team1"] if match["odds"]["team1"] < match["odds"]["team2"] else match["team2"]
        underdog = match["team2"] if match["odds"]["team1"] < match["odds"]["team2"] else match["team1"]
        
        # +1.5 Maps for underdog
        handicap_prob = 0.45 if abs(match["rankings"]["team1"] - match["rankings"]["team2"]) > 10 else 0.55
        alternatives.append({
            "bet_type": f"{underdog} +1.5 Maps",
            "estimated_odds": 1.65,
            "true_prob": handicap_prob,
            "reasoning": f"{underdog} มีโอกาสได้อย่างน้อย 1 แมป",
            "expected_value": (handicap_prob * 0.65) - (1 - handicap_prob),
            "risk_level": "Low"
        })
        
        return alternatives
    
    def analyze_map_pool_deep(self, match: Dict) -> Dict:
        """วิเคราะห์แมพพูลเจาะลึก"""
        team1_maps = match["map_pool"]["team1"]
        team2_maps = match["map_pool"]["team2"]
        
        # Find map advantages
        map_analysis = {}
        for map_name in team1_maps["win_rates"]:
            if map_name in team2_maps["win_rates"]:
                team1_wr = team1_maps["win_rates"][map_name]
                team2_wr = team2_maps["win_rates"][map_name]
                edge = team1_wr - team2_wr
                
                map_analysis[map_name] = {
                    "team1_wr": team1_wr,
                    "team2_wr": team2_wr,
                    "edge": edge,
                    "favorite": match["team1"] if edge > 0 else match["team2"],
                    "advantage_level": "High" if abs(edge) > 20 else "Medium" if abs(edge) > 10 else "Low"
                }
        
        return map_analysis
    
    def analyze_carry_potential(self, match: Dict) -> Dict:
        """วิเคราะห์ศักยภาพการแบกทีม"""
        team1_carries = match["player_stats"]["team1"]["carry_potential"]
        team2_carries = match["player_stats"]["team2"]["carry_potential"]
        
        team1_top = max(team1_carries, key=team1_carries.get)
        team2_top = max(team2_carries, key=team2_carries.get)
        
        return {
            "team1_top_carry": {"player": team1_top, "rating": team1_carries[team1_top]},
            "team2_top_carry": {"player": team2_top, "rating": team2_carries[team2_top]},
            "carry_advantage": match["team1"] if team1_carries[team1_top] > team2_carries[team2_top] else match["team2"]
        }
    
    def analyze_clutch_rates(self, match: Dict) -> Dict:
        """วิเคราะห์อัตราการคลัตช์"""
        team1_clutch = match["player_stats"]["team1"]["clutch_rates"]
        team2_clutch = match["player_stats"]["team2"]["clutch_rates"]
        
        team1_avg = sum(team1_clutch.values()) / len(team1_clutch)
        team2_avg = sum(team2_clutch.values()) / len(team2_clutch)
        
        team1_best = max(team1_clutch, key=team1_clutch.get)
        team2_best = max(team2_clutch, key=team2_clutch.get)
        
        return {
            "team1_avg": team1_avg,
            "team2_avg": team2_avg,
            "advantage": match["team1"] if team1_avg > team2_avg else match["team2"],
            "best_clutchers": {
                "team1": {"player": team1_best, "rate": team1_clutch[team1_best]},
                "team2": {"player": team2_best, "rate": team2_clutch[team2_best]}
            }
        }
    
    def analyze_pistol_rounds(self, match: Dict) -> Dict:
        """วิเคราะห์รอบพิสตอล"""
        team1_pistol = match["player_stats"]["team1"]["pistol_wr"]
        team2_pistol = match["player_stats"]["team2"]["pistol_wr"]
        
        return {
            "first_pistol": {
                "team1_wr": team1_pistol["1st"],
                "team2_wr": team2_pistol["1st"],
                "advantage": match["team1"] if team1_pistol["1st"] > team2_pistol["1st"] else match["team2"],
                "edge": abs(team1_pistol["1st"] - team2_pistol["1st"])
            },
            "second_pistol": {
                "team1_wr": team1_pistol["13th"],
                "team2_wr": team2_pistol["13th"],
                "advantage": match["team1"] if team1_pistol["13th"] > team2_pistol["13th"] else match["team2"],
                "edge": abs(team1_pistol["13th"] - team2_pistol["13th"])
            }
        }
    
    def analyze_side_performance(self, match: Dict) -> Dict:
        """วิเคราะห์การเล่นแต่ละข้าง"""
        team1_sides = match["side_stats"]["team1"]
        team2_sides = match["side_stats"]["team2"]
        
        return {
            "ct_advantage": match["team1"] if team1_sides["ct_win_rate"] > team2_sides["ct_win_rate"] else match["team2"],
            "t_advantage": match["team1"] if team1_sides["t_win_rate"] > team2_sides["t_win_rate"] else match["team2"],
            "ct_edge": abs(team1_sides["ct_win_rate"] - team2_sides["ct_win_rate"]),
            "t_edge": abs(team1_sides["t_win_rate"] - team2_sides["t_win_rate"])
        }
    
    def generate_first_map_predictions(self, match: Dict, map_analysis: Dict) -> List[Dict]:
        """ทำนายการเดิมพันแมพแรก"""
        predictions = []
        
        # Top 3 maps with biggest advantages
        sorted_maps = sorted(map_analysis.items(), key=lambda x: abs(x[1]["edge"]), reverse=True)[:3]
        
        for map_name, data in sorted_maps:
            if abs(data["edge"]) > 15:  # Only significant advantages
                predictions.append({
                    "map": map_name,
                    "recommended_bet": f"{data['favorite']} Map 1 (หาก {map_name} ถูกเล่น)",
                    "win_probability": max(data["team1_wr"], data["team2_wr"]) / 100,
                    "edge": abs(data["edge"]),
                    "confidence": data["advantage_level"]
                })
        
        return predictions
    
    def generate_multi_profit_opportunities(self, match: Dict, analyses: Dict) -> List[Dict]:
        """สร้างโอกาสทำกำไรหลายมุมมอง"""
        opportunities = []
        
        # Map-based opportunities
        for pred in analyses.get("first_map_predictions", [])[:2]:
            opportunities.append({
                "category": "🗺️ Map Betting",
                "bet": pred["recommended_bet"],
                "probability": f"{pred['win_probability']:.1%}",
                "edge": f"{pred['edge']:.0f}%",
                "risk": "Medium"
            })
        
        # Player performance
        carry_analysis = analyses.get("carry_potential", {})
        if carry_analysis:
            top_carry = carry_analysis.get("team1_top_carry", {}) if carry_analysis.get("carry_advantage") == match["team1"] else carry_analysis.get("team2_top_carry", {})
            if top_carry.get("rating", 0) > 8.0:
                opportunities.append({
                    "category": "👤 Player Performance",
                    "bet": f"{top_carry['player']} MVP (หากมี)",
                    "probability": f"{min(95, top_carry['rating'] * 10):.0f}%",
                    "edge": "High carry potential",
                    "risk": "Medium"
                })
        
        # Pistol rounds
        pistol_analysis = analyses.get("pistol_rounds", {})
        if pistol_analysis:
            first_pistol = pistol_analysis.get("first_pistol", {})
            if first_pistol.get("edge", 0) > 8:
                opportunities.append({
                    "category": "🔫 Pistol Rounds",
                    "bet": f"{first_pistol['advantage']} ชนะรอบพิสตอลแรก",
                    "probability": f"{max(first_pistol['team1_wr'], first_pistol['team2_wr']):.0f}%",
                    "edge": f"{first_pistol['edge']:.0f}%",
                    "risk": "High"
                })
        
        # Side-specific bets
        side_analysis = analyses.get("side_performance", {})
        if side_analysis and side_analysis.get("ct_edge", 0) > 6:
            opportunities.append({
                "category": "⚔️ Side Performance",
                "bet": f"{side_analysis['ct_advantage']} ได้เปรียบฝั่ง CT",
                "probability": "Situational",
                "edge": f"{side_analysis['ct_edge']:.0f}%",
                "risk": "Low"
            })
        
        return opportunities
    
    def analyze_match_deep(self, match: Dict) -> Dict:
        """วิเคราะห์แมตช์เจาะลึกแบบสมบูรณ์"""
        
        print(f"\n{'='*80}")
        print(f"🔍 การวิเคราะห์เจาะลึก: {match['team1']} vs {match['team2']}")
        print(f"{'='*80}")
        
        # Basic info
        print(f"⏰ เวลา: {match['time']} | สถานะ: {match['status']}")
        print(f"🏆 อันดับ: #{match['rankings']['team1']} vs #{match['rankings']['team2']}")
        print(f"📊 ฟอร์มล่าสุด: {match['recent_form']['team1']} vs {match['recent_form']['team2']}")
        
        # Advanced analyses
        map_analysis = self.analyze_map_pool_deep(match)
        carry_analysis = self.analyze_carry_potential(match)
        clutch_analysis = self.analyze_clutch_rates(match)
        pistol_analysis = self.analyze_pistol_rounds(match)
        side_analysis = self.analyze_side_performance(match)
        first_map_predictions = self.generate_first_map_predictions(match, map_analysis)
        
        # Map Pool Analysis
        print(f"\n🗺️ การวิเคราะห์แมพพูล:")
        strong_maps_team1 = match["map_pool"]["team1"]["strong"]
        strong_maps_team2 = match["map_pool"]["team2"]["strong"]
        print(f"   {match['team1']} เก่ง: {', '.join(strong_maps_team1)}")
        print(f"   {match['team2']} เก่ง: {', '.join(strong_maps_team2)}")
        
        print(f"\n   📊 Win Rate เฉพาะแมพ:")
        for map_name, data in sorted(map_analysis.items(), key=lambda x: abs(x[1]["edge"]), reverse=True)[:3]:
            print(f"      {map_name}: {match['team1']} {data['team1_wr']}% vs {match['team2']} {data['team2_wr']}% (Edge: {data['edge']:+.0f}%)")
        
        # First Map Predictions
        if first_map_predictions:
            print(f"\n   🎯 การเดิมพันแมพแรก:")
            for pred in first_map_predictions:
                print(f"      • {pred['recommended_bet']} (Edge: {pred['edge']:.0f}%)")
        
        # Player Analysis
        print(f"\n👥 การวิเคราะห์ผู้เล่น:")
        print(f"   🎯 ผู้เล่นแบกทีม:")
        print(f"      {match['team1']}: {carry_analysis['team1_top_carry']['player']} ({carry_analysis['team1_top_carry']['rating']:.1f})")
        print(f"      {match['team2']}: {carry_analysis['team2_top_carry']['player']} ({carry_analysis['team2_top_carry']['rating']:.1f})")
        print(f"      ได้เปรียบ: {carry_analysis['carry_advantage']}")
        
        print(f"\n   🔥 อัตราการคลัตช์:")
        print(f"      {match['team1']}: {clutch_analysis['team1_avg']:.1f}% (ดีสุด: {clutch_analysis['best_clutchers']['team1']['player']} {clutch_analysis['best_clutchers']['team1']['rate']}%)")
        print(f"      {match['team2']}: {clutch_analysis['team2_avg']:.1f}% (ดีสุด: {clutch_analysis['best_clutchers']['team2']['player']} {clutch_analysis['best_clutchers']['team2']['rate']}%)")
        print(f"      ได้เปรียบ: {clutch_analysis['advantage']}")
        
        # Pistol Round Analysis
        print(f"\n🔫 การวิเคราะห์รอบพิสตอล:")
        print(f"   รอบที่ 1: {match['team1']} {pistol_analysis['first_pistol']['team1_wr']}% vs {match['team2']} {pistol_analysis['first_pistol']['team2_wr']}%")
        print(f"   รอบที่ 13: {match['team1']} {pistol_analysis['second_pistol']['team1_wr']}% vs {match['team2']} {pistol_analysis['second_pistol']['team2_wr']}%")
        if pistol_analysis['first_pistol']['edge'] > 8:
            print(f"   💡 แนะนำ: เดิมพัน {pistol_analysis['first_pistol']['advantage']} ชนะรอบพิสตอลแรก")
        
        # Side Performance
        print(f"\n⚔️ การวิเคราะห์ข้าง CT/T:")
        print(f"   CT Side: {side_analysis['ct_advantage']} ได้เปรียบ ({side_analysis['ct_edge']:.0f}%)")
        print(f"   T Side: {side_analysis['t_advantage']} ได้เปรียบ ({side_analysis['t_edge']:.0f}%)")
        
        # Original analysis
        true_probs = self.calculate_true_probability(match)
        print(f"\n🎯 ความน่าจะเป็นจริง:")
        print(f"   {match['team1']}: {true_probs['team1']:.1%}")
        print(f"   {match['team2']}: {true_probs['team2']:.1%}")
        
        value_bets = self.find_value_bets(match)
        print(f"\n💎 Value Bets:")
        if value_bets:
            for bet in value_bets:
                print(f"   🎯 {bet['team']} ML @ {bet['odds']} (Kelly: {bet['kelly_fraction']:.1%})")
        else:
            print("   ❌ ไม่พบ value bet ในราคา moneyline")
        
        # Multi-angle opportunities
        analyses_dict = {
            "map_analysis": map_analysis,
            "carry_potential": carry_analysis,
            "clutch_rates": clutch_analysis,
            "pistol_rounds": pistol_analysis,
            "side_performance": side_analysis,
            "first_map_predictions": first_map_predictions
        }
        
        opportunities = self.generate_multi_profit_opportunities(match, analyses_dict)
        print(f"\n💰 โอกาสทำกำไรหลายมุมมอง:")
        for opp in opportunities:
            print(f"   {opp['category']}: {opp['bet']}")
            print(f"      ความน่าจะเป็น: {opp['probability']} | Edge: {opp['edge']} | ความเสี่ยง: {opp['risk']}")
        
        # Final recommendation
        alternatives = self.generate_alternative_bets(match)
        recommendation = self.get_final_recommendation(match, value_bets, alternatives)
        print(f"\n🎯 คำแนะนำสุดท้าย:")
        print(f"   🥇 เดิมพันหลัก: {recommendation['primary']}")
        print(f"   🥈 เดิมพันสำรอง: {recommendation['secondary']}")
        print(f"   💰 กลยุทธ์: {recommendation['strategy']}")
        print(f"   ⚠️ ความเสี่ยง: {recommendation['risk_assessment']}")
        
        return {
            "match_id": match["id"],
            "map_analysis": map_analysis,
            "carry_analysis": carry_analysis,
            "clutch_analysis": clutch_analysis,
            "pistol_analysis": pistol_analysis,
            "side_analysis": side_analysis,
            "first_map_predictions": first_map_predictions,
            "profit_opportunities": opportunities,
            "true_probabilities": true_probs,
            "value_bets": value_bets,
            "recommendation": recommendation
        }
    
    def get_match_insights(self, match: Dict) -> List[str]:
        """ข้อมูลเชิงลึกเฉพาะแมตช์"""
        
        insights = []
        
        if match["id"] == "vitality_m80":
            insights.extend([
                "ZywOo เป็นปัจจัยสำคัญ - ถ้าเขาเล่นได้ดี Vitality ชนะง่าย",
                "M80 เป็นทีมอเมริกันที่มี upset potential สูง",
                "แมตช์ LIVE - สามารถดู momentum ก่อนเดิมพัน live betting",
                "Vitality มีประสบการณ์ระดับ Tier 1 มากกว่า"
            ])
        
        elif match["id"] == "gamerlegion_virtuspro":
            insights.extend([
                "คู่ที่ใกล้เคียงที่สุด - อันดับต่างกันแค่ 1 อันดับ",
                "REZ vs electroNic - การดวลของ star players",
                "Virtus.pro มีประสบการณ์ในทัวร์นาเมนต์ใหญ่มากกว่า",
                "GamerLegion มีการเตรียมตัวที่ดีและมีแผนการเล่นชัดเจน"
            ])
        
        elif match["id"] == "faze_ecstatic":
            insights.extend([
                "FaZe มี firepower สูงแต่บางครั้งไม่สม่ำเสมอ",
                "ECSTATIC มีฟอร์มดีล่าสุด (4 ชนะใน 5 แมตช์)",
                "jcobbb เป็นผู้เล่นใหม่ที่อาจสร้างความแปลกใจ",
                "แข่งเวลา 22:00 - เวลาที่ดีสำหรับทีมยุโรป"
            ])
        
        elif match["id"] == "navi_fnatic":
            insights.extend([
                "s1mple กลับมาแล้ว - เพิ่มพลังให้ NAVI มาก",
                "fnatic กำลังสร้างทีมใหม่ - อาจยังไม่เข้าขั้น",
                "แข่งเวลาดึก (00:30) - อาจมีผลต่อการเล่น",
                "NAVI มีประสบการณ์ Major มากกว่า"
            ])
        
        return insights
    
    def get_final_recommendation(self, match: Dict, value_bets: List[Dict], alternatives: List[Dict]) -> Dict:
        """คำแนะนำสุดท้ายสำหรับการเดิมพัน"""
        
        if value_bets:
            best_value = value_bets[0]
            primary = f"{best_value['team']} ML @ {best_value['odds']} (Kelly: {best_value['kelly_fraction']:.1%})"
        else:
            # ไม่มี value bet - แนะนำ alternative
            if match["odds"]["team1"] < match["odds"]["team2"]:
                primary = f"รอราคาดีขึ้น หรือ {match['team2']} +1.5 Maps"
            else:
                primary = f"รอราคาดีขึ้น หรือ {match['team1']} +1.5 Maps"
        
        # Secondary bet
        if alternatives:
            best_alt = max(alternatives, key=lambda x: x["expected_value"])
            secondary = f"{best_alt['bet_type']} @ ~{best_alt['estimated_odds']}"
        else:
            secondary = "Over 2.5 Maps (ถ้ามี)"
        
        # Strategy based on match
        if match["status"] == "live":
            strategy = "รอดู momentum ในแมตช์ก่อนเดิมพัน live betting"
        elif len(value_bets) > 0 and value_bets[0]["confidence"] == "High":
            strategy = "เดิมพันตาม Kelly Criterion เต็มจำนวน"
        else:
            strategy = "เดิมพันระมัดระวัง หรือข้ามแมตช์นี้"
        
        # Risk assessment
        if match["id"] == "vitality_m80":
            risk = "🟢 ต่ำ-ปานกลาง (Vitality เต็งแรง แต่ M80 มี upset potential)"
        elif match["id"] == "gamerlegion_virtuspro":
            risk = "🟡 ปานกลาง-สูง (คู่ใกล้เคียง คาดเดายาก)"
        elif match["id"] == "faze_ecstatic":
            risk = "🟡 ปานกลาง (FaZe ไม่สม่ำเสมอ, ECSTATIC ฟอร์มดี)"
        else:  # navi_fnatic
            risk = "🟢 ต่ำ-ปานกลาง (NAVI เต็งแต่เล่นดึก)"
        
        return {
            "primary": primary,
            "secondary": secondary,
            "strategy": strategy,
            "risk_assessment": risk
        }
    
    async def run_deep_analysis(self):
        """รันการวิเคราะห์เจาะลึกทั้งหมด"""
        
        print("🎯 BLAST Open London 2025 - การวิเคราะห์เจาะลึกเพื่อกำไรสูงสุด")
        print(f"⏰ เวลาวิเคราะห์: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        all_analyses = []
        
        for match in self.matches:
            analysis = self.analyze_match_deep(match)
            all_analyses.append(analysis)
        
        # Portfolio summary
        self.generate_portfolio_summary(all_analyses)
        
        # Save results
        await self.save_deep_analysis(all_analyses)
        
        return all_analyses
    
    def generate_portfolio_summary(self, analyses: List[Dict]):
        """สรุปพอร์ตการเดิมพันทั้งหมด"""
        
        print(f"\n{'='*80}")
        print("📊 สรุปพอร์ตการเดิมพัน BLAST Open London 2025")
        print(f"{'='*80}")
        
        total_value_bets = sum(len(analysis["value_bets"]) for analysis in analyses)
        
        print(f"🎯 Value Bets ที่พบทั้งหมด: {total_value_bets} เดิมพัน")
        
        if total_value_bets > 0:
            print(f"\n💎 Top Value Bets:")
            
            all_value_bets = []
            for analysis in analyses:
                for bet in analysis["value_bets"]:
                    bet["match_id"] = analysis["match_id"]
                    all_value_bets.append(bet)
            
            # Sort by expected value
            top_bets = sorted(all_value_bets, key=lambda x: x["expected_value"], reverse=True)[:3]
            
            for i, bet in enumerate(top_bets, 1):
                print(f"   {i}. {bet['team']} @ {bet['odds']} (EV: {bet['expected_value']:.3f})")
                print(f"      Kelly: {bet['kelly_fraction']:.1%} | Edge: {bet['edge']:.1%}")
        
        print(f"\n🎲 กลยุทธ์การจัดการเงิน:")
        print(f"   💰 แบ่ง bankroll เป็น 4 ส่วนสำหรับแต่ละแมตช์")
        print(f"   📊 ใช้ Kelly Criterion สำหรับขนาดเดิมพัน")
        print(f"   🛡️ ไม่เดิมพันเกิน 10% ของ bankroll ในแมตช์เดียว")
        print(f"   ⏰ รอดูแมตช์ LIVE ก่อนตัดสินใจ")
        
        print(f"\n⚠️ คำเตือนสำคัญ:")
        print(f"   • การเดิมพันมีความเสี่ยง อาจสูญเสียเงินได้")
        print(f"   • ข้อมูลนี้เป็นการวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำทางการเงิน")
        print(f"   • ควรเดิมพันด้วยเงินที่สามารถเสียได้เท่านั้น")
        print(f"   • หยุดเดิมพันหากขาดทุนเกิน 20% ของ bankroll")
    
    async def save_deep_analysis(self, analyses: List[Dict]):
        """บันทึกผลการวิเคราะห์เจาะลึก"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/deep_blast_analysis_{timestamp}.json"
        
        try:
            os.makedirs("data", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "tournament": "BLAST Open London 2025",
                    "analysis_type": "deep_profitable_analysis",
                    "total_matches": len(analyses),
                    "total_value_bets": sum(len(a["value_bets"]) for a in analyses),
                    "analyses": analyses
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 บันทึกผลการวิเคราะห์ที่: {filename}")
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถบันทึกไฟล์: {e}")

async def main():
    """ฟังก์ชันหลัก"""
    try:
        analyzer = DeepBlastAnalyzer()
        await analyzer.run_deep_analysis()
        
    except KeyboardInterrupt:
        print("\n⏹️ หยุดการวิเคราะห์โดยผู้ใช้")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
