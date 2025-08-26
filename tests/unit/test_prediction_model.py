import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import asyncio

from cs2_betting_system.models.prediction_model import PredictionModel
from cs2_betting_system.scrapers.enhanced_live_match_scraper import EnhancedPredictionModel


class TestPredictionModel:
    """Unit tests for PredictionModel"""
    
    @pytest.fixture
    def prediction_model(self):
        """Create PredictionModel instance for testing"""
        return PredictionModel()
    
    @pytest.fixture
    def enhanced_prediction_model(self):
        """Create EnhancedPredictionModel instance for testing"""
        model = EnhancedPredictionModel()
        # Mock the model loading
        model.models = {
            'test_model': Mock()
        }
        model.scaler = Mock()
        return model
    
    def test_extract_features_basic(self, prediction_model, enhanced_match_with_stats):
        """Test basic feature extraction"""
        features = prediction_model.extract_features(enhanced_match_with_stats)
        
        assert isinstance(features, dict)
        assert 'team1_rank' in features
        assert 'team2_rank' in features
        assert 'odds_team1' in features
        assert 'odds_team2' in features
        assert features['team1_rank'] == 1
        assert features['team2_rank'] == 3
    
    def test_extract_features_missing_data(self, prediction_model):
        """Test feature extraction with missing data"""
        minimal_match = {
            'match_id': 'test_001',
            'team1': 'Team A',
            'team2': 'Team B'
        }
        
        features = prediction_model.extract_features(minimal_match)
        
        assert isinstance(features, dict)
        # Should have default values for missing data
        assert features['team1_rank'] == 50  # Default ranking
        assert features['team2_rank'] == 50
        assert features['odds_team1'] == 2.0  # Default odds
    
    def test_predict_basic(self, prediction_model, enhanced_match_with_stats):
        """Test basic prediction functionality"""
        with patch.object(prediction_model, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict_proba.return_value = [[0.4, 0.6]]
            mock_load.return_value = mock_model
            
            result = prediction_model.predict(enhanced_match_with_stats)
            
            assert isinstance(result, dict)
            assert 'team1_prob' in result
            assert 'team2_prob' in result
            assert 'predicted_winner' in result
            assert 'expected_value' in result
            assert result['team1_prob'] == 0.6
            assert result['team2_prob'] == 0.4
    
    def test_calculate_expected_value(self, prediction_model):
        """Test expected value calculation"""
        ev = prediction_model._calculate_expected_value(0.6, 0.4, 1.8, 2.2)
        
        assert isinstance(ev, dict)
        assert 'team1_ev' in ev
        assert 'team2_ev' in ev
        assert 'best_bet' in ev
        
        # Team1 EV = (0.6 * (1.8 - 1)) - (1 - 0.6) = 0.48 - 0.4 = 0.08
        assert abs(ev['team1_ev'] - 0.08) < 0.01
    
    def test_heuristic_prediction_fallback(self, prediction_model, enhanced_match_with_stats):
        """Test heuristic prediction when model is unavailable"""
        with patch.object(prediction_model, '_load_model', return_value=None):
            result = prediction_model.predict(enhanced_match_with_stats)
            
            assert isinstance(result, dict)
            assert result['predicted_winner'] == 'Natus Vincere'  # Higher ranked team
            assert result['team1_prob'] > result['team2_prob']
    
    @pytest.mark.asyncio
    async def test_enhanced_extract_features(self, enhanced_prediction_model, enhanced_match_with_stats):
        """Test enhanced feature extraction with HLTV stats"""
        features_df = await enhanced_prediction_model.extract_features(enhanced_match_with_stats)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 1
        
        # Check key features
        assert 'team1_rank' in features_df.columns
        assert 'team2_rank' in features_df.columns
        assert 'rank_diff' in features_df.columns
        assert 'team1_form_10' in features_df.columns
        assert 'team2_form_10' in features_df.columns
        assert 'form_diff_10' in features_df.columns
        assert 'team1_avg_rating' in features_df.columns
        assert 'rating_diff' in features_df.columns
        assert 'h2h_total_matches' in features_df.columns
        assert 'odds_team1_avg' in features_df.columns
        assert 'strength_diff' in features_df.columns
        
        # Check values
        row = features_df.iloc[0]
        assert row['team1_rank'] == 1
        assert row['team2_rank'] == 3
        assert row['rank_diff'] == -2  # team1_rank - team2_rank
        assert row['team1_form_10'] == 0.8
        assert row['h2h_total_matches'] == 8
    
    @pytest.mark.asyncio
    async def test_enhanced_predict_match(self, enhanced_prediction_model, enhanced_match_with_stats):
        """Test enhanced match prediction"""
        # Mock model predictions
        mock_model = Mock()
        mock_model.predict_proba.return_value = [[0.35, 0.65]]
        enhanced_prediction_model.models = {'test_model': mock_model}
        
        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        enhanced_prediction_model.scaler = mock_scaler
        
        result = await enhanced_prediction_model.predict_match(enhanced_match_with_stats)
        
        assert isinstance(result, dict)
        assert 'match_id' in result
        assert 'prediction' in result
        assert 'expected_value' in result
        assert 'recommendation' in result
        
        prediction = result['prediction']
        assert 'team1_probability' in prediction
        assert 'team2_probability' in prediction
        assert 'predicted_winner' in prediction
        assert 'confidence' in prediction
        
        # Check probabilities sum to 1
        assert abs(prediction['team1_probability'] + prediction['team2_probability'] - 1.0) < 0.001
    
    def test_map_encoding(self, enhanced_prediction_model):
        """Test map name encoding"""
        assert enhanced_prediction_model._encode_map('mirage') == 4
        assert enhanced_prediction_model._encode_map('inferno') == 3
        assert enhanced_prediction_model._encode_map('unknown_map') == 0
        assert enhanced_prediction_model._encode_map('TBA') == 0
    
    def test_confidence_bucket_calculation(self, enhanced_prediction_model):
        """Test confidence bucket calculation"""
        assert enhanced_prediction_model._get_confidence_bucket(0.5) == "low"
        assert enhanced_prediction_model._get_confidence_bucket(0.65) == "medium"
        assert enhanced_prediction_model._get_confidence_bucket(0.8) == "high"
        assert enhanced_prediction_model._get_confidence_bucket(0.95) == "very_high"
    
    def test_data_completeness_calculation(self, enhanced_prediction_model):
        """Test data completeness calculation"""
        complete_features = {
            'team1_rank': 1,
            'team2_rank': 3,
            'team1_form_10': 0.8,
            'team2_form_10': 0.7,
            'team1_avg_rating': 1.12,
            'team2_avg_rating': 1.08,
            'h2h_total_matches': 8
        }
        
        completeness = enhanced_prediction_model._calculate_data_completeness(complete_features)
        assert completeness == 1.0
        
        incomplete_features = {
            'team1_rank': 1,
            'team2_rank': 0,  # Missing data
            'team1_form_10': 0,  # Missing data
            'team2_form_10': 0.7,
            'team1_avg_rating': 1.12,
            'team2_avg_rating': 0,  # Missing data
            'h2h_total_matches': 8
        }
        
        completeness = enhanced_prediction_model._calculate_data_completeness(incomplete_features)
        assert completeness < 1.0
    
    def test_confidence_score_calculation(self, enhanced_prediction_model):
        """Test confidence score calculation"""
        high_quality_features = {
            'h2h_total_matches': 15,
            'team1_map_matches': 20,
            'team2_map_matches': 18,
            'form_diff_10': 0.4,
            'odds_sources_count': 5
        }
        
        confidence = enhanced_prediction_model._calculate_confidence_score(high_quality_features)
        assert confidence > 0.5
        assert confidence <= 0.95
        
        low_quality_features = {
            'h2h_total_matches': 2,
            'team1_map_matches': 5,
            'team2_map_matches': 3,
            'form_diff_10': 0.1,
            'odds_sources_count': 1
        }
        
        confidence = enhanced_prediction_model._calculate_confidence_score(low_quality_features)
        assert confidence <= 0.7
    
    def test_expected_value_calculation(self, enhanced_prediction_model):
        """Test expected value calculation"""
        ev_result = enhanced_prediction_model._calculate_expected_value(0.6, 0.4, 1.8, 2.2)
        
        assert isinstance(ev_result, dict)
        assert 'team1_ev' in ev_result
        assert 'team2_ev' in ev_result
        assert 'team1_ev_percent' in ev_result
        assert 'team2_ev_percent' in ev_result
        assert 'best_bet' in ev_result
        assert 'edge' in ev_result
        
        # Team1 EV = (0.6 * (1.8 - 1)) - (1 - 0.6) = 0.48 - 0.4 = 0.08
        expected_ev1 = 0.08
        assert abs(ev_result['team1_ev'] - expected_ev1) < 0.01
        
        # Team2 EV = (0.4 * (2.2 - 1)) - (1 - 0.4) = 0.48 - 0.6 = -0.12
        expected_ev2 = -0.12
        assert abs(ev_result['team2_ev'] - expected_ev2) < 0.01
        
        assert ev_result['best_bet'] == 'team1'
        assert ev_result['edge'] == ev_result['team1_ev']
    
    def test_betting_recommendation_generation(self, enhanced_prediction_model):
        """Test betting recommendation generation"""
        ev_analysis = {
            'team1_ev': 0.08,
            'team2_ev': -0.05,
            'edge': 0.08,
            'best_bet': 'team1'
        }
        
        recommendation = enhanced_prediction_model._generate_betting_recommendation(
            0.6, 0.4, ev_analysis, 0.8
        )
        
        assert isinstance(recommendation, dict)
        assert 'action' in recommendation
        assert 'team' in recommendation
        assert 'suggested_stake' in recommendation
        assert 'kelly_criterion' in recommendation
        assert 'reasoning' in recommendation
        
        assert recommendation['action'] == 'BET'
        assert recommendation['team'] == 'team1'
        assert recommendation['kelly_criterion'] > 0
        assert len(recommendation['reasoning']) > 0
    
    def test_betting_recommendation_skip(self, enhanced_prediction_model):
        """Test betting recommendation when should skip"""
        ev_analysis = {
            'team1_ev': 0.02,  # Low EV
            'team2_ev': -0.05,
            'edge': 0.02,
            'best_bet': 'team1'
        }
        
        recommendation = enhanced_prediction_model._generate_betting_recommendation(
            0.52, 0.48, ev_analysis, 0.4  # Low confidence
        )
        
        assert recommendation['action'] == 'SKIP'
        assert recommendation['suggested_stake'] == 0
        assert len(recommendation['reasoning']) > 0
    
    @pytest.mark.asyncio
    async def test_model_loading_error_handling(self, enhanced_prediction_model):
        """Test error handling in model loading"""
        with patch.object(enhanced_prediction_model, 'load_models') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            # Should not raise exception
            await enhanced_prediction_model.initialize()
            
            # Models should be empty
            assert len(enhanced_prediction_model.models) == 0
    
    def test_feature_column_ensuring(self, enhanced_prediction_model):
        """Test ensuring all required feature columns exist"""
        incomplete_df = pd.DataFrame([{
            'team1_rank': 1,
            'team2_rank': 3,
            'some_extra_feature': 0.5
        }])
        
        complete_df = enhanced_prediction_model._ensure_feature_columns(incomplete_df)
        
        required_columns = [
            'team1_rank', 'team2_rank', 'rank_diff', 'team1_form_10', 'team2_form_10',
            'form_diff_10', 'team1_momentum', 'team2_momentum', 'momentum_diff',
            'team1_avg_rating', 'team2_avg_rating', 'rating_diff',
            'h2h_total_matches', 'h2h_team1_winrate', 'odds_team1_avg', 'odds_team2_avg',
            'strength_diff', 'data_completeness', 'prediction_confidence'
        ]
        
        for col in required_columns:
            assert col in complete_df.columns
        
        # Original columns should be preserved
        assert 'some_extra_feature' in complete_df.columns
        
        # Missing columns should be filled with 0
        assert complete_df['team1_form_10'].iloc[0] == 0
        assert complete_df['h2h_total_matches'].iloc[0] == 0
