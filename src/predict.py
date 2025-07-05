"""
Prediction module for tennis match prediction.
Loads trained model and scaler to make predictions on new matches.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .data_loader import TennisDataLoader
from .feature_engineering import TennisFeatureEngineer
from .elo import TennisElo


class TennisMatchPredictor:
    """Make predictions on tennis matches using a trained model."""
    
    def __init__(self, model_path: str = "models/gradient_boosting.pkl",
                 scaler_path: str = "models/scaler.pkl"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Load model and scaler
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler."""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            print(f"Model loaded from {self.model_path}")
            print(f"Scaler loaded from {self.scaler_path}")
            
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please train a model first using src/train.py")
            raise
    
    def prepare_match_features(self, player1: str, player2: str, 
                             surface: str = None, tournament_level: str = None,
                             recent_matches_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for a specific match.
        
        Args:
            player1: Name of first player
            player2: Name of second player
            surface: Surface type (optional)
            tournament_level: Tournament level (optional)
            recent_matches_df: DataFrame with recent matches for feature calculation
            
        Returns:
            DataFrame with features for the match
        """
        if recent_matches_df is None:
            raise ValueError("recent_matches_df is required for feature calculation")
        
        # Create a single row DataFrame for the match
        match_data = {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'tourney_level': tournament_level
        }
        
        # Calculate features for both players
        features = self._calculate_player_features(player1, player2, recent_matches_df)
        match_data.update(features)
        
        return pd.DataFrame([match_data])
    
    def _calculate_player_features(self, player1: str, player2: str, 
                                 matches_df: pd.DataFrame) -> Dict:
        """Calculate features for two players."""
        features = {}
        
        # Get recent matches for each player
        player1_matches = matches_df[
            (matches_df['winner_name'] == player1) | (matches_df['loser_name'] == player1)
        ].sort_values('tourney_date', ascending=False)
        
        player2_matches = matches_df[
            (matches_df['winner_name'] == player2) | (matches_df['loser_name'] == player2)
        ].sort_values('tourney_date', ascending=False)
        
        # Head-to-head features
        h2h_matches = matches_df[
            (
                ((matches_df['winner_name'] == player1) & (matches_df['loser_name'] == player2)) |
                ((matches_df['winner_name'] == player2) & (matches_df['loser_name'] == player1))
            )
        ]
        
        p1_h2h_wins = len(h2h_matches[
            (h2h_matches['winner_name'] == player1) & (h2h_matches['loser_name'] == player2)
        ])
        p2_h2h_wins = len(h2h_matches[
            (h2h_matches['winner_name'] == player2) & (h2h_matches['loser_name'] == player1)
        ])
        
        features.update({
            'h2h_wins_player1': p1_h2h_wins,
            'h2h_losses_player1': p2_h2h_wins,
            'h2h_wins_player2': p2_h2h_wins,
            'h2h_losses_player2': p1_h2h_wins
        })
        
        # Recent form (last 365 days)
        recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        recent_matches = matches_df[matches_df['tourney_date'] >= recent_cutoff]
        
        p1_recent_wins = len(recent_matches[recent_matches['winner_name'] == player1])
        p1_recent_losses = len(recent_matches[recent_matches['loser_name'] == player1])
        p2_recent_wins = len(recent_matches[recent_matches['winner_name'] == player2])
        p2_recent_losses = len(recent_matches[recent_matches['loser_name'] == player2])
        
        features.update({
            'recent_wins_player1': p1_recent_wins,
            'recent_losses_player1': p1_recent_losses,
            'recent_wins_player2': p2_recent_wins,
            'recent_losses_player2': p2_recent_losses
        })
        
        # Surface-specific features
        if surface:
            surface_matches = matches_df[matches_df['surface'] == surface]
            
            p1_surface_wins = len(surface_matches[
                (surface_matches['winner_name'] == player1) & (surface_matches['loser_name'] != player1)
            ])
            p1_surface_losses = len(surface_matches[
                (surface_matches['loser_name'] == player1) & (surface_matches['winner_name'] != player1)
            ])
            
            p2_surface_wins = len(surface_matches[
                (surface_matches['winner_name'] == player2) & (surface_matches['loser_name'] != player2)
            ])
            p2_surface_losses = len(surface_matches[
                (surface_matches['loser_name'] == player2) & (surface_matches['winner_name'] != player2)
            ])
            
            surface_lower = surface.lower()
            features.update({
                f'{surface_lower}_wins_player1': p1_surface_wins,
                f'{surface_lower}_losses_player1': p1_surface_losses,
                f'{surface_lower}_wins_player2': p2_surface_wins,
                f'{surface_lower}_losses_player2': p2_surface_losses
            })
        
        # Elo ratings (if available)
        if 'winner_elo' in matches_df.columns and 'loser_elo' in matches_df.columns:
            # Get most recent Elo ratings
            p1_elo = self._get_latest_elo(player1, matches_df)
            p2_elo = self._get_latest_elo(player2, matches_df)
            
            features.update({
                'winner_elo': p1_elo,
                'loser_elo': p2_elo
            })
        
        # Ranking features (if available)
        if 'winner_rank' in matches_df.columns and 'loser_rank' in matches_df.columns:
            p1_rank = self._get_latest_rank(player1, matches_df)
            p2_rank = self._get_latest_rank(player2, matches_df)
            
            features.update({
                'winner_rank': p1_rank,
                'loser_rank': p2_rank,
                'rank_diff': p2_rank - p1_rank if p1_rank and p2_rank else 0
            })
        
        # Default values for missing features
        default_features = {
            'sets_played': 3,  # Default to 3 sets
            'round_importance': 4  # Default to R16 level
        }
        
        features.update(default_features)
        
        return features
    
    def _get_latest_elo(self, player: str, matches_df: pd.DataFrame) -> float:
        """Get the most recent Elo rating for a player."""
        player_matches = matches_df[
            (matches_df['winner_name'] == player) | (matches_df['loser_name'] == player)
        ].sort_values('tourney_date', ascending=False)
        
        if len(player_matches) == 0:
            return 1500  # Default Elo rating
        
        latest_match = player_matches.iloc[0]
        if latest_match['winner_name'] == player:
            return latest_match.get('winner_elo', 1500)
        else:
            return latest_match.get('loser_elo', 1500)
    
    def _get_latest_rank(self, player: str, matches_df: pd.DataFrame) -> Optional[int]:
        """Get the most recent ranking for a player."""
        player_matches = matches_df[
            (matches_df['winner_name'] == player) | (matches_df['loser_name'] == player)
        ].sort_values('tourney_date', ascending=False)
        
        if len(player_matches) == 0:
            return None
        
        latest_match = player_matches.iloc[0]
        if latest_match['winner_name'] == player:
            return latest_match.get('winner_rank')
        else:
            return latest_match.get('loser_rank')
    
    def predict_match(self, player1: str, player2: str, surface: str = None,
                     tournament_level: str = None, recent_matches_df: pd.DataFrame = None) -> Dict:
        """
        Predict the outcome of a match.
        
        Args:
            player1: Name of first player
            player2: Name of second player
            surface: Surface type (optional)
            tournament_level: Tournament level (optional)
            recent_matches_df: DataFrame with recent matches for feature calculation
            
        Returns:
            Dictionary with prediction results
        """
        # Prepare features
        features_df = self.prepare_match_features(
            player1, player2, surface, tournament_level, recent_matches_df
        )
        
        # Select feature columns (must match training features)
        feature_columns = [
            'h2h_wins_player1', 'h2h_losses_player1', 'h2h_wins_player2', 'h2h_losses_player2',
            'recent_wins_player1', 'recent_losses_player1', 'recent_wins_player2', 'recent_losses_player2',
            'winner_elo', 'loser_elo', 'winner_rank', 'loser_rank', 'rank_diff',
            'sets_played', 'round_importance'
        ]
        
        # Add surface features if available
        if surface:
            surface_lower = surface.lower()
            feature_columns.extend([
                f'{surface_lower}_wins_player1', f'{surface_lower}_losses_player1',
                f'{surface_lower}_wins_player2', f'{surface_lower}_losses_player2'
            ])
        
        # Filter available features
        available_features = [col for col in feature_columns if col in features_df.columns]
        features = features_df[available_features].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Prepare result
        result = {
            'player1': player1,
            'player2': player2,
            'surface': surface,
            'tournament_level': tournament_level,
            'prediction': 'player1_wins' if prediction == 1 else 'player2_wins',
            'player1_win_probability': probability[1] if prediction == 1 else probability[0],
            'player2_win_probability': probability[0] if prediction == 1 else probability[1],
            'confidence': max(probability),
            'features_used': available_features
        }
        
        return result
    
    def predict_multiple_matches(self, matches: List[Dict], 
                               recent_matches_df: pd.DataFrame) -> List[Dict]:
        """
        Predict multiple matches.
        
        Args:
            matches: List of match dictionaries with player1, player2, surface, tournament_level
            recent_matches_df: DataFrame with recent matches for feature calculation
            
        Returns:
            List of prediction results
        """
        predictions = []
        
        for match in matches:
            prediction = self.predict_match(
                player1=match['player1'],
                player2=match['player2'],
                surface=match.get('surface'),
                tournament_level=match.get('tournament_level'),
                recent_matches_df=recent_matches_df
            )
            predictions.append(prediction)
        
        return predictions
    
    def get_prediction_summary(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        Create a summary of predictions.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            DataFrame with prediction summary
        """
        summary_data = []
        
        for pred in predictions:
            summary_data.append({
                'Player 1': pred['player1'],
                'Player 2': pred['player2'],
                'Surface': pred['surface'],
                'Prediction': pred['prediction'],
                'Player 1 Win %': f"{pred['player1_win_probability']:.1%}",
                'Player 2 Win %': f"{pred['player2_win_probability']:.1%}",
                'Confidence': f"{pred['confidence']:.1%}"
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Example usage of the predictor."""
    # Initialize predictor
    predictor = TennisMatchPredictor()
    
    # Load recent data for feature calculation
    loader = TennisDataLoader()
    recent_matches = loader.load_matches(start_year=2020)
    
    # Example predictions
    matches_to_predict = [
        {
            'player1': 'Novak Djokovic',
            'player2': 'Rafael Nadal',
            'surface': 'Clay',
            'tournament_level': 'G'
        },
        {
            'player1': 'Roger Federer',
            'player2': 'Andy Murray',
            'surface': 'Grass',
            'tournament_level': 'G'
        }
    ]
    
    # Make predictions
    predictions = predictor.predict_multiple_matches(matches_to_predict, recent_matches)
    
    # Show summary
    summary = predictor.get_prediction_summary(predictions)
    print("\nMatch Predictions:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main() 