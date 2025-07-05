"""
Feature engineering for tennis match prediction.
Creates features like head-to-head records, surface statistics, and other tennis-specific features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class TennisFeatureEngineer:
    """Engineers features for tennis match prediction."""
    
    def __init__(self, matches_df: pd.DataFrame, players_df: pd.DataFrame = None):
        """
        Initialize the feature engineer.
        
        Args:
            matches_df: DataFrame containing match data
            players_df: DataFrame containing player data (optional)
        """
        self.matches_df = matches_df.copy()
        self.players_df = players_df
        
        # Convert date columns
        if 'tourney_date' in self.matches_df.columns:
            self.matches_df['tourney_date'] = pd.to_datetime(self.matches_df['tourney_date'], format='%Y%m%d')
    
    def create_head_to_head_features(self) -> pd.DataFrame:
        """Create head-to-head features for each match."""
        print("Creating head-to-head features...")
        
        # Initialize H2H columns
        self.matches_df['h2h_wins_player1'] = 0
        self.matches_df['h2h_losses_player1'] = 0
        self.matches_df['h2h_wins_player2'] = 0
        self.matches_df['h2h_losses_player2'] = 0
        
        # Sort by date for chronological processing
        self.matches_df = self.matches_df.sort_values('tourney_date')
        
        for idx, row in self.matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']):
                continue
                
            # Get previous matches between these players
            player1, player2 = row['winner_name'], row['loser_name']
            
            # Look at matches before current match
            prev_matches = self.matches_df[
                (self.matches_df['tourney_date'] < row['tourney_date']) &
                (
                    ((self.matches_df['winner_name'] == player1) & (self.matches_df['loser_name'] == player2)) |
                    ((self.matches_df['winner_name'] == player2) & (self.matches_df['loser_name'] == player1))
                )
            ]
            
            # Count wins for each player
            p1_wins = len(prev_matches[
                (prev_matches['winner_name'] == player1) & (prev_matches['loser_name'] == player2)
            ])
            p2_wins = len(prev_matches[
                (prev_matches['winner_name'] == player2) & (prev_matches['loser_name'] == player1)
            ])
            
            # Update current row
            self.matches_df.loc[idx, 'h2h_wins_player1'] = p1_wins
            self.matches_df.loc[idx, 'h2h_losses_player1'] = p2_wins
            self.matches_df.loc[idx, 'h2h_wins_player2'] = p2_wins
            self.matches_df.loc[idx, 'h2h_losses_player1'] = p1_wins
        
        return self.matches_df
    
    def create_surface_features(self) -> pd.DataFrame:
        """Create surface-specific performance features."""
        print("Creating surface features...")
        
        # Initialize surface columns
        surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        for surface in surfaces:
            self.matches_df[f'{surface.lower()}_wins_player1'] = 0
            self.matches_df[f'{surface.lower()}_losses_player1'] = 0
            self.matches_df[f'{surface.lower()}_wins_player2'] = 0
            self.matches_df[f'{surface.lower()}_losses_player2'] = 0
        
        # Sort by date
        self.matches_df = self.matches_df.sort_values('tourney_date')
        
        for idx, row in self.matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']) or pd.isna(row['surface']):
                continue
                
            player1, player2 = row['winner_name'], row['loser_name']
            surface = row['surface']
            
            # Get previous matches on this surface
            prev_matches = self.matches_df[
                (self.matches_df['tourney_date'] < row['tourney_date']) &
                (self.matches_df['surface'] == surface)
            ]
            
            # Count wins/losses for each player on this surface
            p1_wins = len(prev_matches[
                (prev_matches['winner_name'] == player1) & (prev_matches['loser_name'] != player1)
            ])
            p1_losses = len(prev_matches[
                (prev_matches['loser_name'] == player1) & (prev_matches['winner_name'] != player1)
            ])
            
            p2_wins = len(prev_matches[
                (prev_matches['winner_name'] == player2) & (prev_matches['loser_name'] != player2)
            ])
            p2_losses = len(prev_matches[
                (prev_matches['loser_name'] == player2) & (prev_matches['winner_name'] != player2)
            ])
            
            # Update current row
            surface_lower = surface.lower()
            if surface_lower in ['hard', 'clay', 'grass', 'carpet']:
                self.matches_df.loc[idx, f'{surface_lower}_wins_player1'] = p1_wins
                self.matches_df.loc[idx, f'{surface_lower}_losses_player1'] = p1_losses
                self.matches_df.loc[idx, f'{surface_lower}_wins_player2'] = p2_wins
                self.matches_df.loc[idx, f'{surface_lower}_losses_player2'] = p2_losses
        
        return self.matches_df
    
    def create_recent_form_features(self, days_back: int = 365) -> pd.DataFrame:
        """Create recent form features based on last N days."""
        print(f"Creating recent form features (last {days_back} days)...")
        
        # Initialize form columns
        self.matches_df['recent_wins_player1'] = 0
        self.matches_df['recent_losses_player1'] = 0
        self.matches_df['recent_wins_player2'] = 0
        self.matches_df['recent_losses_player2'] = 0
        
        # Sort by date
        self.matches_df = self.matches_df.sort_values('tourney_date')
        
        for idx, row in self.matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']) or pd.isna(row['tourney_date']):
                continue
                
            player1, player2 = row['winner_name'], row['loser_name']
            match_date = row['tourney_date']
            
            # Calculate cutoff date
            cutoff_date = match_date - timedelta(days=days_back)
            
            # Get recent matches for each player
            recent_matches = self.matches_df[
                (self.matches_df['tourney_date'] >= cutoff_date) &
                (self.matches_df['tourney_date'] < match_date)
            ]
            
            # Count recent wins/losses for each player
            p1_wins = len(recent_matches[recent_matches['winner_name'] == player1])
            p1_losses = len(recent_matches[recent_matches['loser_name'] == player1])
            
            p2_wins = len(recent_matches[recent_matches['winner_name'] == player2])
            p2_losses = len(recent_matches[recent_matches['loser_name'] == player2])
            
            # Update current row
            self.matches_df.loc[idx, 'recent_wins_player1'] = p1_wins
            self.matches_df.loc[idx, 'recent_losses_player1'] = p1_losses
            self.matches_df.loc[idx, 'recent_wins_player2'] = p2_wins
            self.matches_df.loc[idx, 'recent_losses_player2'] = p2_losses
        
        return self.matches_df
    
    def create_ranking_features(self, rankings_df: pd.DataFrame) -> pd.DataFrame:
        """Create ranking-based features."""
        print("Creating ranking features...")
        
        # Initialize ranking columns
        self.matches_df['winner_rank'] = np.nan
        self.matches_df['loser_rank'] = np.nan
        self.matches_df['rank_diff'] = np.nan
        
        # Sort by date
        self.matches_df = self.matches_df.sort_values('tourney_date')
        rankings_df = rankings_df.sort_values('ranking_date')
        
        for idx, row in self.matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']) or pd.isna(row['tourney_date']):
                continue
                
            winner, loser = row['winner_name'], row['loser_name']
            match_date = row['tourney_date']
            
            # Get most recent rankings before match date
            recent_rankings = rankings_df[
                rankings_df['ranking_date'] <= match_date
            ].groupby('player').last().reset_index()
            
            # Get rankings for winner and loser
            winner_rank = recent_rankings[recent_rankings['player'] == winner]['rank'].iloc[0] if len(recent_rankings[recent_rankings['player'] == winner]) > 0 else np.nan
            loser_rank = recent_rankings[recent_rankings['player'] == loser]['rank'].iloc[0] if len(recent_rankings[recent_rankings['player'] == loser]) > 0 else np.nan
            
            # Update current row
            self.matches_df.loc[idx, 'winner_rank'] = winner_rank
            self.matches_df.loc[idx, 'loser_rank'] = loser_rank
            
            if not pd.isna(winner_rank) and not pd.isna(loser_rank):
                self.matches_df.loc[idx, 'rank_diff'] = loser_rank - winner_rank
        
        return self.matches_df
    
    def create_match_features(self) -> pd.DataFrame:
        """Create match-specific features."""
        print("Creating match features...")
        
        # Tournament level (Grand Slam, Masters, etc.)
        self.matches_df['tourney_level'] = self.matches_df['tourney_level'].astype('category')
        
        # Match length (sets played)
        self.matches_df['sets_played'] = self.matches_df['score'].str.count('-')
        
        # Round importance (final, semi-final, etc.)
        round_importance = {
            'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3, 'R64': 2, 'R128': 1
        }
        self.matches_df['round_importance'] = self.matches_df['round'].map(round_importance)
        
        # Surface type
        self.matches_df['surface'] = self.matches_df['surface'].astype('category')
        
        return self.matches_df
    
    def engineer_all_features(self, rankings_df: pd.DataFrame = None) -> pd.DataFrame:
        """Engineer all features."""
        print("Starting feature engineering...")
        
        # Create all features
        self.create_match_features()
        self.create_head_to_head_features()
        self.create_surface_features()
        self.create_recent_form_features()
        
        if rankings_df is not None:
            self.create_ranking_features(rankings_df)
        
        print("Feature engineering completed!")
        return self.matches_df


def prepare_features_for_training(matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for training a model.
    
    Args:
        matches_df: DataFrame with engineered features
        
    Returns:
        Tuple of (features, target)
    """
    # Select features for training
    feature_columns = [
        # H2H features
        'h2h_wins_player1', 'h2h_losses_player1', 'h2h_wins_player2', 'h2h_losses_player2',
        
        # Surface features
        'hard_wins_player1', 'hard_losses_player1', 'hard_wins_player2', 'hard_losses_player2',
        'clay_wins_player1', 'clay_losses_player1', 'clay_wins_player2', 'clay_losses_player2',
        'grass_wins_player1', 'grass_losses_player1', 'grass_wins_player2', 'grass_losses_player2',
        
        # Recent form
        'recent_wins_player1', 'recent_losses_player1', 'recent_wins_player2', 'recent_losses_player2',
        
        # Ranking features
        'winner_rank', 'loser_rank', 'rank_diff',
        
        # Match features
        'sets_played', 'round_importance'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in matches_df.columns]
    
    # Create features DataFrame
    features = matches_df[available_features].copy()
    
    # Handle missing values
    features = features.fillna(0)
    
    # Create target (1 if player1 wins, 0 if player2 wins)
    # For this example, we'll use winner_name vs loser_name
    # In practice, you might want to create balanced training data
    target = (matches_df['winner_name'] == matches_df['winner_name']).astype(int)
    
    return features, target 