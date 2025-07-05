"""
Elo rating system for tennis players.
Builds and updates Elo ratings based on match outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class TennisElo:
    """Elo rating system for tennis players."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """
        Initialize the Elo rating system.
        
        Args:
            k_factor: K-factor determines how much ratings change after each match
            initial_rating: Starting rating for new players
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.player_ratings = {}  # player_name -> current_rating
        self.rating_history = {}  # player_name -> [(date, rating), ...]
    
    def get_player_rating(self, player_name: str) -> float:
        """Get current rating for a player."""
        return self.player_ratings.get(player_name, self.initial_rating)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B
            
        Returns:
            Expected score (0-1) for player A
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner: str, loser: str, surface: Optional[str] = None,
                      tournament_level: Optional[str] = None) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.
        
        Args:
            winner: Name of the winning player
            loser: Name of the losing player
            surface: Surface type (optional, for surface-specific ratings)
            tournament_level: Tournament level (optional, for level-specific ratings)
            
        Returns:
            Tuple of (new_winner_rating, new_loser_rating)
        """
        # Get current ratings
        winner_rating = self.get_player_rating(winner)
        loser_rating = self.get_player_rating(loser)
        
        # Calculate expected scores
        winner_expected = self.expected_score(winner_rating, loser_rating)
        loser_expected = self.expected_score(loser_rating, winner_rating)
        
        # Calculate rating changes
        winner_change = self.k_factor * (1 - winner_expected)
        loser_change = self.k_factor * (0 - loser_expected)
        
        # Apply adjustments for surface and tournament level
        if surface:
            winner_change *= self._get_surface_multiplier(surface)
            loser_change *= self._get_surface_multiplier(surface)
        
        if tournament_level:
            winner_change *= self._get_tournament_multiplier(tournament_level)
            loser_change *= self._get_tournament_multiplier(tournament_level)
        
        # Update ratings
        new_winner_rating = winner_rating + winner_change
        new_loser_rating = loser_rating + loser_change
        
        # Store new ratings
        self.player_ratings[winner] = new_winner_rating
        self.player_ratings[loser] = new_loser_rating
        
        return new_winner_rating, new_loser_rating
    
    def _get_surface_multiplier(self, surface: str) -> float:
        """Get multiplier for surface-specific ratings."""
        multipliers = {
            'Hard': 1.0,
            'Clay': 1.0,
            'Grass': 1.0,
            'Carpet': 1.0
        }
        return multipliers.get(surface, 1.0)
    
    def _get_tournament_multiplier(self, tournament_level: str) -> float:
        """Get multiplier for tournament level."""
        multipliers = {
            'G': 1.5,  # Grand Slam
            'M': 1.2,  # Masters
            'A': 1.0,  # ATP Tour
            'C': 0.8,  # Challenger
            'F': 0.6,  # Futures
            'D': 0.4   # Davis Cup
        }
        return multipliers.get(tournament_level, 1.0)
    
    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches and update Elo ratings.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            DataFrame with added Elo columns
        """
        print("Processing matches for Elo ratings...")
        
        # Initialize Elo columns
        matches_df['winner_elo'] = np.nan
        matches_df['loser_elo'] = np.nan
        matches_df['winner_elo_change'] = np.nan
        matches_df['loser_elo_change'] = np.nan
        
        # Sort by date for chronological processing
        matches_df = matches_df.sort_values('tourney_date')
        
        for idx, row in matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']):
                continue
            
            winner = row['winner_name']
            loser = row['loser_name']
            surface = row.get('surface', None)
            tournament_level = row.get('tourney_level', None)
            
            # Get current ratings before the match
            winner_rating_before = self.get_player_rating(winner)
            loser_rating_before = self.get_player_rating(loser)
            
            # Update ratings
            new_winner_rating, new_loser_rating = self.update_ratings(
                winner, loser, surface, tournament_level
            )
            
            # Store in DataFrame
            matches_df.loc[idx, 'winner_elo'] = winner_rating_before
            matches_df.loc[idx, 'loser_elo'] = loser_rating_before
            matches_df.loc[idx, 'winner_elo_change'] = new_winner_rating - winner_rating_before
            matches_df.loc[idx, 'loser_elo_change'] = new_loser_rating - loser_rating_before
            
            # Store in history
            match_date = row['tourney_date']
            if winner not in self.rating_history:
                self.rating_history[winner] = []
            if loser not in self.rating_history:
                self.rating_history[loser] = []
            
            self.rating_history[winner].append((match_date, new_winner_rating))
            self.rating_history[loser].append((match_date, new_loser_rating))
        
        print(f"Processed {len(matches_df)} matches for {len(self.player_ratings)} players")
        return matches_df
    
    def get_top_players(self, n: int = 10) -> pd.DataFrame:
        """Get top N players by current Elo rating."""
        sorted_players = sorted(
            self.player_ratings.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_players = []
        for i, (player, rating) in enumerate(sorted_players[:n]):
            top_players.append({
                'rank': i + 1,
                'player': player,
                'elo_rating': rating
            })
        
        return pd.DataFrame(top_players)
    
    def get_player_history(self, player_name: str) -> pd.DataFrame:
        """Get rating history for a specific player."""
        if player_name not in self.rating_history:
            return pd.DataFrame()
        
        history = self.rating_history[player_name]
        df = pd.DataFrame(history, columns=['date', 'rating'])
        df = df.sort_values('date')
        return df
    
    def save_ratings(self, filename: str = "elo_ratings.csv"):
        """Save current ratings to CSV."""
        ratings_df = pd.DataFrame([
            {'player': player, 'elo_rating': rating}
            for player, rating in self.player_ratings.items()
        ])
        ratings_df = ratings_df.sort_values('elo_rating', ascending=False)
        ratings_df.to_csv(filename, index=False)
        print(f"Saved Elo ratings to {filename}")
    
    def load_ratings(self, filename: str):
        """Load ratings from CSV."""
        try:
            ratings_df = pd.read_csv(filename)
            self.player_ratings = dict(zip(ratings_df['player'], ratings_df['elo_rating']))
            print(f"Loaded Elo ratings from {filename}")
        except FileNotFoundError:
            print(f"Ratings file {filename} not found")


class SurfaceSpecificElo(TennisElo):
    """Elo rating system with surface-specific ratings."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        super().__init__(k_factor, initial_rating)
        self.surface_ratings = {}  # (player, surface) -> rating
    
    def get_player_surface_rating(self, player_name: str, surface: str) -> float:
        """Get current rating for a player on a specific surface."""
        return self.surface_ratings.get((player_name, surface), self.initial_rating)
    
    def update_surface_ratings(self, winner: str, loser: str, surface: str,
                             tournament_level: Optional[str] = None) -> Tuple[float, float]:
        """Update surface-specific Elo ratings."""
        # Get current surface ratings
        winner_rating = self.get_player_surface_rating(winner, surface)
        loser_rating = self.get_player_surface_rating(loser, surface)
        
        # Calculate expected scores
        winner_expected = self.expected_score(winner_rating, loser_rating)
        loser_expected = self.expected_score(loser_rating, winner_rating)
        
        # Calculate rating changes
        winner_change = self.k_factor * (1 - winner_expected)
        loser_change = self.k_factor * (0 - loser_expected)
        
        # Apply tournament level adjustment
        if tournament_level:
            winner_change *= self._get_tournament_multiplier(tournament_level)
            loser_change *= self._get_tournament_multiplier(tournament_level)
        
        # Update surface ratings
        new_winner_rating = winner_rating + winner_change
        new_loser_rating = loser_rating + loser_change
        
        self.surface_ratings[(winner, surface)] = new_winner_rating
        self.surface_ratings[(loser, surface)] = new_loser_rating
        
        return new_winner_rating, new_loser_rating
    
    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches with surface-specific ratings."""
        print("Processing matches for surface-specific Elo ratings...")
        
        # Initialize Elo columns
        matches_df['winner_surface_elo'] = np.nan
        matches_df['loser_surface_elo'] = np.nan
        matches_df['winner_surface_elo_change'] = np.nan
        matches_df['loser_surface_elo_change'] = np.nan
        
        # Sort by date
        matches_df = matches_df.sort_values('tourney_date')
        
        for idx, row in matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']) or pd.isna(row['surface']):
                continue
            
            winner = row['winner_name']
            loser = row['loser_name']
            surface = row['surface']
            tournament_level = row.get('tourney_level', None)
            
            # Get current surface ratings
            winner_rating_before = self.get_player_surface_rating(winner, surface)
            loser_rating_before = self.get_player_surface_rating(loser, surface)
            
            # Update surface ratings
            new_winner_rating, new_loser_rating = self.update_surface_ratings(
                winner, loser, surface, tournament_level
            )
            
            # Store in DataFrame
            matches_df.loc[idx, 'winner_surface_elo'] = winner_rating_before
            matches_df.loc[idx, 'loser_surface_elo'] = loser_rating_before
            matches_df.loc[idx, 'winner_surface_elo_change'] = new_winner_rating - winner_rating_before
            matches_df.loc[idx, 'loser_surface_elo_change'] = new_loser_rating - loser_rating_before
        
        print(f"Processed {len(matches_df)} matches for surface-specific ratings")
        return matches_df 