"""
Data loader for tennis match data.
Reads and concatenates raw CSV files from the tennis_atp directory.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Optional


class TennisDataLoader:
    """Loads and processes tennis match data from CSV files."""
    
    def __init__(self, data_dir: str = "data/raw/tennis_atp"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        
    def get_match_files(self) -> List[Path]:
        """Get all match CSV files (excluding doubles, futures, qualifiers)."""
        files = []
        for file in self.data_dir.glob("atp_matches_*.csv"):
            # Skip doubles, futures, and qualifier files
            if any(skip in file.name for skip in ["doubles", "futures", "qual_chall"]):
                continue
            files.append(file)
        return sorted(files)
    
    def load_matches(self, start_year: Optional[int] = None, 
                    end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load match data from CSV files.
        
        Args:
            start_year: Filter matches from this year onwards
            end_year: Filter matches up to this year
            
        Returns:
            Concatenated DataFrame of all matches
        """
        files = self.get_match_files()
        
        if not files:
            raise FileNotFoundError(f"No match files found in {self.data_dir}")
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                # Extract year from filename for filtering
                year = int(file.stem.split('_')[-1])
                
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
                    
                dfs.append(df)
                print(f"Loaded {file.name} ({len(df)} matches)")
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data loaded")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total matches loaded: {len(combined_df)}")
        
        return combined_df
    
    def load_players(self) -> pd.DataFrame:
        """Load player data."""
        players_file = self.data_dir / "atp_players.csv"
        if not players_file.exists():
            raise FileNotFoundError(f"Players file not found: {players_file}")
        
        return pd.read_csv(players_file)
    
    def load_rankings(self) -> pd.DataFrame:
        """Load ranking data."""
        ranking_files = list(self.data_dir.glob("atp_rankings_*.csv"))
        if not ranking_files:
            raise FileNotFoundError(f"No ranking files found in {self.data_dir}")
        
        dfs = []
        for file in ranking_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Loaded {file.name} ({len(df)} rankings)")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No ranking data loaded")
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Total rankings loaded: {len(combined_df)}")
        
        return combined_df


def save_processed_data(df: pd.DataFrame, filename: str = "matches_combined.csv"):
    """Save processed data to the processed directory."""
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = TennisDataLoader()
    
    # Load recent matches (last 10 years)
    matches = loader.load_matches(start_year=2014)
    
    # Basic cleaning
    matches = matches.dropna(subset=['winner_name', 'loser_name'])
    
    # Save processed data
    save_processed_data(matches) 