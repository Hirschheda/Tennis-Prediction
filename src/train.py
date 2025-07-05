"""
Training module for tennis match prediction.
Trains and calibrates a classifier for predicting match outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import pickle
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import TennisDataLoader
from .feature_engineering import TennisFeatureEngineer, prepare_features_for_training
from .elo import TennisElo


class TennisMatchPredictor:
    """Train and evaluate tennis match prediction models."""
    
    def __init__(self, data_dir: str = "data/raw/tennis_atp"):
        """
        Initialize the predictor.
        
        Args:
            data_dir: Path to the tennis data directory
        """
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_info = {}
        
    def load_and_prepare_data(self, start_year: int = 2014, end_year: int = None) -> pd.DataFrame:
        """
        Load and prepare data for training.
        
        Args:
            start_year: Start year for training data
            end_year: End year for training data
            
        Returns:
            Prepared DataFrame with features
        """
        print("Loading data...")
        
        # Load data
        loader = TennisDataLoader(self.data_dir)
        matches_df = loader.load_matches(start_year=start_year, end_year=end_year)
        
        # Basic cleaning
        matches_df = matches_df.dropna(subset=['winner_name', 'loser_name', 'surface'])
        
        # Engineer features
        print("Engineering features...")
        feature_engineer = TennisFeatureEngineer(matches_df)
        matches_df = feature_engineer.engineer_all_features()
        
        # Add Elo ratings
        print("Adding Elo ratings...")
        elo_system = TennisElo()
        matches_df = elo_system.process_matches(matches_df)
        
        return matches_df
    
    def create_balanced_dataset(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a balanced dataset for training.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            Balanced DataFrame
        """
        print("Creating balanced dataset...")
        
        # Create balanced training data
        balanced_matches = []
        
        for idx, row in matches_df.iterrows():
            if pd.isna(row['winner_name']) or pd.isna(row['loser_name']):
                continue
            
            # Original match (winner vs loser)
            match_data = row.copy()
            match_data['player1'] = row['winner_name']
            match_data['player2'] = row['loser_name']
            match_data['player1_wins'] = 1
            
            # Swap players for balanced data
            swap_data = row.copy()
            swap_data['player1'] = row['loser_name']
            swap_data['player2'] = row['winner_name']
            swap_data['player1_wins'] = 0
            
            # Swap feature columns
            self._swap_player_features(swap_data)
            
            balanced_matches.append(match_data)
            balanced_matches.append(swap_data)
        
        balanced_df = pd.DataFrame(balanced_matches)
        print(f"Created balanced dataset with {len(balanced_df)} samples")
        
        return balanced_df
    
    def _swap_player_features(self, row: pd.Series):
        """Swap player-specific features in a row."""
        feature_pairs = [
            ('h2h_wins_player1', 'h2h_wins_player2'),
            ('h2h_losses_player1', 'h2h_losses_player2'),
            ('recent_wins_player1', 'recent_wins_player2'),
            ('recent_losses_player1', 'recent_losses_player2'),
            ('winner_elo', 'loser_elo'),
            ('winner_rank', 'loser_rank')
        ]
        
        # Add surface features
        surfaces = ['hard', 'clay', 'grass']
        for surface in surfaces:
            feature_pairs.extend([
                (f'{surface}_wins_player1', f'{surface}_wins_player2'),
                (f'{surface}_losses_player1', f'{surface}_losses_player2')
            ])
        
        # Swap values
        for feat1, feat2 in feature_pairs:
            if feat1 in row and feat2 in row:
                row[feat1], row[feat2] = row[feat2], row[feat1]
    
    def prepare_features(self, matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            Tuple of (features, target)
        """
        # Select feature columns
        feature_columns = [
            # H2H features
            'h2h_wins_player1', 'h2h_losses_player1', 'h2h_wins_player2', 'h2h_losses_player2',
            
            # Surface features
            'hard_wins_player1', 'hard_losses_player1', 'hard_wins_player2', 'hard_losses_player2',
            'clay_wins_player1', 'clay_losses_player1', 'clay_wins_player2', 'clay_losses_player2',
            'grass_wins_player1', 'grass_losses_player1', 'grass_wins_player2', 'grass_losses_player2',
            
            # Recent form
            'recent_wins_player1', 'recent_losses_player1', 'recent_wins_player2', 'recent_losses_player2',
            
            # Elo ratings
            'winner_elo', 'loser_elo',
            
            # Ranking features
            'winner_rank', 'loser_rank', 'rank_diff',
            
            # Match features
            'sets_played', 'round_importance'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in matches_df.columns]
        self.feature_columns = available_features
        
        # Create features DataFrame
        features = matches_df[available_features].copy()
        
        # Handle missing values
        features = features.fillna(0)
        
        # Create target
        target = matches_df['player1_wins']
        
        return features, target
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, 
                   model_type: str = 'gradient_boosting') -> Dict[str, Any]:
        """
        Train a prediction model.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            model_type: Type of model to train
            
        Returns:
            Dictionary with model and training info
        """
        print(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model = models[model_type]
        model.fit(X_train_scaled, y_train)
        
        # Calibrate model for better probability estimates
        calibrated_model = CalibratedClassifierCV(model, cv=5)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = calibrated_model.predict(X_test_scaled)
        y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Store results
        self.model = calibrated_model
        self.model_info = {
            'model_type': model_type,
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': self._get_feature_importance(calibrated_model, features.columns),
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_targets': y_test
        }
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model_info
    
    def _get_feature_importance(self, model, feature_names):
        """Get feature importance from the model."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_[0])))
        else:
            return {}
    
    def save_model(self, model_path: str = "models/gradient_boosting.pkl",
                   scaler_path: str = "models/scaler.pkl"):
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save model info
        info_path = model_path.replace('.pkl', '_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(self.model_info, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str = "models/gradient_boosting.pkl",
                   scaler_path: str = "models/scaler.pkl"):
        """Load a trained model and scaler."""
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load model info
        info_path = model_path.replace('.pkl', '_info.pkl')
        with open(info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot feature importance."""
        if not self.model_info.get('feature_importance'):
            print("No feature importance available")
            return
        
        importance = self.model_info['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        if not self.model_info.get('test_probabilities'):
            print("No test probabilities available")
            return
        
        y_test = self.model_info['test_targets']
        y_pred_proba = self.model_info['test_probabilities']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = self.model_info['auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    """Main training function."""
    # Initialize predictor
    predictor = TennisMatchPredictor()
    
    # Load and prepare data
    matches_df = predictor.load_and_prepare_data(start_year=2014)
    
    # Create balanced dataset
    balanced_df = predictor.create_balanced_dataset(matches_df)
    
    # Prepare features
    features, target = predictor.prepare_features(balanced_df)
    
    # Train model
    model_info = predictor.train_model(features, target, model_type='gradient_boosting')
    
    # Save model
    predictor.save_model()
    
    # Plot results
    predictor.plot_feature_importance()
    predictor.plot_roc_curve()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 