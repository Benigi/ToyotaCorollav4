"""
Model Training Module
Handles model training, evaluation, and saving
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    \"\"\"Train and manage Random Forest regression model\"\"\"
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 20, 
                 min_samples_split: int = 5, random_state: int = 42):
        \"\"\"
        Initialize ModelTrainer
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split a node
        random_state : int
            Random seed for reproducibility
        \"\"\"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.model = None
        self.training_history = {}
        self.cv_scores = None
    
    def build_model(self) -> RandomForestRegressor:
        \"\"\"
        Build the Random Forest model
        
        Returns
        -------
        RandomForestRegressor
            Initialized model
        \"\"\"
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        logger.info(f\"✓ Model built with {self.n_estimators} trees\")
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        \"\"\"
        Train the model
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        
        Returns
        -------
        Dict
            Training information
        \"\"\"
        if self.model is None:
            self.build_model()
        
        logger.info(f\"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features\")
        
        self.model.fit(X_train, y_train)
        
        # Get training score
        y_train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        self.training_history = {
            'samples': X_train.shape[0],
            'features': X_train.shape[1],
            'train_r2': train_r2,
            'train_rmse': train_rmse
        }
        
        logger.info(f\"✓ Training completed. R² = {train_r2:.4f}, RMSE = €{train_rmse:.2f}\")
        
        return self.training_history
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        \"\"\"
        Evaluate model on test set
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        
        Returns
        -------
        Dict
            Evaluation metrics
        \"\"\"
        if self.model is None:
            raise RuntimeError(\"Model not trained. Call train() first.\")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f\"✓ Evaluation completed. R² = {metrics['r2']:.4f}, RMSE = €{metrics['rmse']:.2f}\")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        \"\"\"
        Perform cross-validation
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int
            Number of folds
        
        Returns
        -------
        np.ndarray
            Cross-validation scores
        \"\"\"
        if self.model is None:
            self.build_model()
        
        logger.info(f\"Performing {cv}-fold cross-validation...\")
        
        self.cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring='r2', n_jobs=-1
        )
        
        logger.info(f\"✓ CV scores: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})\")
        
        return self.cv_scores
    
    def get_feature_importance(self) -> pd.DataFrame:
        \"\"\"
        Get feature importance scores
        
        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        \"\"\"
        if self.model is None:
            raise RuntimeError(\"Model not trained.\")
        
        importance_df = pd.DataFrame({
            'Feature': self.model.feature_names_in_,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        \"\"\"
        Make predictions
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        
        Returns
        -------
        np.ndarray
            Predictions
        \"\"\"
        if self.model is None:
            raise RuntimeError(\"Model not trained.\")
        
        return self.model.predict(X)
    
    def save(self, path: str = 'data/models/rf_model.pkl'):
        \"\"\"
        Save model to disk
        
        Parameters
        ----------
        path : str
            Path to save model
        \"\"\"
        if self.model is None:
            raise RuntimeError(\"Model not trained.\")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f\"✓ Model saved to {path}\")
    
    def load(self, path: str = 'data/models/rf_model.pkl'):
        \"\"\"
        Load model from disk
        
        Parameters
        ----------
        path : str
            Path to model file
        \"\"\"
        self.model = joblib.load(path)
        logger.info(f\"✓ Model loaded from {path}\")


class ModelEvaluator:
    \"\"\"Comprehensive model evaluation\"\"\"
    
    @staticmethod
    def generate_report(y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        \"\"\"
        Generate detailed evaluation report
        
        Parameters
        ----------
        y_true : pd.Series
            Actual values
        y_pred : np.ndarray
            Predicted values
        
        Returns
        -------
        Dict
            Comprehensive evaluation metrics
        \"\"\"
        residuals = y_true - y_pred
        
        report = {
            'regression_metrics': {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                'mape_median': np.median(np.abs((y_true - y_pred) / y_true)) * 100
            },
            'residual_statistics': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max(),
                'q25': residuals.quantile(0.25),
                'q75': residuals.quantile(0.75)
            },
            'prediction_range': {
                'min': y_pred.min(),
                'max': y_pred.max(),
                'mean': y_pred.mean(),
                'std': y_pred.std()
            }
        }
        
        return report


def train_model(X: pd.DataFrame, y: pd.Series, 
                test_size: float = 0.2) -> Tuple[ModelTrainer, Dict, Dict]:
    \"\"\"
    Complete model training pipeline
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Test set proportion
    
    Returns
    -------
    Tuple[ModelTrainer, Dict, Dict]
        Trained model, training info, evaluation metrics
    \"\"\"
    logger.info(\"Starting model training pipeline...\")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f\"Train/test split: {len(X_train)}/{len(X_test)}\")
    
    # Initialize and train
    trainer = ModelTrainer(n_estimators=200, max_depth=20)
    train_info = trainer.train(X_train, y_train)
    
    # Evaluate
    eval_metrics = trainer.evaluate(X_test, y_test)
    
    # Cross-validation
    cv_scores = trainer.cross_validate(X, y, cv=5)
    eval_metrics['cv_mean'] = cv_scores.mean()
    eval_metrics['cv_std'] = cv_scores.std()
    
    logger.info(\"✓ Model training pipeline completed\")
    
    return trainer, train_info, eval_metrics


if __name__ == \"__main__\":
    # Example usage
    from data_loader import DataLoader
    from data_processor import process_data
    
    # Load and process data
    loader = DataLoader('data/raw/ToyotaCorolla.csv')
    data = loader.load()
    X, y, scaler = process_data(data)
    
    # Train model
    trainer, train_info, eval_metrics = train_model(X, y)
    
    print(\"Training Info:\", train_info)
    print(\"\\nEvaluation Metrics:\", eval_metrics)
    print(\"\\nFeature Importance:\\n\", trainer.get_feature_importance())
