"""
Data Processing Module
Handles data cleaning, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess data"""
    
    # Define categorical mappings
    FUEL_TYPE_MAPPING = {'Diesel': 1, 'Petrol': 0}
    BINARY_MAPPINGS = {
        'Automatic': {1: 'Automatic', 0: 'Manual'},
        'Met_Color': {1: 'Metallic', 0: 'Standard'},
        'ABS': {1: 'Yes', 0: 'No'},
        'Airco': {1: 'Yes', 0: 'No'}
    }
    
    @staticmethod
    def remove_non_numeric_rows(data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Remove rows with non-numeric values in numeric columns
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        \"\"\"
        initial_rows = len(data)
        
        # Identify numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Remove rows with invalid values
        data_clean = data.copy()
        for col in numeric_cols:
            data_clean = data_clean[pd.to_numeric(data_clean[col], errors='coerce').notna()]
        
        removed = initial_rows - len(data_clean)
        logger.info(f"Removed {removed} rows with non-numeric values")
        
        return data_clean
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        \"\"\"
        Handle missing values
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy: 'drop' or 'mean'
        
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        \"\"\"
        initial_rows = len(data)
        
        if strategy == 'drop':
            data_clean = data.dropna()
            removed = initial_rows - len(data_clean)
            logger.info(f"Removed {removed} rows with missing values")
        
        elif strategy == 'mean':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_clean = data.copy()
            for col in numeric_cols:
                data_clean[col].fillna(data_clean[col].mean(), inplace=True)
            logger.info("Filled missing numeric values with mean")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return data_clean
    
    @staticmethod
    def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Encode categorical variables
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        
        Returns
        -------
        pd.DataFrame
            Data with encoded categorical variables
        \"\"\"
        data_encoded = data.copy()
        
        # Encode Fuel_Type
        if 'Fuel_Type' in data_encoded.columns:
            data_encoded['Fuel_Type'] = data_encoded['Fuel_Type'].map(
                DataCleaner.FUEL_TYPE_MAPPING
            )
            logger.info("✓ Encoded Fuel_Type")
        
        # Drop Model column (too many unique values)
        if 'Model' in data_encoded.columns:
            data_encoded = data_encoded.drop('Model', axis=1)
            logger.info("✓ Dropped Model column")
        
        return data_encoded
    
    @staticmethod
    def clean(data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Apply complete cleaning pipeline
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        \"\"\"
        logger.info("Starting data cleaning pipeline...")
        
        # Step 1: Remove non-numeric values
        data = DataCleaner.remove_non_numeric_rows(data)
        
        # Step 2: Encode categorical variables
        data = DataCleaner.encode_categorical(data)
        
        # Step 3: Handle missing values
        data = DataCleaner.handle_missing_values(data, strategy='drop')
        
        # Step 4: Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        removed = initial_rows - len(data)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        logger.info(f"✓ Data cleaning completed. Final shape: {data.shape}")
        
        return data


class DataNormalizer:
    \"\"\"Normalize and scale data\"\"\"
    
    def __init__(self, scaler_type: str = 'minmax'):
        \"\"\"
        Initialize DataNormalizer
        
        Parameters
        ----------
        scaler_type : str
            Type of scaler: 'minmax' or 'standard'
        \"\"\"
        self.scaler_type = scaler_type
        
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'DataNormalizer':
        \"\"\"
        Fit the scaler
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        
        Returns
        -------
        DataNormalizer
            Self for chaining
        \"\"\"
        self.scaler.fit(X)
        self.fitted = True
        logger.info(f"✓ {self.scaler_type.upper()} scaler fitted on {X.shape} data")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Transform data using fitted scaler
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to transform
        
        Returns
        -------
        pd.DataFrame
            Normalized data
        \"\"\"
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Fit and transform in one step
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to fit and transform
        
        Returns
        -------
        pd.DataFrame
            Normalized data
        \"\"\"
        self.fit(X)
        return self.transform(X)


def prepare_features_and_target(data: pd.DataFrame, 
                                target: str = 'Price') -> Tuple[pd.DataFrame, pd.Series]:
    \"\"\"
    Separate features and target variable
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    target : str
        Target column name
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features and target
    \"\"\"
    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' not found")
    
    X = data.drop(target, axis=1)
    y = data[target]
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    return X, y


def process_data(data: pd.DataFrame, 
                 normalize: bool = True,
                 target: str = 'Price') -> Tuple[pd.DataFrame, pd.Series, DataNormalizer]:
    \"\"\"
    Complete data processing pipeline
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    normalize : bool
        Whether to normalize features
    target : str
        Target column name
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, DataNormalizer]
        Processed features, target, and scaler
    \"\"\"
    # Clean data
    data_clean = DataCleaner.clean(data)
    
    # Separate features and target
    X, y = prepare_features_and_target(data_clean, target)
    
    # Normalize if requested
    if normalize:
        normalizer = DataNormalizer(scaler_type='minmax')
        X_normalized = normalizer.fit_transform(X)
        logger.info("✓ Data normalization completed")
    else:
        X_normalized = X
        normalizer = None
    
    return X_normalized, y, normalizer


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader('data/raw/ToyotaCorolla.csv')
    data = loader.load()
    
    # Process
    X, y, scaler = process_data(data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\\nFeatures sample:\\n{X.head()}")
    print(f"\\nTarget sample:\\n{y.head()}")
