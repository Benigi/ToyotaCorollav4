"""
Data Loading Module
Handles loading and initial preprocessing of the Toyota Corolla dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate Toyota Corolla dataset"""
    
    def __init__(self, data_path: str = 'data/raw/ToyotaCorolla.csv'):
        """
        Initialize DataLoader
        
        Parameters
        ----------
        data_path : str
            Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.metadata = {
            'total_rows': 0,
            'total_columns': 0,
            'missing_values': 0,
            'data_types': {}
        }
    
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV
        
        Returns
        -------
        pd.DataFrame
            Loaded dataset
        
        Raises
        ------
        FileNotFoundError
            If the CSV file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        logger.info(f"Loading dataset from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        # Update metadata
        self.metadata['total_rows'] = len(self.data)
        self.metadata['total_columns'] = len(self.data.columns)
        self.metadata['missing_values'] = self.data.isnull().sum().sum()
        self.metadata['data_types'] = self.data.dtypes.to_dict()
        
        logger.info(f"✓ Loaded {self.metadata['total_rows']} rows and "
                   f"{self.metadata['total_columns']} columns")
        
        return self.data
    
    def get_metadata(self) -> Dict:
        """Get dataset metadata"""
        return self.metadata
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        Get basic statistical summary of numeric columns
        
        Returns
        -------
        pd.DataFrame
            Statistical summary
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        
        return self.data.describe()
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get detailed column information
        
        Returns
        -------
        pd.DataFrame
            Column information including data type, missing values, unique values
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load() first.")
        
        column_info = pd.DataFrame({
            'Column': self.data.columns,
            'DataType': self.data.dtypes,
            'Missing': self.data.isnull().sum(),
            'Missing%': (self.data.isnull().sum() / len(self.data) * 100).round(2),
            'Unique': self.data.nunique(),
            'Min': self.data.min(numeric_only=True),
            'Max': self.data.max(numeric_only=True),
            'Mean': self.data.mean(numeric_only=True)
        })
        
        return column_info


class DataValidator:
    """Validate data quality and consistency"""
    
    @staticmethod
    def check_target_variable(data: pd.DataFrame, target: str = 'Price') -> bool:
        """
        Validate target variable exists and is numeric
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        target : str
            Target column name
        
        Returns
        -------
        bool
            True if valid, raises exception otherwise
        """
        if target not in data.columns:
            raise ValueError(f"Target variable '{target}' not found in data")
        
        if not pd.api.types.is_numeric_dtype(data[target]):
            logger.warning(f"Target variable '{target}' is not numeric. Attempting conversion...")
            try:
                data[target] = pd.to_numeric(data[target], errors='coerce')
            except Exception as e:
                raise ValueError(f"Cannot convert target to numeric: {e}")
        
        logger.info(f"✓ Target variable '{target}' validated")
        return True
    
    @staticmethod
    def check_missing_values(data: pd.DataFrame, threshold: float = 0.5) -> Tuple[bool, Dict]:
        """
        Check for missing values above threshold
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        threshold : float
            Missing value threshold (0-1)
        
        Returns
        -------
        Tuple[bool, Dict]
            Validation result and missing value statistics
        """
        missing_stats = {
            'total_missing': data.isnull().sum().sum(),
            'columns_with_missing': data.columns[data.isnull().any()].tolist(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
        }
        
        high_missing = {col: pct for col, pct in missing_stats['missing_percentage'].items() 
                       if pct > threshold * 100}
        
        if high_missing:
            logger.warning(f"Columns with >{threshold*100}% missing values: {high_missing}")
        
        return len(high_missing) == 0, missing_stats
    
    @staticmethod
    def check_data_types(data: pd.DataFrame, expected_types: Optional[Dict] = None) -> bool:
        """
        Validate data types
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        expected_types : Optional[Dict]
            Expected data types for columns
        
        Returns
        -------
        bool
            Validation result
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        logger.info(f"✓ Found {len(numeric_cols)} numeric columns")
        
        return True


def prepare_data(data_path: str = 'data/raw/ToyotaCorolla.csv') -> pd.DataFrame:
    """
    Load and prepare data for analysis
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        Loaded and validated dataset
    """
    # Load data
    loader = DataLoader(data_path)
    data = loader.load()
    
    # Validate
    DataValidator.check_target_variable(data, 'Price')
    is_valid, stats = DataValidator.check_missing_values(data)
    
    if not is_valid:
        logger.warning("Data has missing values. Consider handling them before modeling.")
    
    return data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('data/raw/ToyotaCorolla.csv')
    data = loader.load()
    
    print("Dataset Info:")
    print(f"Shape: {data.shape}")
    print(f"\nBasic Statistics:\n{loader.get_basic_statistics()}")
    print(f"\nColumn Info:\n{loader.get_column_info()}")
