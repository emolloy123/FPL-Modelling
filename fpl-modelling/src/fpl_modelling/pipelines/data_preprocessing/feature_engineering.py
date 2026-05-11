import pandas as pd
from typing import Dict, Optional
from abc import ABC, abstractmethod


class RollingFeatureEngineer(ABC):
    """Base class for rolling feature engineering strategies."""
    
    @abstractmethod
    def transform(
        self, 
        series: pd.Series, 
        grouper: pd.Series, 
        n: int, 
        min_periods: int
    ) -> pd.Series:
        """Apply the rolling transformation."""
        pass
    
    @abstractmethod
    def get_suffix(self, n: int) -> str:
        """Get the feature name suffix."""
        pass


class EWMEngineer(RollingFeatureEngineer):
    """Exponential weighted mean feature engineer."""
    
    def transform(
        self, 
        series: pd.Series, 
        grouper: pd.Series, 
        n: int, 
        min_periods: int
    ) -> pd.Series:
        return (
            series.groupby(grouper, sort=False)
            .ewm(span=n, adjust=False, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    def get_suffix(self, n: int) -> str:
        return f"ewm_{n}"


class RollingWindowEngineer(RollingFeatureEngineer):
    """Rolling window mean feature engineer."""
    
    def transform(
        self, 
        series: pd.Series, 
        grouper: pd.Series, 
        n: int, 
        min_periods: int
    ) -> pd.Series:
        return (
            series.groupby(grouper, sort=False)
            .rolling(window=n, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    def get_suffix(self, n: int) -> str:
        return f"roll_{n}"


class FeatureEngineeringPipeline:
    """Pipeline for creating rolling average features."""
    
    _ENGINEERS = {
        'ewm': EWMEngineer(),
        'rolling': RollingWindowEngineer(),
    }
    
    def __init__(
        self,
        player_col: str = "player_id",
        time_col: str = "gameweek",
        rolling_config: Optional[Dict] = None
    ):
        self.player_col = player_col
        self.time_col = time_col
        self.rolling_config = rolling_config or {
            'transfers_in': {'method': 'ewm', 'n': 3},
            'ict_index': {'method': 'ewm', 'n': 5},
        }
    
    @classmethod
    def register_engineer(cls, name: str, engineer: RollingFeatureEngineer):
        """Register a new feature engineering strategy."""
        cls._ENGINEERS[name] = engineer
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required_cols = {self.player_col, self.time_col} | set(self.rolling_config.keys())
        if missing := required_cols - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort and copy dataframe."""
        return df.copy().sort_values([self.player_col, self.time_col])
    
    def _engineer_feature(
        self, 
        df: pd.DataFrame, 
        col: str, 
        cfg: Dict
    ) -> pd.Series:
        """Engineer a single feature using the configured method."""
        method = cfg.get("method", "rolling")
        n = cfg["n"]
        min_periods = cfg.get("min_periods", 1)
        
        engineer = self._ENGINEERS.get(method)
        if not engineer:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(self._ENGINEERS.keys())}"
            )
        
        # Shift to prevent leakage
        shifted = df.groupby(self.player_col, sort=False)[col].shift(1)
        
        # Apply transformation
        return engineer.transform(shifted, df[self.player_col], n, min_periods)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all rolling average features."""
        self._validate_dataframe(df)
        df = self._prepare_dataframe(df)
        
        # Engineer all features
        for col, cfg in self.rolling_config.items():
            method = cfg.get("method", "rolling")
            n = cfg["n"]
            engineer = self._ENGINEERS[method]
            
            new_col = f"{col}_{engineer.get_suffix(n)}"
            df[new_col] = self._engineer_feature(df, col, cfg)
        
        return df

