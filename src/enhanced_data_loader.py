"""
Enhanced data loading system with mock data support for development environment.
Extends the existing EventDataLoader with configuration-driven data sources and mock data generation.
"""

import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import warnings
from pathlib import Path
import gc

from .config import get_config, ModelConfig
from .logging_config import get_logger, log_data_load, LogContext
from .event_processor import EventDataLoader  # Import existing loader


class MockDataGenerator:
    """Generates synthetic event and stock data for development and testing."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = get_logger(__name__, {'component': 'MockDataGenerator'})
        np.random.seed(42)  # For reproducible mock data
    
    def generate_mock_events(self, n_events: int = None, event_types: List[str] = None) -> pl.DataFrame:
        """Generate mock event data."""
        n_events = n_events or self.config.data.mock_events
        event_types = event_types or ['FDA', 'earnings']
        
        self.logger.info(f"Generating {n_events} mock events")
        
        # Generate random dates over the past 2 years
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now() - timedelta(days=30)
        
        # Generate mock tickers (realistic looking)
        ticker_prefixes = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ORCL']
        ticker_suffixes = ['', 'A', 'B', 'C', 'X', 'Y', 'Z']
        
        mock_tickers = []
        for prefix in ticker_prefixes:
            for suffix in ticker_suffixes:
                mock_tickers.append(f"{prefix}{suffix}")
                if len(mock_tickers) >= self.config.data.mock_assets:
                    break
            if len(mock_tickers) >= self.config.data.mock_assets:
                break
        
        # Generate events
        events_data = []
        for i in range(n_events):
            event_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            ticker = np.random.choice(mock_tickers)
            event_type = np.random.choice(event_types)
            
            # Add outcome for some events (positive/negative/neutral)
            outcome = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
            
            events_data.append({
                'ticker': ticker,
                'Event Date': event_date,
                'event_type': event_type,
                'outcome': outcome,
                'event_id': f"{ticker}_{event_date.strftime('%Y%m%d')}_{event_type}"
            })
        
        events_df = pl.DataFrame(events_data)
        self.logger.info(f"Generated {len(events_df)} mock events")
        return events_df
    
    def generate_mock_stock_data(self, tickers: List[str], start_date: datetime, 
                                end_date: datetime) -> pl.DataFrame:
        """Generate mock stock price data with realistic patterns."""
        self.logger.info(f"Generating mock stock data for {len(tickers)} tickers")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = [d for d in date_range if d.weekday() < 5]  # Only weekdays
        
        stock_data = []
        
        for ticker in tickers:
            # Set random seed based on ticker for consistency
            ticker_seed = hash(ticker) % 2**32
            np.random.seed(ticker_seed)
            
            # Initial price
            initial_price = np.random.uniform(20, 500)
            
            # Generate price series with realistic volatility
            n_days = len(date_range)
            returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
            
            # Add some autocorrelation to returns
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            # Calculate prices
            prices = [initial_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate volume data
            base_volume = np.random.uniform(100000, 10000000)
            volumes = np.random.lognormal(np.log(base_volume), 0.5, n_days)
            
            # Create stock data for this ticker
            for i, date in enumerate(date_range):
                if i < len(prices):
                    price = prices[i]
                    volume = volumes[i]
                    ret = returns[i] if i > 0 else 0.0
                    
                    # Generate OHLC data
                    daily_vol = abs(ret) * 0.5
                    high = price * (1 + np.random.uniform(0, daily_vol))
                    low = price * (1 - np.random.uniform(0, daily_vol))
                    open_price = price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))
                    
                    stock_data.append({
                        'date': date,
                        'ticker': ticker,
                        'prc': price,
                        'ret': ret,
                        'vol': volume,
                        'openprc': open_price,
                        'askhi': high,
                        'bidlo': low
                    })
        
        stock_df = pl.DataFrame(stock_data)
        self.logger.info(f"Generated {len(stock_df)} mock stock records")
        return stock_df
    
    def generate_mock_dataset(self) -> pl.DataFrame:
        """Generate complete mock dataset with events and stock data."""
        with LogContext(self.logger, operation='generate_mock_dataset'):
            # Generate events
            events_df = self.generate_mock_events()
            
            # Get unique tickers and date range
            tickers = events_df['ticker'].unique().to_list()
            min_date = events_df['Event Date'].min()
            max_date = events_df['Event Date'].max()
            
            # Extend date range for window analysis
            buffer_days = self.config.data.window_days + 5
            start_date = min_date - timedelta(days=buffer_days)
            end_date = max_date + timedelta(days=buffer_days)
            
            # Generate stock data
            stock_df = self.generate_mock_stock_data(tickers, start_date, end_date)
            
            # Merge events with stock data (similar to existing loader logic)
            merged_df = stock_df.join(
                events_df.select(['ticker', 'Event Date']), 
                on='ticker', 
                how='inner'
            )
            
            # Calculate days to event
            merged_df = merged_df.with_columns(
                (pl.col('date') - pl.col('Event Date')).dt.total_days().cast(pl.Int32).alias('days_to_event')
            )
            
            # Filter to analysis window
            window_days = self.config.data.window_days
            filtered_df = merged_df.filter(
                (pl.col('days_to_event') >= -window_days) &
                (pl.col('days_to_event') <= window_days)
            )
            
            # Add additional columns
            filtered_df = filtered_df.with_columns([
                (pl.col('days_to_event') == 0).cast(pl.Int8).alias('is_event_date'),
                (pl.col("ticker") + "_" + pl.col("Event Date").dt.strftime('%Y%m%d')).alias('event_id')
            ])
            
            self.logger.info(f"Generated complete mock dataset with {len(filtered_df)} records")
            return filtered_df


class EnhancedEventDataLoader(EventDataLoader):
    """Enhanced version of EventDataLoader with configuration support and mock data."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__, {'component': 'EnhancedEventDataLoader'})
        self.mock_generator = MockDataGenerator(self.config)
        self._production_initialized = False
        
        # Initialize parent class with config values if not using mock data
        if not self.config.data.use_mock_data:
            try:
                # Determine event file and columns based on configuration
                event_path, event_date_col, ticker_col = self._get_event_file_config()
                
                super().__init__(
                    event_path=event_path,
                    stock_paths=self.config.data.stock_files,
                    window_days=self.config.data.window_days,
                    event_date_col=event_date_col,
                    ticker_col=ticker_col
                )
                self._production_initialized = True
            except FileNotFoundError as e:
                self.logger.warning(f"Production files not found: {e}. Will fallback to mock data.")
                # Initialize with dummy values for fallback mode
                super().__init__(
                    event_path="dummy",
                    stock_paths=["dummy"],
                    window_days=self.config.data.window_days
                )
        else:
            # Initialize with dummy values for mock mode
            super().__init__(
                event_path="dummy",
                stock_paths=["dummy"],
                window_days=self.config.data.window_days
            )
    
    def _get_event_file_config(self) -> Tuple[str, str, str]:
        """Determine which event file to use based on configuration."""
        if self.config.data.fda_events and os.path.exists(self.config.data.fda_events):
            return self.config.data.fda_events, "Approval Date", "ticker"
        elif self.config.data.earnings_events and os.path.exists(self.config.data.earnings_events):
            return self.config.data.earnings_events, "ANNDATS", "ticker"
        else:
            raise FileNotFoundError("No valid event files found in configuration")
    
    def load_data(self) -> Optional[pl.DataFrame]:
        """Load data with configuration-driven source selection."""
        start_time = datetime.now()
        
        try:
            if self.config.data.use_mock_data:
                self.logger.info("Loading mock data for development environment")
                data = self._load_mock_data()
            else:
                self.logger.info("Loading production data")
                data = self._load_production_data()
            
            if data is not None:
                duration = (datetime.now() - start_time).total_seconds()
                log_data_load(self.logger, "event_stock_data", len(data), duration)
                
                # Log data quality metrics
                self._log_data_quality(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}", exc_info=True)
            
            # Fallback to mock data if production data fails
            if not self.config.data.use_mock_data:
                self.logger.warning("Falling back to mock data due to production data error")
                return self._load_mock_data()
            
            return None
    
    def _load_mock_data(self) -> pl.DataFrame:
        """Load mock data for development."""
        with LogContext(self.logger, data_source='mock'):
            return self.mock_generator.generate_mock_dataset()
    
    def _load_production_data(self) -> Optional[pl.DataFrame]:
        """Load production data using parent class method."""
        with LogContext(self.logger, data_source='production'):
            # Check if production was properly initialized
            if not self._production_initialized:
                self.logger.warning("Production data not properly initialized, falling back to mock data")
                return self._load_mock_data()
            
            # Validate that required files exist
            try:
                self._validate_production_files()
                # Use parent class load_data method
                return super().load_data()
            except (FileNotFoundError, ValueError) as e:
                self.logger.warning(f"Production data validation failed: {e}. Falling back to mock data.")
                return self._load_mock_data()
    
    def _validate_production_files(self):
        """Validate that production data files exist."""
        issues = []
        
        # Check stock files (CRSP parquet files)
        for stock_file in self.config.data.stock_files:
            if not os.path.exists(stock_file):
                issues.append(f"CRSP stock file not found: {stock_file}")
            elif not stock_file.endswith('.parquet'):
                self.logger.warning(f"Stock file {stock_file} is not a parquet file. CRSP files should be in parquet format.")
        
        # Check event files
        if self.config.data.fda_events and not os.path.exists(self.config.data.fda_events):
            issues.append(f"FDA events file not found: {self.config.data.fda_events}")
        
        if self.config.data.earnings_events and not os.path.exists(self.config.data.earnings_events):
            issues.append(f"Earnings events file not found: {self.config.data.earnings_events}")
        
        if issues:
            self.logger.warning(f"Production file validation issues: {issues}")
            if not self.config.data.fda_events and not self.config.data.earnings_events:
                raise FileNotFoundError("No event files configured or found")
    
    def _log_data_quality(self, data: pl.DataFrame):
        """Log data quality metrics."""
        if data is None or data.is_empty():
            self.logger.warning("Data quality check: Dataset is empty")
            return
        
        # Calculate quality metrics
        total_records = len(data)
        unique_events = data['event_id'].n_unique() if 'event_id' in data.columns else 0
        unique_tickers = data['ticker'].n_unique() if 'ticker' in data.columns else 0
        date_range = None
        
        if 'date' in data.columns:
            min_date = data['date'].min()
            max_date = data['date'].max()
            date_range = f"{min_date} to {max_date}"
        
        # Check for missing values
        missing_data = {}
        for col in data.columns:
            null_count = data[col].null_count()
            if null_count > 0:
                missing_data[col] = {
                    'null_count': null_count,
                    'null_percentage': (null_count / total_records) * 100
                }
        
        quality_metrics = {
            'total_records': total_records,
            'unique_events': unique_events,
            'unique_tickers': unique_tickers,
            'date_range': date_range,
            'missing_data': missing_data,
            'columns': data.columns
        }
        
        self.logger.log_data_quality('event_stock_data', quality_metrics)
        
        # Log warnings for data quality issues
        if missing_data:
            high_missing_cols = [col for col, info in missing_data.items() 
                               if info['null_percentage'] > 10]
            if high_missing_cols:
                self.logger.warning(f"Columns with >10% missing data: {high_missing_cols}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data."""
        try:
            data = self.load_data()
            if data is None:
                return {'status': 'error', 'message': 'No data available'}
            
            summary = {
                'status': 'success',
                'data_source': 'mock' if self.config.data.use_mock_data else 'production',
                'total_records': len(data),
                'unique_events': data['event_id'].n_unique() if 'event_id' in data.columns else 0,
                'unique_tickers': data['ticker'].n_unique() if 'ticker' in data.columns else 0,
                'columns': data.columns,
                'memory_usage_mb': data.estimated_size("mb")
            }
            
            if 'date' in data.columns:
                summary['date_range'] = {
                    'start': str(data['date'].min()),
                    'end': str(data['date'].max())
                }
            
            return summary
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


def create_data_loader(config: Optional[ModelConfig] = None) -> EnhancedEventDataLoader:
    """Factory function to create enhanced data loader."""
    return EnhancedEventDataLoader(config)


# Convenience functions for common data loading patterns
def load_event_data(config: Optional[ModelConfig] = None) -> Optional[pl.DataFrame]:
    """Load event data using configuration."""
    loader = create_data_loader(config)
    return loader.load_data()

def validate_data_files(config: Optional[ModelConfig] = None) -> List[str]:
    """Validate that configured data files exist."""
    config = config or get_config()
    issues = []
    
    if not config.data.use_mock_data:
        # Check stock files (CRSP parquet files)
        for stock_file in config.data.stock_files:
            if not os.path.exists(stock_file):
                issues.append(f"CRSP stock file not found: {stock_file}")
            elif not stock_file.endswith('.parquet'):
                issues.append(f"Stock file {stock_file} should be in parquet format for CRSP data")
        
        # Check event files
        if config.data.fda_events and not os.path.exists(config.data.fda_events):
            issues.append(f"FDA events file not found: {config.data.fda_events}")
        
        if config.data.earnings_events and not os.path.exists(config.data.earnings_events):
            issues.append(f"Earnings events file not found: {config.data.earnings_events}")
    
    return issues


def create_result_directories(base_path: str = "results") -> Dict[str, str]:
    """Create organized result directories for FDA and earnings analysis."""
    directories = {
        'base': base_path,
        'fda': os.path.join(base_path, 'fda_analysis'),
        'earnings': os.path.join(base_path, 'earnings_analysis'),
        'combined': os.path.join(base_path, 'combined_analysis')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def get_result_file_prefix(event_type: str, analysis_date: Optional[datetime] = None) -> str:
    """Get appropriate file prefix for result files based on event type."""
    if analysis_date is None:
        analysis_date = datetime.now()
    
    date_str = analysis_date.strftime('%Y%m%d')
    
    if event_type.lower() == 'fda':
        return f"fda_analysis_{date_str}"
    elif event_type.lower() == 'earnings':
        return f"earnings_analysis_{date_str}"
    else:
        return f"analysis_{date_str}"


def save_analysis_results(data: pl.DataFrame, event_type: str, 
                         base_path: str = "results", 
                         file_format: str = "parquet") -> str:
    """Save analysis results with proper organization and file prefixes."""
    directories = create_result_directories(base_path)
    
    # Determine target directory
    if event_type.lower() == 'fda':
        target_dir = directories['fda']
    elif event_type.lower() == 'earnings':
        target_dir = directories['earnings']
    else:
        target_dir = directories['combined']
    
    # Generate filename with prefix
    file_prefix = get_result_file_prefix(event_type)
    filename = f"{file_prefix}.{file_format}"
    file_path = os.path.join(target_dir, filename)
    
    # Save data
    if file_format.lower() == 'parquet':
        data.write_parquet(file_path)
    elif file_format.lower() == 'csv':
        data.write_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return file_path