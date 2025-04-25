import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.impute import SimpleImputer

class EventDataLoader:
    def __init__(self, event_path, stock_paths, window_days=100, event_date_col='Event Date', 
                 event_type_col=None, event_name_col=None, event_value_col=None):
        """
        Initialize EventDataLoader with event data path and a list of stock data paths.
        
        Parameters:
        event_path (str): Path to event data CSV (e.g., FDA approvals, earnings announcements)
        stock_paths (list): List of paths to stock data CSV files
        window_days (int): Number of days to look before and after event date
        event_date_col (str): Name of the column containing event dates
        event_type_col (str, optional): Name of the column containing event types (e.g., 'FDA Approval', 'Earnings')
        event_name_col (str, optional): Name of the column containing event details (e.g., drug names)
        event_value_col (str, optional): Name of the column containing numeric event values (e.g., EPS surprise)
        """
        self.event_path = event_path
        self.stock_paths = stock_paths if isinstance(stock_paths, list) else [stock_paths]
        self.window_days = window_days
        self.event_date_col = event_date_col
        self.event_type_col = event_type_col
        self.event_name_col = event_name_col
        self.event_value_col = event_value_col
        
    def _load_single_stock_file(self, stock_path):
        """Load and process a single stock data file"""
        try:
            # Read just the header row to check column names
            headers = pd.read_csv(stock_path, nrows=0).columns
            
            # Check for various possible date column names (case-insensitive)
            date_col = None
            for col in headers:
                if col.lower() in ['date', 'trade_date', 'trading_date', 'tradedate', 'dt']:
                    date_col = col
                    break
            
            # If we found a date column, load with parse_dates
            if date_col:
                stock_data = pd.read_csv(stock_path, parse_dates=[date_col])
                # Rename the date column to 'date' for consistency
                stock_data = stock_data.rename(columns={date_col: 'date'})
            else:
                # No obvious date column found, load without parsing dates
                stock_data = pd.read_csv(stock_path)
                print(f"Warning: No date column identified in stock data file {stock_path}. Please check the file format.")
                
            # Rename sym_root to ticker for consistency
            if 'sym_root' in stock_data.columns and 'ticker' not in stock_data.columns:
                stock_data = stock_data.rename(columns={'sym_root': 'ticker'})
                
            # Ensure a date column exists
            if 'date' not in stock_data.columns:
                # Check for other possible date columns
                date_columns = [col for col in stock_data.columns if 
                               col.lower() in ['date', 'trade_date', 'trading_date', 'tradedate', 'dt', 'time', 'timestamp', 'date_time']]
                
                if date_columns:
                    # Use the first matching date column
                    original_date_col = date_columns[0]
                    stock_data = stock_data.rename(columns={original_date_col: 'date'})
                    print(f"Renamed column '{original_date_col}' to 'date'")
                else:
                    raise ValueError(f"No date column found in stock data file {stock_path}. Please ensure your CSV has a date column.")
            
            # Ensure date column is datetime format
            if not pd.api.types.is_datetime64_any_dtype(stock_data['date']):
                stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
                # Check if conversion was successful
                if stock_data['date'].isna().all():
                    raise ValueError(f"Could not convert date column to datetime format in file {stock_path}. Check the date values in your CSV.")
                
            return stock_data
            
        except Exception as e:
            print(f"Error loading stock data from {stock_path}: {e}")
            raise ValueError(f"Failed to load stock data from {stock_path}. Please check the file exists and is properly formatted.")
    
    def load_data(self):
        """Load event data and stock data, then merge them"""
        # Load event data
        event_data = pd.read_csv(self.event_path, parse_dates=[self.event_date_col])
            
        # Ensure event date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(event_data[self.event_date_col]):
            event_data[self.event_date_col] = pd.to_datetime(event_data[self.event_date_col])
            
        # Extract ticker from event data if needed
        if 'ticker' not in event_data.columns and 'Ticker' in event_data.columns:
            event_data['ticker'] = event_data['Ticker'].apply(
                lambda x: x.split()[-1].strip() if isinstance(x, str) and ' ' in x else x
            )
        
        # Get unique ticker and event date combinations with additional event info
        event_cols = ['ticker', self.event_date_col]
        if self.event_type_col and self.event_type_col in event_data.columns:
            event_cols.append(self.event_type_col)
        if self.event_name_col and self.event_name_col in event_data.columns:
            event_cols.append(self.event_name_col)
        if self.event_value_col and self.event_value_col in event_data.columns:
            event_cols.append(self.event_value_col)
            
        event_instances = event_data[event_cols].drop_duplicates()
        print(f"Found {len(event_instances)} unique events")
        
        # Load and merge all stock data files
        stock_data_list = []
        for stock_path in self.stock_paths:
            stock_data = self._load_single_stock_file(stock_path)
            stock_data_list.append(stock_data)
            
        # Concatenate all stock data files
        if not stock_data_list:
            raise ValueError("No stock data files were successfully loaded")
            
        stock_data = pd.concat(stock_data_list, ignore_index=True)
        
        # Remove duplicates in stock data
        stock_data = stock_data.drop_duplicates(subset=['date', 'ticker'])
        
        # Filter stock data to only include tickers with events
        tickers_with_events = event_instances['ticker'].unique()
        stock_data_filtered = stock_data[stock_data['ticker'].isin(tickers_with_events)]
        print(f"Filtered stock data to {len(tickers_with_events)} tickers with events")
        
        # Create separate dataframes for each event
        event_dataframes = []
        events_with_data = 0
        
        for _, event in event_instances.iterrows():
            ticker = event['ticker']
            event_date = event[self.event_date_col]
            
            # Define date range around event
            start_date = event_date - pd.Timedelta(days=self.window_days)
            end_date = event_date + pd.Timedelta(days=self.window_days)
            
            # Filter stock data for this ticker and date range
            event_window_data = stock_data_filtered[
                (stock_data_filtered['ticker'] == ticker) &
                (stock_data_filtered['date'] >= start_date) &
                (stock_data_filtered['date'] <= end_date)
            ].copy()
            
            # Skip if no data available
            if len(event_window_data) == 0:
                continue
                
            # Add event information
            event_window_data[self.event_date_col] = event_date
            if self.event_type_col and self.event_type_col in event.index:
                event_window_data[self.event_type_col] = event[self.event_type_col]
            if self.event_name_col and self.event_name_col in event.index:
                event_window_data[self.event_name_col] = event[self.event_name_col]
            if self.event_value_col and self.event_value_col in event.index:
                event_window_data[self.event_value_col] = event[self.event_value_col]
            event_window_data['is_event_date'] = (event_window_data['date'] == event_date)
            
            event_dataframes.append(event_window_data)
            events_with_data += 1
        
        # Combine all event dataframes
        if not event_dataframes:
            raise ValueError("No matching data found for any events")
            
        combined_data = pd.concat(event_dataframes, ignore_index=True)
        print(f"Created data for {events_with_data} events with available stock data")
        
        return combined_data

class EventFeatureEngineer:
    def __init__(self, prediction_window=5, event_date_col='Event Date', event_value_col=None):
        """
        Initialize EventFeatureEngineer.
        
        Parameters:
        prediction_window (int): Number of days to look ahead for prediction target
        event_date_col (str): Name of the column containing event dates
        event_value_col (str, optional): Name of the column containing numeric event values
        """
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        self.event_date_col = event_date_col
        self.event_value_col = event_value_col
        
    def create_target(self, df, price_col='prc'):
        """
        Create the target variable: future returns over prediction_window days
        """
        df = df.copy()
        
        # Calculate future returns for each ticker and event separately
        df['future_ret'] = df.groupby(['ticker', self.event_date_col])[price_col].transform(
            lambda x: x.pct_change(self.prediction_window).shift(-self.prediction_window)
        )
        
        return df
    
    def calculate_features(self, df):
        """
        Calculate features for event analysis
        """
        df = df.copy()
        
        # Price momentum over different windows
        for window in self.windows:
            df[f'momentum_{window}'] = df.groupby(['ticker', self.event_date_col])['prc'].transform(
                lambda x: (x - x.shift(window)) / x.shift(window)
            )
        
        # Momentum differences
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']
        
        # Return volatility over different windows
        for window in self.windows:
            df[f'volatility_{window}'] = df.groupby(['ticker', self.event_date_col])['ret'].transform(
                lambda x: x.rolling(window, min_periods=max(2, window//5)).std()
            )
        
        # Volatility differences
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']
        
        # Log returns
        df['log_ret'] = df.groupby(['ticker', self.event_date_col])['prc'].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )
        
        # Days relative to event date
        df['days_to_event'] = (df['date'] - df[self.event_date_col]).dt.days
        
        # Lagged returns (1, 2, 3 days)
        for lag in range(1, 4):
            df[f'ret_lag_{lag}'] = df.groupby(['ticker', self.event_date_col])['ret'].shift(lag)
        
        # Pre-event returns (cumulative 30-day)
        pre_event_rets = {}
        for (ticker, event_date), group in df.groupby(['ticker', self.event_date_col]):
            pre_event_data = group[
                (group['date'] <= event_date) & 
                (group['date'] > event_date - pd.Timedelta(days=30))
            ]
            if not pre_event_data.empty:
                pre_event_ret = pre_event_data['ret'].sum()
                pre_event_rets[(ticker, event_date)] = pre_event_ret
        
        # Apply pre-event returns to the dataframe
        df['pre_event_ret'] = df.apply(
            lambda row: pre_event_rets.get((row['ticker'], row[self.event_date_col]), np.nan) 
            if row['days_to_event'] >= 0 else np.nan,
            axis=1
        )
        
        # Add event-specific features if event value is provided
        if self.event_value_col and self.event_value_col in df.columns:
            # Event value relative to historical values
            df['event_value_zscore'] = df.groupby('ticker')[self.event_value_col].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
            # Event value momentum (change from last event)
            df['event_value_mom'] = df.groupby('ticker')[self.event_value_col].transform(
                lambda x: x.pct_change()
            )
        
        # Fill NaN values with group medians for each feature
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_features:
            if col != 'future_ret':  # Don't fill target variable
                df[col] = df.groupby(['ticker', self.event_date_col])[col].transform(
                    lambda x: x.fillna(x.median()) if x.isna().any() else x
                )
        
        # Add previous day volatility feature
        df['prev_day_volatility'] = df.groupby(['ticker', self.event_date_col])['ret'].transform(
            lambda x: x.shift(1).abs()  # Simple measure of previous day volatility
        )
            
        # Add previous day volatility as standard deviation
        df['prev_day_vol_std'] = df.groupby(['ticker', self.event_date_col])['ret'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).std()
        )

        return df
    
    def get_features_target(self, df):
        """
        Extract feature matrix X and target vector y, handling missing values
        """
        df = df.copy()
        
        # Identify feature columns (exclude metadata and target)
        exclude_cols = [
            # Date-related columns
            'date', self.event_date_col, 'days_to_event',
            # Identifier columns
            'ticker', 'permno', 'permco', 'issuno', 'hexcd', 'hsiccd', 
            'shrout', 'cfacpr', 'cfacshr', 'shrcd', 'exchcd', 'siccd', 'ncusip',
            'comnam', 'shrcls', 'tsymbol', 'naics', 'primexch', 'trdstat', 'secstat',
            'compno', 'is_event_date',
            # Target variable
            'future_ret'
        ]
        
        # Only include columns that exist in the dataframe
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
        
        # Print feature columns for debugging
        print("\nFeature columns being used:")
        for i, col in enumerate(feature_cols):
            print(f"{i+1:2d}. {col}")
        
        # Filter to non-NaN target values
        df_with_target = df.dropna(subset=['future_ret'])
        
        # Ensure we have data after filtering
        if len(df_with_target) == 0:
            raise ValueError("No data remains after filtering for valid target values")
        
        # Convert feature matrix to numeric, dropping any non-numeric columns
        X = df_with_target[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Drop any columns that couldn't be converted to numeric
        X = X.dropna(axis=1)
        
        # Use SimpleImputer to handle any remaining NaN values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Print final feature matrix info
        print(f"\nFinal feature matrix shape: {X.shape}")
        print("Final feature columns:")
        for i, col in enumerate(X.columns):
            print(f"{i+1:2d}. {col}")
        
        # Verify no NaN values remain
        if X.isna().any().any():
            raise ValueError("NaN values still present in feature matrix after imputation")
        
        return X, df_with_target['future_ret']

class EventAnalysis:
    def __init__(self, data_loader, feature_engineer):
        """
        Initialize EventAnalysis with a data loader and feature engineer.
        
        Parameters:
        data_loader (EventDataLoader): Instance of EventDataLoader
        feature_engineer (EventFeatureEngineer): Instance of EventFeatureEngineer
        """
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data = None
        self.X = None
        self.y = None
        self.models = {}
        
    def load_and_prepare_data(self):
        """Load raw data, create target variable, and calculate features"""
        # Load raw data
        self.data = self.data_loader.load_data()
        print(f"Loaded data shape: {self.data.shape}")
        
        # Create target variable
        self.data = self.feature_engineer.create_target(self.data)
        print(f"Created target variable: {self.data['future_ret'].notna().sum()} non-null values")
        
        # Calculate features
        self.data = self.feature_engineer.calculate_features(self.data)
        
        # Get features and target
        self.X, self.y = self.feature_engineer.get_features_target(self.data)
        print(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")
        
        return self.data
    
    def train_models(self, test_size=0.2):
        """Train models using time-based cross validation"""
        # Get the original data with dates
        original_data = self.data.copy()

        # Get unique events
        events = original_data[['ticker', self.feature_engineer.event_date_col]].drop_duplicates()

        # Sort by event date
        events = events.sort_values(self.feature_engineer.event_date_col)

        # Calculate the split point (80% train, 20% test)
        split_idx = int(len(events) * (1 - test_size))
        split_date = events.iloc[split_idx][self.feature_engineer.event_date_col]

        print(f"Using events before {split_date} for training and after for testing")

        # Create masks for train and test based on event dates
        train_events = events.iloc[:split_idx]
        test_events = events.iloc[split_idx:]

        # Create a mapping from (ticker, event_date) to train/test sets
        train_pairs = set(zip(train_events['ticker'], train_events[self.feature_engineer.event_date_col]))
        test_pairs = set(zip(test_events['ticker'], test_events[self.feature_engineer.event_date_col]))

        # Create boolean masks for the feature matrix
        train_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, self.feature_engineer.event_date_col]
        ) in train_pairs)

        test_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, self.feature_engineer.event_date_col]
        ) in test_pairs)

        # Split the data
        X_train, y_train = self.X[train_mask], self.y[train_mask]
        X_test, y_test = self.X[test_mask], self.y[test_mask]

        print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")

        # Train TimeSeriesRidge model
        print("Training TimeSeriesRidge model...")
        ts_ridge = TimeSeriesRidge(alpha=1.0, lambda2=0.1)
        ts_ridge.fit(X_train, y_train)
        self.models['TimeSeriesRidge'] = ts_ridge

        # Train XGBoostDecile model
        print("Training XGBoostDecile model...")
        xgb_decile = XGBoostDecileModel(weight=0.7, momentum_feature='momentum_5')
        xgb_decile.fit(X_train, y_train)
        self.models['XGBoostDecile'] = xgb_decile

        # Store data splits for evaluation
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return self.models

    def evaluate_models(self):
        """Evaluate all trained models and return performance metrics"""
        results = {}
        
        for name, model in self.models.items():
            # Make predictions on test set
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'R2': r2
            }
            
            print(f"{name} - MSE: {mse:.6f}, RMSE: {np.sqrt(mse):.6f}, R2: {r2:.6f}")
        
        return results
    
    def plot_event_analysis(self, window_start=-10, window_end=20):
        """
        Analyze returns around event dates
        
        Parameters:
        window_start (int): Days before event to start analysis
        window_end (int): Days after event to end analysis
        """
        # Initialize DataFrame for aligned returns
        aligned_returns = pd.DataFrame(
            index=range(window_start, window_end + 1),
            columns=['avg_ret', 'cum_ret', 'count', 'pos_count', 'neg_count']
        ).fillna(0)
        
        # Collect returns for each day relative to event
        for ticker, event_date in self.data[['ticker', self.feature_engineer.event_date_col]].drop_duplicates().values:
            event_data = self.data[
                (self.data['ticker'] == ticker) & 
                (self.data[self.feature_engineer.event_date_col] == event_date)
            ].copy()
            
            # Only include events with sufficient data
            if len(event_data) < (window_end - window_start + 1):
                continue
            
            # Get returns indexed by days to event
            for day in range(window_start, window_end + 1):
                day_data = event_data[event_data['days_to_event'] == day]
                if not day_data.empty:
                    ret = day_data['ret'].iloc[0]
                    aligned_returns.loc[day, 'avg_ret'] += ret
                    aligned_returns.loc[day, 'count'] += 1
                    if ret > 0:
                        aligned_returns.loc[day, 'pos_count'] += 1
                    elif ret < 0:
                        aligned_returns.loc[day, 'neg_count'] += 1
        
        # Calculate averages and cumulative returns
        aligned_returns['avg_ret'] = aligned_returns['avg_ret'] / aligned_returns['count']
        aligned_returns['cum_ret'] = (1 + aligned_returns['avg_ret']).cumprod() - 1
        aligned_returns['pos_pct'] = aligned_returns['pos_count'] / aligned_returns['count'] * 100
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot average returns
        ax1.bar(aligned_returns.index, aligned_returns['avg_ret'] * 100, alpha=0.6)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Average Returns Around Event')
        ax1.set_xlabel('Days Relative to Event')
        ax1.set_ylabel('Average Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative returns
        ax2.plot(aligned_returns.index, aligned_returns['cum_ret'] * 100, 'b-')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative Returns Around Event')
        ax2.set_xlabel('Days Relative to Event')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return aligned_returns, fig
    
    def find_sample_events(self, n=5):
        """Find sample events to analyze"""
        # Get unique events that have data
        events = self.data[['ticker', self.feature_engineer.event_date_col]].drop_duplicates()
        
        # Sort by event date (most recent first)
        events[self.feature_engineer.event_date_col] = pd.to_datetime(events[self.feature_engineer.event_date_col])
        events = events.sort_values(self.feature_engineer.event_date_col, ascending=False).head(n)
        
        return events 