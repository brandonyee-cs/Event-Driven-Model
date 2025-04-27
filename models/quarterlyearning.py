import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, earnings_path, stock_paths, window_days=30):
        """
        Initialize DataLoader with earnings data path and a list of stock data paths.
        
        Parameters:
        earnings_path (str): Path to earnings announcement data CSV
        stock_paths (list): List of paths to stock data CSV files
        window_days (int): Number of days to look before and after earnings announcement
        """
        self.earnings_path = earnings_path
        self.stock_paths = stock_paths if isinstance(stock_paths, list) else [stock_paths]
        self.window_days = window_days
        
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
                
            # Rename ticker-related columns for consistency
            if 'sym_root' in stock_data.columns and 'ticker' not in stock_data.columns:
                stock_data = stock_data.rename(columns={'sym_root': 'ticker'})
            elif 'symbol' in stock_data.columns and 'ticker' not in stock_data.columns:
                stock_data = stock_data.rename(columns={'symbol': 'ticker'})
                
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
        """Load earnings announcement data and stock data, then merge them"""
        # Load earnings announcement data
        earnings_data = pd.read_csv(self.earnings_path, parse_dates=['Announcement Date'])
            
        # Ensure announcement date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(earnings_data['Announcement Date']):
            earnings_data['Announcement Date'] = pd.to_datetime(earnings_data['Announcement Date'])
            
        # Extract ticker from earnings data if needed
        if 'ticker' not in earnings_data.columns and 'Ticker' in earnings_data.columns:
            earnings_data['ticker'] = earnings_data['Ticker']
        
        # Get unique ticker and announcement date combinations
        earnings_events = earnings_data[['ticker', 'Announcement Date']].drop_duplicates()
        
        # Add quarter info if available
        if 'Quarter' in earnings_data.columns:
            earnings_events = earnings_data[['ticker', 'Announcement Date', 'Quarter']].drop_duplicates()
        elif 'Fiscal Quarter' in earnings_data.columns:
            earnings_events = earnings_data[['ticker', 'Announcement Date', 'Fiscal Quarter']].drop_duplicates()
            earnings_events = earnings_events.rename(columns={'Fiscal Quarter': 'Quarter'})
        
        print(f"Found {len(earnings_events)} unique quarterly earnings events")
        
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
        
        # Filter stock data to only include tickers with earnings announcements
        tickers_with_earnings = earnings_events['ticker'].unique()
        stock_data_filtered = stock_data[stock_data['ticker'].isin(tickers_with_earnings)]
        print(f"Filtered stock data to {len(tickers_with_earnings)} tickers with earnings announcements")
        
        # Create separate dataframes for each earnings event
        event_dataframes = []
        events_with_data = 0
        
        for _, event in earnings_events.iterrows():
            ticker = event['ticker']
            announcement_date = event['Announcement Date']
            
            # Add quarter info if available
            quarter = event.get('Quarter', None)
            
            # Define date range around announcement
            start_date = announcement_date - pd.Timedelta(days=self.window_days)
            end_date = announcement_date + pd.Timedelta(days=self.window_days)
            
            # Filter stock data for this ticker and date range
            event_data = stock_data_filtered[
                (stock_data_filtered['ticker'] == ticker) &
                (stock_data_filtered['date'] >= start_date) &
                (stock_data_filtered['date'] <= end_date)
            ].copy()
            
            # Skip if no data available
            if len(event_data) == 0:
                continue
                
            # Add earnings announcement information
            event_data['Announcement Date'] = announcement_date
            if quarter is not None:
                event_data['Quarter'] = quarter
            event_data['is_announcement_date'] = (event_data['date'] == announcement_date)
            
            # Add earnings surprise if available in earnings_data
            if 'Surprise' in earnings_data.columns:
                surprise = earnings_data.loc[
                    (earnings_data['ticker'] == ticker) & 
                    (earnings_data['Announcement Date'] == announcement_date),
                    'Surprise'
                ].values
                if len(surprise) > 0:
                    event_data['Surprise'] = surprise[0]
            
            # Add expected vs actual EPS if available
            for col in ['Expected EPS', 'Actual EPS']:
                if col in earnings_data.columns:
                    value = earnings_data.loc[
                        (earnings_data['ticker'] == ticker) & 
                        (earnings_data['Announcement Date'] == announcement_date),
                        col
                    ].values
                    if len(value) > 0:
                        event_data[col] = value[0]
            
            event_dataframes.append(event_data)
            events_with_data += 1
        
        # Combine all event dataframes
        if not event_dataframes:
            raise ValueError("No matching data found for any earnings announcement events")
            
        combined_data = pd.concat(event_dataframes, ignore_index=True)
        print(f"Created data for {events_with_data} earnings announcement events with available stock data")
        
        return combined_data

class FeatureEngineer:
    def __init__(self, prediction_window=3):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        
    def create_target(self, df, price_col='prc'):
        """
        Create the target variable: future returns over prediction_window days
        Following equation (8) from the paper: yi = (Pi+h - Pi) / Pi
        """
        df = df.copy()
        
        # Calculate future returns for each ticker and announcement event separately
        df['future_ret'] = df.groupby(['ticker', 'Announcement Date'])[price_col].transform(
            lambda x: x.pct_change(self.prediction_window).shift(-self.prediction_window)
        )
        
        return df
    
    def calculate_features(self, df):
        """
        Calculate all features for earnings analysis
        """
        df = df.copy()
        
        # Price momentum over different windows
        for window in self.windows:
            df[f'momentum_{window}'] = df.groupby(['ticker', 'Announcement Date'])['prc'].transform(
                lambda x: (x - x.shift(window)) / x.shift(window)
            )
        
        # Momentum differences
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']
        
        # Return volatility over different windows
        for window in self.windows:
            df[f'volatility_{window}'] = df.groupby(['ticker', 'Announcement Date'])['ret'].transform(
                lambda x: x.rolling(window, min_periods=max(2, window//5)).std()
            )
        
        # Volatility differences
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']
        
        # Log returns
        df['log_ret'] = df.groupby(['ticker', 'Announcement Date'])['prc'].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )
        
        # Days relative to announcement date
        df['days_to_announcement'] = (df['date'] - df['Announcement Date']).dt.days
        
        # Lagged returns (1, 2, 3 days)
        for lag in range(1, 4):
            df[f'ret_lag_{lag}'] = df.groupby(['ticker', 'Announcement Date'])['ret'].shift(lag)
        
        # Pre-announcement returns (cumulative 30-day)
        pre_announcement_rets = {}
        for (ticker, announcement_date), group in df.groupby(['ticker', 'Announcement Date']):
            pre_announcement_data = group[
                (group['date'] <= announcement_date) & 
                (group['date'] > announcement_date - pd.Timedelta(days=30))
            ]
            if not pre_announcement_data.empty:
                pre_announcement_ret = pre_announcement_data['ret'].sum()
                pre_announcement_rets[(ticker, announcement_date)] = pre_announcement_ret
        
        # Apply pre-announcement returns to the dataframe
        df['pre_announcement_ret'] = df.apply(
            lambda row: pre_announcement_rets.get((row['ticker'], row['Announcement Date']), np.nan) 
            if row['days_to_announcement'] >= 0 else np.nan,
            axis=1
        )
        
        # Volume-based features
        if 'vol' in df.columns:
            # Normalized volume
            df['norm_vol'] = df.groupby(['ticker', 'Announcement Date'])['vol'].transform(
                lambda x: x / x.rolling(20, min_periods=5).mean()
            )
            
            # Volume momentum
            for window in [5, 10]:
                df[f'vol_momentum_{window}'] = df.groupby(['ticker', 'Announcement Date'])['vol'].transform(
                    lambda x: (x - x.shift(window)) / x.shift(window)
                )
        
        # Earnings surprise features (if available)
        if 'Surprise' in df.columns:
            # Create dummy variables for positive/negative surprises
            df['pos_surprise'] = (df['Surprise'] > 0).astype(int)
            df['neg_surprise'] = (df['Surprise'] < 0).astype(int)
            
            # Surprise magnitude
            df['surprise_magnitude'] = df['Surprise'].abs()
        
        # Create time-of-day features for announcements (BMO, AMC)
        # BMO = Before Market Open, AMC = After Market Close
        if 'Time' in df.columns or 'Announcement Time' in df.columns:
            time_col = 'Time' if 'Time' in df.columns else 'Announcement Time'
            
            # Convert time strings to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df['announcement_hour'] = pd.to_datetime(df[time_col], errors='coerce').dt.hour
            else:
                df['announcement_hour'] = df[time_col].dt.hour
                
            # Create BMO/AMC indicators
            df['is_bmo'] = ((df['announcement_hour'] >= 0) & (df['announcement_hour'] < 9)).astype(int)
            df['is_amc'] = ((df['announcement_hour'] >= 16) & (df['announcement_hour'] <= 23)).astype(int)
            df['is_market_hours'] = ((~df['is_bmo']) & (~df['is_amc'])).astype(int)
        
        # Quarter features if available
        if 'Quarter' in df.columns:
            # Extract quarter number (e.g., Q1, Q2, Q3, Q4)
            df['quarter_num'] = df['Quarter'].str.extract(r'Q(\d)').astype(float)
            
            # Create quarter dummies
            for i in range(1, 5):
                df[f'is_q{i}'] = (df['quarter_num'] == i).astype(int)
        
        # Previous quarter surprise impact (if available)
        if 'Surprise' in df.columns:
            # Group by ticker and sort by announcement date
            previous_surprises = {}
            
            for ticker, group in df.groupby('ticker'):
                sorted_announcements = group[['Announcement Date', 'Surprise']].drop_duplicates().sort_values('Announcement Date')
                
                # Create a mapping of announcement dates to previous surprise
                prev_surprise_dict = {}
                prev_surprise = np.nan
                
                for _, row in sorted_announcements.iterrows():
                    announcement_date = row['Announcement Date']
                    prev_surprise_dict[announcement_date] = prev_surprise
                    prev_surprise = row['Surprise']
                
                previous_surprises[ticker] = prev_surprise_dict
            
            # Apply previous surprise to the dataframe
            df['prev_quarter_surprise'] = df.apply(
                lambda row: previous_surprises.get(row['ticker'], {}).get(row['Announcement Date'], np.nan),
                axis=1
            )
            
            # Create indicators for consecutive beats/misses
            df['consecutive_beat'] = ((df['Surprise'] > 0) & (df['prev_quarter_surprise'] > 0)).astype(int)
            df['consecutive_miss'] = ((df['Surprise'] < 0) & (df['prev_quarter_surprise'] < 0)).astype(int)
        
        # Add sector/industry dummy variables if available
        if 'Sector' in df.columns:
            sector_dummies = pd.get_dummies(df['Sector'], prefix='sector', drop_first=True)
            df = pd.concat([df, sector_dummies], axis=1)
        
        if 'Industry' in df.columns:
            # Get top N industries by frequency to avoid too many dummies
            top_industries = df['Industry'].value_counts().head(10).index
            df['Industry_Top'] = df['Industry'].apply(lambda x: x if x in top_industries else 'Other')
            industry_dummies = pd.get_dummies(df['Industry_Top'], prefix='industry', drop_first=True)
            df = pd.concat([df, industry_dummies], axis=1)
        
        # Fill NaN values with group medians for each feature
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_features:
            if col != 'future_ret':  # Don't fill target variable
                df[col] = df.groupby(['ticker', 'Announcement Date'])[col].transform(
                    lambda x: x.fillna(x.median()) if x.isna().any() else x
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
            'date', 'Announcement Date', 'Time', 'Announcement Time', 'days_to_announcement',
            # Identifier columns
            'ticker', 'permno', 'permco', 'issuno', 'hexcd', 'hsiccd', 'Quarter',
            'shrout', 'cfacpr', 'cfacshr', 'shrcd', 'exchcd', 'siccd', 'ncusip',
            'comnam', 'shrcls', 'tsymbol', 'naics', 'primexch', 'trdstat', 'secstat',
            'compno', 'Sector', 'Industry', 'Industry_Top', 'is_announcement_date',
            # Target variable
            'future_ret'
        ]
        
        # Only include columns that exist in the dataframe
        feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
        
        # Print feature columns for debugging
        print("\nFeature columns being used:")
        for i, col in enumerate(feature_cols):
            print(f"{i+1:2d}. {col}")
        
        # Filter to non-NaN target values and announcement days only for prediction
        df_with_target = df.dropna(subset=['future_ret'])
        
        # Optionally, filter to only include rows on announcement dates (or the day before/after)
        # Uncomment next line to predict only on announcement days
        # df_with_target = df_with_target[df_with_target['is_announcement_date']]
        
        # Ensure we have data after filtering
        if len(df_with_target) == 0:
            raise ValueError("No data remains after filtering for valid target values")
        
        # Convert feature matrix to numeric, dropping any non-numeric columns
        X = df_with_target[feature_cols].apply(pd.to_numeric, errors='coerce')
        
        # Drop any columns that couldn't be converted to numeric
        X = X.dropna(axis=1)
        
        # Use SimpleImputer to handle any remaining NaN values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        # Print final feature matrix info
        print(f"\nFinal feature matrix shape: {X.shape}")
        print("Final feature columns:")
        for i, col in enumerate(X.columns):
            print(f"{i+1:2d}. {col}")
        
        # Verify no NaN values remain
        if X.isna().any().any():
            raise ValueError("NaN values still present in feature matrix after imputation")
        
        return X, df_with_target['future_ret']

class TimeSeriesRidge(Ridge):
    """
    Ridge regression with temporal smoothing penalty.
    """
    def __init__(self, alpha=1.0, lambda2=0.1, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.lambda2 = lambda2
        
    def _get_differencing_matrix(self, n_features):
        """Create differencing matrix D"""
        # Check if we have enough features to create a differencing matrix
        if n_features <= 1:
            return np.zeros((0, n_features))  # Return empty matrix if only one feature
            
        D = np.zeros((n_features-1, n_features))
        for i in range(n_features-1):
            D[i, i] = 1
            D[i, i+1] = -1
        return D
        
    def fit(self, X, y, sample_weight=None):
        """Fit model with the combined penalty"""
        # Convert inputs to numpy arrays and ensure they're float64
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Get differencing matrix
        D = self._get_differencing_matrix(X.shape[1])
        
        # If we don't have enough features for differencing, fall back to standard Ridge
        if D.shape[0] == 0:
            return super().fit(X, y, sample_weight)
        
        # Create effective regularization matrix
        DT_D = D.T @ D
        effective_alpha = self.alpha * np.eye(X.shape[1]) + self.lambda2 * DT_D
        
        # Store original alpha to restore it later
        original_alpha = self.alpha
        
        # Temporarily modify alpha for fitting
        self.alpha = 0.0
        
        # Create augmented data for custom regularization
        effective_alpha = np.abs(effective_alpha)  # Ensure positive values
        sqrt_alpha = np.sqrt(effective_alpha)
        
        # Create augmented data
        X_augmented = np.vstack([X, sqrt_alpha])
        y_augmented = np.concatenate([y, np.zeros(X.shape[1])])
        
        # Fit the model with augmented data
        result = super().fit(X_augmented, y_augmented, sample_weight)
        
        # Restore original alpha
        self.alpha = original_alpha
        
        return result

class XGBoostDecileModel:
    """
    XGBoostDecile Ensemble Model for earnings announcements.
    Combines an XGBoost model with a decile-based model.
    """
    def __init__(self, weight=0.5, momentum_feature='momentum_5', n_deciles=10, 
                 alpha=0.1, lambda_smooth=0.1, xgb_params=None):
        self.weight = weight
        self.momentum_feature = momentum_feature
        self.n_deciles = n_deciles
        self.alpha = alpha
        self.lambda_smooth = lambda_smooth
        
        # Set default XGBoost parameters if not provided
        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        else:
            self.xgb_params = xgb_params
            
        # Initialize the XGBoost model
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        
        # Initialize decile models
        self.decile_models = [None] * n_deciles
        self.decile_boundaries = None
        self.numeric_columns = None
        
    def _convert_to_numeric(self, X):
        """Convert all features to numeric type, handling categorical variables if present"""
        X = X.copy()
        
        # If this is the first time, determine which columns need conversion
        if self.numeric_columns is None:
            self.numeric_columns = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Try to convert to numeric directly first
                    try:
                        X[col] = pd.to_numeric(X[col], errors='raise')
                        self.numeric_columns[col] = 'numeric'
                    except:
                        # If conversion fails, treat as categorical
                        self.numeric_columns[col] = 'categorical'
                        unique_values = X[col].unique()
                        self.numeric_columns[f'{col}_mapping'] = {val: idx for idx, val in enumerate(unique_values)}
                else:
                    self.numeric_columns[col] = 'numeric'
        
        # Convert columns based on stored mappings
        for col, col_type in self.numeric_columns.items():
            if col.endswith('_mapping'):
                continue
                
            if col_type == 'categorical':
                mapping = self.numeric_columns[f'{col}_mapping']
                X[col] = X[col].map(mapping).fillna(-1)
            elif col_type == 'numeric':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0)
        
        return X
        
    def _create_decile_assignments(self, X):
        """Assign observations to deciles based on the momentum feature value"""
        momentum_values = X[self.momentum_feature].values
        self.decile_boundaries = [np.quantile(momentum_values, q/self.n_deciles) 
                                  for q in range(1, self.n_deciles)]
        
        # Add extreme boundaries
        self.decile_boundaries = [float('-inf')] + self.decile_boundaries + [float('inf')]
        
        # Assign each observation to a decile
        decile_assignments = np.zeros(len(X), dtype=int)
        for i in range(self.n_deciles):
            mask = (momentum_values >= self.decile_boundaries[i]) & (momentum_values < self.decile_boundaries[i+1])
            decile_assignments[mask] = i
            
        return decile_assignments
    
    def _get_differencing_matrix(self, n_features):
        """Create differencing matrix D for regularization"""
        # Check if we have enough features to create a differencing matrix
        if n_features <= 1:
            return np.zeros((0, n_features))  # Return empty matrix if only one feature
            
        D = np.zeros((n_features-1, n_features))
        for i in range(n_features-1):
            D[i, i] = 1
            D[i, i+1] = -1
        return D
        
    def _fit_decile_model(self, X_decile, y_decile):
        """Fit a ridge regression model for a specific decile with temporal smoothing."""
        X_matrix = X_decile.values
        y_vector = y_decile.values
        
        # Check if we have enough data points to fit a model
        if len(X_matrix) < max(5, X_matrix.shape[1]):
            return None
        
        # Create differencing matrix for temporal smoothing
        D = self._get_differencing_matrix(X_matrix.shape[1])
        
        # Calculate β = (XᵀX + αI + λDᵀD)⁻¹Xᵀy
        XTX = X_matrix.T @ X_matrix
        XTy = X_matrix.T @ y_vector
        DTD = D.T @ D
        
        # Add regularization terms
        regularized_matrix = XTX + self.alpha * np.eye(X_matrix.shape[1]) + self.lambda_smooth * DTD
        
        # Solve for β
        try:
            beta = np.linalg.solve(regularized_matrix, XTy)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            beta = np.linalg.lstsq(regularized_matrix, XTy, rcond=None)[0]
            
        return beta
    
    def fit(self, X, y):
        """
        Fit both the XGBoost model and the decile-based models
        """
        # Convert features to numeric type
        X = self._convert_to_numeric(X)
        
        # Fit the XGBoost model
        self.xgb_model.fit(X, y)
        
        # Create decile assignments based on momentum feature using only training data
        decile_assignments = self._create_decile_assignments(X)
        
        # Fit a separate model for each decile
        for d in range(self.n_deciles):
            decile_mask = (decile_assignments == d)
            
            # If there are enough observations in this decile
            if np.sum(decile_mask) >= max(5, X.shape[1]):
                X_decile = X.loc[decile_mask]
                y_decile = y.loc[decile_mask]
                
                # Fit decile model
                self.decile_models[d] = self._fit_decile_model(X_decile, y_decile)
            else:
                # Not enough data for this decile, use global model
                self.decile_models[d] = None
        
        return self
    
    def predict(self, X):
        """
        Generate predictions using the ensemble of XGBoost and decile-based models
        """
        # Convert features to numeric type
        X = self._convert_to_numeric(X)
        
        # Get XGBoost predictions
        xgb_preds = self.xgb_model.predict(X)
        
        # Initialize decile predictions
        decile_preds = np.zeros_like(xgb_preds)
        
        # Assign observations to deciles
        momentum_values = X[self.momentum_feature].values
        
        for i, val in enumerate(momentum_values):
            # Find the appropriate decile
            decile = 0  # Default to first decile
            for d in range(self.n_deciles):
                if val >= self.decile_boundaries[d] and val < self.decile_boundaries[d+1]:
                    decile = d
                    break
            
            # If we have a model for this decile, use it
            if self.decile_models[decile] is not None:
                decile_preds[i] = np.dot(X.iloc[i].values, self.decile_models[decile])
            else:
                # Otherwise, use XGBoost prediction
                decile_preds[i] = xgb_preds[i]
        
        # Combine predictions with weight
        ensemble_preds = self.weight * xgb_preds + (1 - self.weight) * decile_preds
        
        return ensemble_preds
    
class SurpriseClassificationModel:
    """
    Model for classifying earnings surprises (positive, negative, or no surprise)
    and predicting post-announcement returns based on surprise characteristics.
    """
    def __init__(self, xgb_params=None):
        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.05,
                'objective': 'binary:logistic',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        else:
            self.xgb_params = xgb_params
            
        # Initialize models
        self.surprise_pos_model = xgb.XGBClassifier(**self.xgb_params)
        self.surprise_neg_model = xgb.XGBClassifier(**self.xgb_params)
        self.return_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        
    def fit(self, X, y, surprise_data):
        """
        Fit models:
        1. Model to predict positive surprise
        2. Model to predict negative surprise
        3. Model to predict returns based on surprise features
        
        Parameters:
        X (DataFrame): Features
        y (Series): Target returns
        surprise_data (Series): Earnings surprise values
        """
        # Create binary targets for surprise classification
        pos_surprise = (surprise_data > 0).astype(int)
        neg_surprise = (surprise_data < 0).astype(int)
        
        # Fit positive surprise model
        self.surprise_pos_model.fit(X, pos_surprise)
        
        # Fit negative surprise model
        self.surprise_neg_model.fit(X, neg_surprise)
        
        # Create features for return prediction
        X_with_surprise = X.copy()
        X_with_surprise['surprise_value'] = surprise_data
        X_with_surprise['pos_surprise'] = pos_surprise
        X_with_surprise['neg_surprise'] = neg_surprise
        X_with_surprise['surprise_magnitude'] = surprise_data.abs()
        
        # Fit return prediction model
        self.return_model.fit(X_with_surprise, y)
        
        return self
    
    def predict(self, X, surprise_data=None):
        """
        Generate predictions for positive surprise, negative surprise, and returns
        
        Parameters:
        X (DataFrame): Features
        surprise_data (Series, optional): Actual surprise values if available
        
        Returns:
        dict: Dictionary with predictions
        """
        # Predict probability of positive surprise
        pos_surprise_prob = self.surprise_pos_model.predict_proba(X)[:, 1]
        
        # Predict probability of negative surprise
        neg_surprise_prob = self.surprise_neg_model.predict_proba(X)[:, 1]
        
        # Predict binary surprise outcomes
        pos_surprise_pred = (pos_surprise_prob > 0.5).astype(int)
        neg_surprise_pred = (neg_surprise_prob > 0.5).astype(int)
        
        # Create features for return prediction
        X_with_surprise = X.copy()
        
        if surprise_data is not None:
            # Use actual surprise data if available
            X_with_surprise['surprise_value'] = surprise_data
            X_with_surprise['pos_surprise'] = (surprise_data > 0).astype(int)
            X_with_surprise['neg_surprise'] = (surprise_data < 0).astype(int)
            X_with_surprise['surprise_magnitude'] = surprise_data.abs()
        else:
            # Use predicted surprises
            X_with_surprise['surprise_value'] = pos_surprise_prob - neg_surprise_prob
            X_with_surprise['pos_surprise'] = pos_surprise_pred
            X_with_surprise['neg_surprise'] = neg_surprise_pred
            X_with_surprise['surprise_magnitude'] = np.abs(pos_surprise_prob - neg_surprise_prob)
        
        # Predict returns
        return_pred = self.return_model.predict(X_with_surprise)
        
        return {
            'pos_surprise_prob': pos_surprise_prob,
            'neg_surprise_prob': neg_surprise_prob,
            'pos_surprise': pos_surprise_pred,
            'neg_surprise': neg_surprise_pred,
            'return_pred': return_pred
        }
    
    def evaluate(self, X, y, surprise_data):
        """Evaluate model performance on test data"""
        # Make predictions
        preds = self.predict(X, surprise_data)
        
        # Evaluate surprise classification
        pos_surprise_actual = (surprise_data > 0).astype(int)
        neg_surprise_actual = (surprise_data < 0).astype(int)
        
        pos_surprise_accuracy = np.mean(preds['pos_surprise'] == pos_surprise_actual)
        neg_surprise_accuracy = np.mean(preds['neg_surprise'] == neg_surprise_actual)
        
        # Evaluate return prediction
        return_mse = mean_squared_error(y, preds['return_pred'])
        return_r2 = r2_score(y, preds['return_pred'])
        
        # Calculate directional accuracy
        return_direction_accuracy = np.mean(np.sign(preds['return_pred']) == np.sign(y))
        
        return {
            'pos_surprise_accuracy': pos_surprise_accuracy,
            'neg_surprise_accuracy': neg_surprise_accuracy,
            'return_mse': return_mse,
            'return_rmse': np.sqrt(return_mse),
            'return_r2': return_r2,
            'return_direction_accuracy': return_direction_accuracy
        }


class EarningsDriftModel:
    """
    Model to capture post-earnings announcement drift (PEAD).
    Uses separate models for different time horizons after earnings.
    """
    def __init__(self, time_horizons=[1, 3, 5, 10, 20]):
        self.time_horizons = time_horizons
        self.models = {
            horizon: xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                objective='reg:squarederror',
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ) for horizon in time_horizons
        }
        
    def prepare_data_for_horizon(self, data, horizon):
        """Prepare features and target for a specific time horizon"""
        # Group by ticker and announcement date
        groups = data.groupby(['ticker', 'Announcement Date'])
        
        X_list = []
        y_list = []
        
        for (ticker, announcement_date), group in groups:
            # Get data for announcement day
            announcement_day = group[group['days_to_announcement'] == 0]
            
            # Skip if no announcement day data
            if len(announcement_day) == 0:
                continue
                
            # Get data for the target horizon day
            horizon_day = group[group['days_to_announcement'] == horizon]
            
            # Skip if no horizon day data
            if len(horizon_day) == 0:
                continue
                
            # Get features from announcement day
            X_announcement = announcement_day.iloc[0].copy()
            
            # Get the target (return) for the horizon day
            cumulative_return = group[(group['days_to_announcement'] > 0) & 
                                     (group['days_to_announcement'] <= horizon)]['ret'].sum()
            
            X_list.append(X_announcement)
            y_list.append(cumulative_return)
        
        # Convert to DataFrames
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        return X, y
        
    def fit(self, data, feature_cols):
        """
        Fit separate models for different time horizons
        
        Parameters:
        data (DataFrame): Data with earnings announcements and returns
        feature_cols (list): List of feature column names
        """
        for horizon in self.time_horizons:
            print(f"Training model for {horizon}-day horizon...")
            
            # Prepare data for this horizon
            X, y = self.prepare_data_for_horizon(data, horizon)
            
            # Filter to include only feature columns
            X = X[feature_cols]
            
            # Fill any NaN values
            X = X.fillna(X.median())
            
            # Fit model for this horizon
            self.models[horizon].fit(X, y)
            
        return self
    
    def predict(self, data, feature_cols):
        """
        Generate predictions for all time horizons
        
        Parameters:
        data (DataFrame): Data with earnings announcements and features
        feature_cols (list): List of feature column names
        
        Returns:
        dict: Dictionary with predictions for each time horizon
        """
        # Filter to announcement days only
        announcement_days = data[data['days_to_announcement'] == 0].copy()
        
        # Get features
        X = announcement_days[feature_cols].fillna(announcement_days[feature_cols].median())
        
        # Make predictions for each horizon
        predictions = {}
        for horizon in self.time_horizons:
            predictions[f'horizon_{horizon}'] = self.models[horizon].predict(X)
            
        # Add predictions to the DataFrame
        for horizon, preds in predictions.items():
            announcement_days[horizon] = preds
            
        return announcement_days
    
    def evaluate(self, data, feature_cols):
        """Evaluate model performance on test data"""
        results = {}
        
        for horizon in self.time_horizons:
            # Prepare data for this horizon
            X, y = self.prepare_data_for_horizon(data, horizon)
            
            # Filter to include only feature columns
            X = X[feature_cols]
            
            # Fill any NaN values
            X = X.fillna(X.median())
            
            # Make predictions
            y_pred = self.models[horizon].predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y))
            
            results[horizon] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'R2': r2,
                'Direction Accuracy': direction_accuracy
            }
            
        return results

class Analysis:
    def __init__(self, data_loader, feature_engineer):
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer
        self.data = None
        self.X = None
        self.y = None
        self.models = {}
        
    def load_and_prepare_data(self):
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
        """Train various models on the data using time-based cross validation"""

        # Get the original data with dates
        original_data = self.data.copy()

        # Get unique earnings announcement events
        events = original_data[['ticker', 'Announcement Date']].drop_duplicates()

        # Sort by announcement date
        events = events.sort_values('Announcement Date')

        # Calculate the split point (80% train, 20% test)
        split_idx = int(len(events) * (1 - test_size))
        split_date = events.iloc[split_idx]['Announcement Date']

        print(f"Using earnings events before {split_date} for training and after for testing")

        # Create masks for train and test based on approval dates
        train_events = events.iloc[:split_idx]
        test_events = events.iloc[split_idx:]

        # Create a mapping from (ticker, announcement_date) to train/test sets
        train_pairs = set(zip(train_events['ticker'], train_events['Announcement Date']))
        test_pairs = set(zip(test_events['ticker'], test_events['Announcement Date']))

        # Create boolean masks for the feature matrix
        train_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, 'Announcement Date']
        ) in train_pairs)

        test_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, 'Announcement Date']
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
    
    def plot_feature_importance(self, model_name='TimeSeriesRidge', top_n=10):
        """Plot feature importance for a specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.models[model_name]
        
        if model_name == 'TimeSeriesRidge':
            # Get feature importance from Ridge model coefficients
            if not hasattr(model, 'coef_'):
                raise ValueError(f"Model {model_name} does not have feature coefficients")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': np.abs(model.coef_)
            })
        elif model_name == 'XGBoostDecile':
            # Get feature importance from XGBoost component
            importance = model.xgb_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': importance
            })
        else:
            raise ValueError(f"Feature importance not implemented for model {model_name}")
        
        # Sort by absolute importance and get top N
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Top {top_n} Features by Importance ({model_name} Model)')
        plt.tight_layout()
        
        return feature_importance
    
    def analyze_earnings_surprise(self):
        """Analyze the impact of earnings surprises on returns"""
        if 'Surprise' not in self.data.columns:
            print("No earnings surprise data available for analysis")
            return None
            
        # Group by surprise direction
        surprise_groups = {}
        surprise_groups['Positive'] = self.data[self.data['Surprise'] > 0]
        surprise_groups['Negative'] = self.data[self.data['Surprise'] < 0]
        surprise_groups['No Surprise'] = self.data[self.data['Surprise'] == 0]
        
        # Calculate average returns for each group
        results = {}
        for group_name, group_data in surprise_groups.items():
            # Skip empty groups
            if len(group_data) == 0:
                continue
                
            # Calculate average returns for different periods
            avg_returns = {
                'Announcement Day': group_data[group_data['days_to_announcement'] == 0]['ret'].mean(),
                '1 Day After': group_data[group_data['days_to_announcement'] == 1]['ret'].mean(),
                '3 Days After': group_data[(group_data['days_to_announcement'] > 0) & 
                                           (group_data['days_to_announcement'] <= 3)]['ret'].mean(),
                '5 Days After': group_data[(group_data['days_to_announcement'] > 0) & 
                                           (group_data['days_to_announcement'] <= 5)]['ret'].mean()
            }
            
            results[group_name] = avg_returns
        
        # Create a DataFrame for easy visualization
        results_df = pd.DataFrame(results)
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        results_df.plot(kind='bar')
        plt.title('Average Returns by Earnings Surprise Type')
        plt.ylabel('Average Return')
        plt.xlabel('Time Period')
        plt.tight_layout()
        
        return results_df
    
    def plot_predictions(self, ticker, announcement_date, model_name='TimeSeriesRidge'):
        """Plot actual vs. predicted returns for a specific earnings announcement event"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        # Convert announcement_date to datetime if it's not already
        announcement_date = pd.to_datetime(announcement_date)
        
        # Filter data for the specific event
        event_data = self.data[
            (self.data['ticker'] == ticker) & 
            (self.data['Announcement Date'] == announcement_date)
        ].sort_values('date')
        
        if len(event_data) == 0:
            raise ValueError(f"No data found for {ticker} on {announcement_date}")
        
        # Get feature columns for prediction
        feature_cols = self.X.columns
        event_features = event_data[feature_cols].copy()
        
        # Fill any NaN values with medians from the training set
        for col in feature_cols:
            if event_features[col].isna().any():
                if hasattr(self, 'X_train') and col in self.X_train:
                    event_features[col] = event_features[col].fillna(self.X_train[col].median())
                else:
                    event_features[col] = event_features[col].fillna(event_features[col].median())
        
        # Make predictions
        model = self.models[model_name]
        event_data['predicted_ret'] = model.predict(event_features)
        
        # Calculate cumulative returns (both actual and predicted)
        event_data['cum_actual'] = (1 + event_data['ret']).cumprod() - 1
        event_data['cum_predicted'] = (1 + event_data['predicted_ret']).cumprod() - 1
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative returns
        plt.subplot(2, 1, 1)
        plt.plot(event_data['date'], event_data['cum_actual'], 'b-', label='Actual Cumulative Return')
        plt.plot(event_data['date'], event_data['cum_predicted'], 'r--', label='Predicted Cumulative Return')
        plt.axvline(x=announcement_date, color='g', linestyle='--', label='Earnings Announcement Date')
        plt.title(f'{ticker} - {announcement_date.strftime("%Y-%m-%d")} - Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot daily returns
        plt.subplot(2, 1, 2)
        plt.plot(event_data['date'], event_data['ret'], 'b-', label='Actual Daily Return')
        plt.plot(event_data['date'], event_data['predicted_ret'], 'r--', label='Predicted Daily Return')
        plt.axvline(x=announcement_date, color='g', linestyle='--', label='Earnings Announcement Date')
        plt.title(f'{ticker} - {announcement_date.strftime("%Y-%m-%d")} - Daily Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Return the data for further analysis
        return event_data
        
    def find_sample_events(self, n=5):
        """Find sample earnings announcement events to analyze"""
        # Get unique approval events that have data
        events = self.data[['ticker', 'Announcement Date']].drop_duplicates()
        
        # Add earnings surprise if available
        if 'Surprise' in self.data.columns:
            events = self.data[['ticker', 'Announcement Date', 'Surprise']].drop_duplicates()
        
        # Sort by announcement date (most recent first)
        events['Announcement Date'] = pd.to_datetime(events['Announcement Date'])
        events = events.sort_values('Announcement Date', ascending=False).head(n)
        
        return events
    
    def plot_returns_around_announcement(self, pre_days=10, post_days=20):
        """Plot average returns around earnings announcement date"""
        # Initialize DataFrame for aligned returns
        aligned_returns = pd.DataFrame(
            index=range(-pre_days, post_days + 1),
            columns=['avg_ret', 'cum_ret', 'count', 'pos_count', 'neg_count']
        ).fillna(0)
        
        # Collect returns for each day relative to announcement
        for ticker, announcement_date in self.data[['ticker', 'Announcement Date']].drop_duplicates().values:
            event_data = self.data[(self.data['ticker'] == ticker) & 
                                   (self.data['Announcement Date'] == announcement_date)].copy()
            
            # Only include events with sufficient data
            if len(event_data) < (pre_days + post_days + 1):
                continue
            
            # Get returns indexed by days to announcement
            for day in range(-pre_days, post_days + 1):
                day_data = event_data[event_data['days_to_announcement'] == day]
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
        
        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Daily Average Returns
        ax1.bar(aligned_returns.index, aligned_returns['avg_ret'] * 100)
        ax1.axvline(x=0, color='r', linestyle='--', label='Earnings Announcement Date')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_title('Average Daily Returns Around Earnings Announcement Date')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cumulative Returns
        ax2.plot(aligned_returns.index, aligned_returns['cum_ret'] * 100, 'b-', linewidth=2)
        ax2.axvline(x=0, color='r', linestyle='--', label='Earnings Announcement Date')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_title('Average Cumulative Returns Around Earnings Announcement Date')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Percentage of Positive Returns
        ax3.bar(aligned_returns.index, aligned_returns['pos_pct'])
        ax3.axvline(x=0, color='r', linestyle='--', label='Earnings Announcement Date')
        ax3.axhline(y=50, color='k', linestyle='-', alpha=0.3, label='50% Threshold')
        ax3.set_title('Percentage of Positive Returns Around Earnings Announcement Date')
        ax3.set_xlabel('Days Relative to Announcement')
        ax3.set_ylabel('Positive Returns (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        return aligned_returns