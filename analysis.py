import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, fda_path, stock_paths, window_days=100):
        """
        Initialize DataLoader with Event data path and a list of stock data paths.
        
        Parameters:
        fda_path (str): Path to Event data CSV
        stock_paths (list): List of paths to stock data CSV files
        window_days (int): Number of days to look before and after event date
        """
        self.fda_path = fda_path
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
        """Load FDA approval data and stock data, then merge them"""
        # Load FDA approval data - always use CSV now
        fda_data = pd.read_csv(self.fda_path, parse_dates=['Approval Date'])
            
        # Ensure approval date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(fda_data['Approval Date']):
            fda_data['Approval Date'] = pd.to_datetime(fda_data['Approval Date'])
            
        # Extract ticker from FDA data if needed
        if 'ticker' not in fda_data.columns and 'Ticker' in fda_data.columns:
            fda_data['ticker'] = fda_data['Ticker'].apply(
                lambda x: x.split()[-1].strip() if isinstance(x, str) and ' ' in x else x
            )
        
        # Get unique ticker and approval date combinations
        approval_events = fda_data[['ticker', 'Approval Date', 'Drug Name']].drop_duplicates()
        print(f"Found {len(approval_events)} unique FDA approval events")
        
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
        
        # Filter stock data to only include tickers with FDA approvals
        tickers_with_approvals = approval_events['ticker'].unique()
        stock_data_filtered = stock_data[stock_data['ticker'].isin(tickers_with_approvals)]
        print(f"Filtered stock data to {len(tickers_with_approvals)} tickers with FDA approvals")
        
        # Create separate dataframes for each approval event
        event_dataframes = []
        events_with_data = 0
        
        for _, event in approval_events.iterrows():
            ticker = event['ticker']
            approval_date = event['Approval Date']
            drug_name = event['Drug Name']
            
            # Define date range around approval
            start_date = approval_date - pd.Timedelta(days=self.window_days)
            end_date = approval_date + pd.Timedelta(days=self.window_days)
            
            # Filter stock data for this ticker and date range
            event_data = stock_data_filtered[
                (stock_data_filtered['ticker'] == ticker) &
                (stock_data_filtered['date'] >= start_date) &
                (stock_data_filtered['date'] <= end_date)
            ].copy()
            
            # Skip if no data available
            if len(event_data) == 0:
                continue
                
            # Add FDA approval information
            event_data['Approval Date'] = approval_date
            event_data['Drug Name'] = drug_name
            event_data['is_approval_date'] = (event_data['date'] == approval_date)
            
            event_dataframes.append(event_data)
            events_with_data += 1
        
        # Combine all event dataframes
        if not event_dataframes:
            raise ValueError("No matching data found for any FDA approval events")
            
        combined_data = pd.concat(event_dataframes, ignore_index=True)
        print(f"Created data for {events_with_data} FDA approval events with available stock data")
        
        return combined_data

class FeatureEngineer:
    def __init__(self, prediction_window=5):
        self.windows = [5, 10, 20]
        self.prediction_window = prediction_window
        
    def create_target(self, df, price_col='prc'):
        """
        Create the target variable: future returns over prediction_window days
        Following equation (8) from the provided pdf: yi = (Pi+h - Pi) / Pi
        """
        df = df.copy()
        
        # Calculate future returns for each ticker and approval event separately
        df['future_ret'] = df.groupby(['ticker', 'Approval Date'])[price_col].transform(
            lambda x: x.pct_change(self.prediction_window).shift(-self.prediction_window)
        )
        
        return df
    
    def calculate_features(self, df):
        """
        Calculate all features according to equation (7) in the provided pdf
        """
        df = df.copy()
        
        # Price momentum over different windows
        for window in self.windows:
            df[f'momentum_{window}'] = df.groupby(['ticker', 'Approval Date'])['prc'].transform(
                lambda x: (x - x.shift(window)) / x.shift(window)
            )
        
        # Momentum differences
        df['delta_momentum_5_10'] = df['momentum_5'] - df['momentum_10']
        df['delta_momentum_10_20'] = df['momentum_10'] - df['momentum_20']
        
        # Return volatility over different windows
        for window in self.windows:
            df[f'volatility_{window}'] = df.groupby(['ticker', 'Approval Date'])['ret'].transform(
                lambda x: x.rolling(window, min_periods=max(2, window//5)).std()
            )
        
        # Volatility differences
        df['delta_volatility_5_10'] = df['volatility_5'] - df['volatility_10']
        df['delta_volatility_10_20'] = df['volatility_10'] - df['volatility_20']
        
        # Log returns
        df['log_ret'] = df.groupby(['ticker', 'Approval Date'])['prc'].transform(
            lambda x: np.log(x) - np.log(x.shift(1))
        )
        
        # Days relative to approval date
        df['days_to_approval'] = (df['date'] - df['Approval Date']).dt.days
        
        # Lagged returns (1, 2, 3 days)
        for lag in range(1, 4):
            df[f'ret_lag_{lag}'] = df.groupby(['ticker', 'Approval Date'])['ret'].shift(lag)
        
        # Pre-approval returns (cumulative 30-day) - More efficient implementation
        pre_approval_rets = {}
        for (ticker, approval_date), group in df.groupby(['ticker', 'Approval Date']):
            pre_approval_data = group[
                (group['date'] <= approval_date) & 
                (group['date'] > approval_date - pd.Timedelta(days=30))
            ]
            if not pre_approval_data.empty:
                pre_approval_ret = pre_approval_data['ret'].sum()
                pre_approval_rets[(ticker, approval_date)] = pre_approval_ret
        
        # Apply pre-approval returns to the dataframe
        df['pre_approval_ret'] = df.apply(
            lambda row: pre_approval_rets.get((row['ticker'], row['Approval Date']), np.nan) 
            if row['days_to_approval'] >= 0 else np.nan,
            axis=1
        )
        
        # Fill NaN values with group medians for each feature
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_features:
            if col != 'future_ret':  # Don't fill target variable
                df[col] = df.groupby(['ticker', 'Approval Date'])[col].transform(
                    lambda x: x.fillna(x.median()) if x.isna().any() else x
                )
        
        # Add previous day volatility feature
            df['prev_day_volatility'] = df.groupby(['ticker', 'Approval Date'])['ret'].transform(
                lambda x: x.shift(1).abs()  # Simple measure of previous day volatility
            )
            
            # Add previous day volatility as standard deviation
            df['prev_day_vol_std'] = df.groupby(['ticker', 'Approval Date'])['ret'].transform(
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
            'date', 'Approval Date', 'days_to_approval',
            # Identifier columns
            'ticker', 'permno', 'permco', 'issuno', 'hexcd', 'hsiccd', 
            'shrout', 'cfacpr', 'cfacshr', 'shrcd', 'exchcd', 'siccd', 'ncusip',
            'comnam', 'shrcls', 'tsymbol', 'naics', 'primexch', 'trdstat', 'secstat',
            'compno', 'Drug Name', 'is_approval_date',
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

class TimeSeriesRidge(Ridge):
    """
    Ridge regression with temporal smoothing penalty.
    
    The model minimizes:
    ||y - Xβ||² + α||β||² + λ₂||Dβ||²
    
    Where:
    - y is the target vector of future returns
    - X is the feature matrix
    - β is the coefficient vector
    - α is the L2 regularization parameter
    - λ₂ is the temporal smoothing parameter
    - D is a differencing matrix that penalizes changes between consecutive coefficients
    """
    def __init__(self, alpha=1.0, lambda2=0.1, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.lambda2 = lambda2
        
    def _get_differencing_matrix(self, n_features):
        """
        Create differencing matrix D as defined in equation (10) from the PDF:
        D = [
            [1, -1, 0, ..., 0],
            [0, 1, -1, ..., 0],
            ...
            [0, 0, 0, ..., -1],
            [0, 0, 0, ..., 1]
        ]
        """
        # Check if we have enough features to create a differencing matrix
        if n_features <= 1:
            return np.zeros((0, n_features))  # Return empty matrix if only one feature
            
        D = np.zeros((n_features-1, n_features))
        for i in range(n_features-1):
            D[i, i] = 1
            D[i, i+1] = -1
        return D
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit model with the combined penalty from equation (9):
        min ||y - Xβ||² + α||β||² + λ₂||Dβ||²
        """
        # Convert inputs to numpy arrays and ensure they're float64
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Print input data info
        print(f"\nInput data shapes: X: {X.shape}, y: {y.shape}")
        print(f"Input data types: X: {X.dtype}, y: {y.dtype}")
        print(f"Input data contains NaN: X: {np.isnan(X).any()}, y: {np.isnan(y).any()}")
        
        # Get differencing matrix
        D = self._get_differencing_matrix(X.shape[1])
        
        # If we don't have enough features for differencing, fall back to standard Ridge
        if D.shape[0] == 0:
            return super().fit(X, y, sample_weight)
        
        # Create effective regularization matrix: α_effective = αI + λ₂D^T D (equation 12)
        DT_D = D.T @ D
        effective_alpha = self.alpha * np.eye(X.shape[1]) + self.lambda2 * DT_D
        
        # Store original alpha to restore it later
        original_alpha = self.alpha
        
        # Temporarily modify alpha for fitting
        self.alpha = 0.0
        
        # Create augmented data for custom regularization
        # First, ensure effective_alpha is positive definite and has no NaN values
        effective_alpha = np.abs(effective_alpha)  # Ensure positive values
        sqrt_alpha = np.sqrt(effective_alpha)
        
        # Create augmented data with explicit dtype and shape checks
        X_augmented = np.vstack([X, sqrt_alpha])
        y_augmented = np.concatenate([y, np.zeros(X.shape[1])])
        
        # Print augmented data info
        print(f"\nAugmented data shapes: X_aug: {X_augmented.shape}, y_aug: {y_augmented.shape}")
        print(f"Augmented data types: X_aug: {X_augmented.dtype}, y_aug: {y_augmented.dtype}")
        print(f"Augmented data contains NaN: X_aug: {np.isnan(X_augmented).any()}, y_aug: {np.isnan(y_augmented).any()}")
        
        # Verify no NaN values in augmented data
        if np.isnan(X_augmented).any() or np.isnan(y_augmented).any():
            # If we have NaN values, try to identify where they are
            if np.isnan(X_augmented).any():
                nan_rows = np.where(np.isnan(X_augmented).any(axis=1))[0]
                print(f"\nNaN values found in X_augmented at rows: {nan_rows}")
                print("First few rows of X_augmented with NaN values:")
                for row in nan_rows[:5]:
                    print(f"Row {row}: {X_augmented[row]}")
            
            if np.isnan(y_augmented).any():
                nan_indices = np.where(np.isnan(y_augmented))[0]
                print(f"\nNaN values found in y_augmented at indices: {nan_indices}")
                print("First few values of y_augmented with NaN values:")
                for idx in nan_indices[:5]:
                    print(f"Index {idx}: {y_augmented[idx]}")
            
            raise ValueError("NaN values detected in augmented data")
        
        # Fit the model with augmented data
        result = super().fit(X_augmented, y_augmented, sample_weight)
        
        # Restore original alpha
        self.alpha = original_alpha
        
        return result

class XGBoostDecileModel:
    """
    XGBoostDecile Ensemble Model that combines an XGBoost model with a decile-based model.
    
    The overall prediction is:
    yᵢ = w · yᵢ,XGBoost + (1 - w) · yᵢ,Decile
    
    Where:
    - w is the weight parameter balancing the two model components
    - yᵢ,XGBoost is the prediction from the XGBoost component
    - yᵢ,Decile is the prediction from the decile-based component
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
        """
        Fit a ridge regression model for a specific decile with temporal smoothing.
        Following equation (4) and (5) from the provided PDF.
        """
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

        # Get unique approval events
        events = original_data[['ticker', 'Approval Date']].drop_duplicates()

        # Sort by approval date
        events = events.sort_values('Approval Date')

        # Calculate the split point (80% train, 20% test)
        split_idx = int(len(events) * (1 - test_size))
        split_date = events.iloc[split_idx]['Approval Date']

        print(f"Using approval events before {split_date} for training and after for testing")

        # Create masks for train and test based on approval dates
        train_events = events.iloc[:split_idx]
        test_events = events.iloc[split_idx:]

        # Create a mapping from (ticker, approval_date) to train/test sets
        train_pairs = set(zip(train_events['ticker'], train_events['Approval Date']))
        test_pairs = set(zip(test_events['ticker'], test_events['Approval Date']))

        # Create boolean masks for the feature matrix
        train_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, 'Approval Date']
        ) in train_pairs)

        test_mask = self.X.index.map(lambda idx: (
            original_data.loc[idx, 'ticker'], 
            original_data.loc[idx, 'Approval Date']
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
    
    def plot_predictions(self, ticker, approval_date, model_name='TimeSeriesRidge'):
        """Plot actual vs. predicted returns for a specific FDA approval event"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        # Convert approval_date to datetime if it's not already
        approval_date = pd.to_datetime(approval_date)
        
        # Filter data for the specific event
        event_data = self.data[
            (self.data['ticker'] == ticker) & 
            (self.data['Approval Date'] == approval_date)
        ].sort_values('date')
        
        if len(event_data) == 0:
            raise ValueError(f"No data found for {ticker} on {approval_date}")
        
        # Check if required columns exist
        required_cols = ['date', 'ret']
        for col in required_cols:
            if col not in event_data.columns:
                raise ValueError(f"Required column '{col}' not found in event data")
        
        # Get feature columns for prediction
        feature_cols = self.X.columns
        missing_features = [col for col in feature_cols if col not in event_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in event data: {missing_features}")
        
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
        plt.axvline(x=approval_date, color='g', linestyle='--', label='FDA Approval Date')
        plt.title(f'{ticker} - {approval_date.strftime("%Y-%m-%d")} - Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot daily returns
        plt.subplot(2, 1, 2)
        plt.plot(event_data['date'], event_data['ret'], 'b-', label='Actual Daily Return')
        plt.plot(event_data['date'], event_data['predicted_ret'], 'r--', label='Predicted Daily Return')
        plt.axvline(x=approval_date, color='g', linestyle='--', label='FDA Approval Date')
        plt.title(f'{ticker} - {approval_date.strftime("%Y-%m-%d")} - Daily Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Return the data for further analysis
        return event_data
        
    def find_sample_events(self, n=5):
        """Find sample FDA approval events to analyze"""
        # Get unique approval events that have data
        events = self.data[['ticker', 'Approval Date', 'Drug Name']].drop_duplicates()
        
        # Sort by approval date (most recent first)
        events['Approval Date'] = pd.to_datetime(events['Approval Date'])
        events = events.sort_values('Approval Date', ascending=False).head(n)
        
        return events