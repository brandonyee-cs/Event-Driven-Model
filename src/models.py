import numpy as np
import polars as pl # Use Polars for type hints if applicable, but core logic is NumPy/Sklearn
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt

pl.Config.set_engine_affinity(engine="streaming")

class GARCH:
    """
    Implements GARCH(1,1) and GJR-GARCH models as described in the paper:
    "Modeling Equilibrium Asset Pricing Around Events with Heterogeneous Beliefs,
    Dynamic Volatility, and a Two-Risk Uncertainty Framework"
    
    The model estimates volatility using standard GARCH(1,1):
    σ²t = ω + α*ε²t-1 + β*σ²t-1
    
    Or GJR-GARCH for asymmetric volatility response:
    σ²t = ω + α*ε²t-1 + β*σ²t-1 + γ*I[εt-1<0]*ε²t-1
    
    Also implements the three-phase volatility process around events:
    σt = σ0 * (1 + φ1(t) * I[t≤tevent] + φ2(t) * I[tevent<t≤tevent+δ] + φ3(t) * I[t>tevent+δ])
    """
    
    def __init__(self, returns=None, omega=0.00001, alpha=0.1, beta=0.8, gamma=0):
        """
        Initialize GARCH model with parameters.
        
        Parameters:
        -----------
        returns : array-like, optional
            Historical returns data
        omega : float, optional
            Long-run average variance (constant term)
        alpha : float, optional
            ARCH parameter measuring impact of past shocks
        beta : float, optional
            GARCH parameter measuring persistence of volatility
        gamma : float, optional
            GJR parameter measuring asymmetric impact of negative shocks
            (if 0, standard GARCH is used; if non-zero, GJR-GARCH is used)
        """
        self.returns = returns
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_gjr = gamma != 0
        self.fitted = False
        self.h = None  # Conditional variances
        self.std_resid = None  # Standardized residuals
        
    def _log_likelihood(self, params):
        """
        Calculate the negative log-likelihood function for GARCH/GJR-GARCH.
        
        Parameters:
        -----------
        params : array-like
            GARCH parameters [omega, alpha, beta, gamma]
            
        Returns:
        --------
        float
            Negative log-likelihood value
        """
        if self.returns is None:
            raise ValueError("Returns data must be provided")
            
        omega, alpha, beta = params[:3]
        gamma = params[3] if len(params) > 3 else 0
        
        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
            return np.inf
        
        # Stationarity condition
        if self.is_gjr and alpha + beta + 0.5 * gamma >= 1:
            return np.inf
        elif not self.is_gjr and alpha + beta >= 1:
            return np.inf
            
        T = len(self.returns)
        h = np.zeros(T)
        h[0] = np.var(self.returns)  # Initialize variance with sample variance
        
        # Compute conditional variances
        for t in range(1, T):
            if self.is_gjr:
                # GJR-GARCH
                leverage = 1 if self.returns[t-1] < 0 else 0
                h[t] = omega + alpha * self.returns[t-1]**2 + beta * h[t-1] + gamma * leverage * self.returns[t-1]**2
            else:
                # Standard GARCH
                h[t] = omega + alpha * self.returns[t-1]**2 + beta * h[t-1]
        
        # Compute log-likelihood
        logliks = -0.5 * (np.log(2 * np.pi) + np.log(h) + self.returns**2 / h)
        loglik = np.sum(logliks)
        
        return -loglik  # Return negative for minimization
    
    def fit(self, returns=None, model_type='garch'):
        """
        Estimate GARCH parameters using maximum likelihood estimation.
        
        Parameters:
        -----------
        returns : array-like, optional
            Historical returns data (if not provided during initialization)
        model_type : str, optional
            'garch' for standard GARCH(1,1), 'gjr' for GJR-GARCH
            
        Returns:
        --------
        self
        """
        if returns is not None:
            self.returns = returns
            
        if isinstance(self.returns, pl.Series):
            self.returns = self.returns.to_numpy()
        elif isinstance(self.returns, pl.DataFrame):
            raise ValueError("Please provide a returns Series, not a DataFrame")
        
        if self.returns is None:
            raise ValueError("Returns data must be provided")
            
        self.is_gjr = model_type.lower() == 'gjr'
        
        # Initial parameter guesses
        if self.is_gjr:
            initial_params = [self.omega, self.alpha, self.beta, self.gamma]
            bounds = [(1e-6, 1), (0, 1), (0, 1), (0, 1)]
        else:
            initial_params = [self.omega, self.alpha, self.beta]
            bounds = [(1e-6, 1), (0, 1), (0, 1)]
        
        # Optimization
        result = minimize(
            self._log_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update parameters
        self.omega = result.x[0]
        self.alpha = result.x[1]
        self.beta = result.x[2]
        if self.is_gjr:
            self.gamma = result.x[3]
        
        # Compute conditional variances and standardized residuals
        self._compute_variances()
        
        self.fitted = True
        return self
    
    def _compute_variances(self):
        """
        Compute conditional variances and standardized residuals.
        """
        T = len(self.returns)
        self.h = np.zeros(T)
        self.h[0] = np.var(self.returns)
        
        for t in range(1, T):
            if self.is_gjr:
                leverage = 1 if self.returns[t-1] < 0 else 0
                self.h[t] = self.omega + self.alpha * self.returns[t-1]**2 + self.beta * self.h[t-1] + self.gamma * leverage * self.returns[t-1]**2
            else:
                self.h[t] = self.omega + self.alpha * self.returns[t-1]**2 + self.beta * self.h[t-1]
        
        self.std_resid = self.returns / np.sqrt(self.h)
    
    def forecast(self, horizon=1, start_idx=None):
        """
        Forecast conditional variances for a given horizon.
        
        Parameters:
        -----------
        horizon : int, optional
            Forecast horizon
        start_idx : int, optional
            Starting index for the forecast (if None, use the last observation)
            
        Returns:
        --------
        array-like
            Forecasted conditional variances
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
            
        if start_idx is None:
            start_idx = len(self.returns) - 1
            
        forecasts = np.zeros(horizon)
        last_var = self.h[start_idx]
        last_ret = self.returns[start_idx]
        
        for h in range(horizon):
            if h == 0:
                if self.is_gjr:
                    leverage = 1 if last_ret < 0 else 0
                    forecasts[h] = self.omega + self.alpha * last_ret**2 + self.beta * last_var + self.gamma * leverage * last_ret**2
                else:
                    forecasts[h] = self.omega + self.alpha * last_ret**2 + self.beta * last_var
            else:
                if self.is_gjr:
                    # For multi-step forecasts in GJR-GARCH, we use the expected value of the leverage term
                    forecasts[h] = self.omega + (self.alpha + 0.5 * self.gamma) * forecasts[h-1] + self.beta * forecasts[h-1]
                else:
                    forecasts[h] = self.omega + (self.alpha + self.beta) * forecasts[h-1]
        
        return forecasts
    
    def three_phase_volatility(self, t, t_event, k1=1.5, k2=2.0, delta=10, dt1=5, dt2=2, dt3=10, sigma_e0=None):
        """
        Calculate the three-phase volatility process as described in the paper.
        
        Parameters:
        -----------
        t : array-like
            Time points
        t_event : float
            Time of the event
        k1 : float, optional
            Pre-event volatility multiplier
        k2 : float, optional
            Post-event volatility multiplier
        delta : float, optional
            Duration of post-event rising phase
        dt1 : float, optional
            Pre-event rise duration parameter
        dt2 : float, optional
            Post-event rise rate parameter
        dt3 : float, optional
            Post-event decay rate parameter
        sigma_e0 : float, optional
            Baseline volatility (if None, use GARCH-estimated value)
            
        Returns:
        --------
        array-like
            Volatility at each time point
        """
        if sigma_e0 is None:
            if not self.fitted:
                raise ValueError("Model must be fitted to use GARCH-estimated baseline volatility")
            sigma_e0 = np.sqrt(self.omega / (1 - self.alpha - self.beta))
        
        t = np.asarray(t)
        sigma = np.zeros_like(t, dtype=float)
        
        # Pre-event phase
        pre_mask = t <= t_event
        if np.any(pre_mask):
            phi1 = (k1 - 1) * np.exp(-((t[pre_mask] - t_event)**2) / (2 * dt1**2))
            sigma[pre_mask] = sigma_e0 * (1 + phi1)
        
        # Post-event rising phase
        post_rise_mask = (t > t_event) & (t <= t_event + delta)
        if np.any(post_rise_mask):
            phi2 = (k2 - 1) * (1 - np.exp(-(t[post_rise_mask] - t_event) / dt2))
            sigma[post_rise_mask] = sigma_e0 * (1 + phi2)
        
        # Post-event decay phase
        post_decay_mask = t > t_event + delta
        if np.any(post_decay_mask):
            phi3 = (k2 - 1) * np.exp(-(t[post_decay_mask] - (t_event + delta)) / dt3)
            sigma[post_decay_mask] = sigma_e0 * (1 + phi3)
        
        return sigma
    
    def simulate(self, n_periods, event_time, params=None, seed=None):
        """
        Simulate returns using the GARCH model with the three-phase volatility process.
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to simulate
        event_time : int
            Index of the event time
        params : dict, optional
            Dictionary with parameters for three_phase_volatility
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (simulated returns, conditional volatilities)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Default parameters for three-phase volatility
        default_params = {
            'k1': 1.5,
            'k2': 2.0,
            'delta': 10,
            'dt1': 5,
            'dt2': 2,
            'dt3': 10
        }
        
        if params is None:
            params = {}
            
        # Update with provided parameters
        for k, v in params.items():
            default_params[k] = v
            
        # Time points
        t = np.arange(n_periods)
        
        # Baseline volatility
        if self.fitted:
            sigma_e0 = np.sqrt(self.omega / (1 - self.alpha - self.beta))
        else:
            sigma_e0 = np.sqrt(self.omega / (1 - self.alpha - self.beta))
            
        # Get three-phase volatility
        sigma = self.three_phase_volatility(
            t, 
            event_time, 
            k1=default_params['k1'],
            k2=default_params['k2'],
            delta=default_params['delta'],
            dt1=default_params['dt1'],
            dt2=default_params['dt2'],
            dt3=default_params['dt3'],
            sigma_e0=sigma_e0
        )
        
        # Simulate returns
        z = np.random.standard_normal(n_periods)
        returns = sigma * z
        
        return returns, sigma
    
    def impact_uncertainty(self):
        """
        Calculate impact uncertainty as defined in the paper.
        
        Returns:
        --------
        array-like
            Impact uncertainty at each time point
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating impact uncertainty")
            
        # Impact uncertainty is defined as the difference between realized and expected conditional variance
        expected_var = np.zeros_like(self.h)
        expected_var[0] = self.h[0]
        
        for t in range(1, len(self.h)):
            expected_var[t] = self.omega + self.alpha * self.returns[t-1]**2 + self.beta * self.h[t-1]
            if self.is_gjr and self.returns[t-1] < 0:
                expected_var[t] += self.gamma * self.returns[t-1]**2
        
        impact_uncertainty = self.h - expected_var
        return impact_uncertainty

    def diagnostic_tests(self):
        """
        Perform diagnostic tests on the fitted model.
        
        Returns:
        --------
        dict
            Dictionary with test results
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before performing diagnostic tests")
            
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        results = {}
        
        # Ljung-Box test for autocorrelation in standardized residuals
        lb_test = acorr_ljungbox(self.std_resid, lags=[10, 15, 20])
        results['ljung_box'] = {
            'statistic': lb_test.iloc[0].values,
            'p_values': lb_test.iloc[1].values
        }
        
        # Ljung-Box test for autocorrelation in squared standardized residuals
        lb_test_squared = acorr_ljungbox(self.std_resid**2, lags=[10, 15, 20])
        results['ljung_box_squared'] = {
            'statistic': lb_test_squared.iloc[0].values,
            'p_values': lb_test_squared.iloc[1].values
        }
        
        # Jarque-Bera test for normality
        jb_test = stats.jarque_bera(self.std_resid)
        results['jarque_bera'] = {
            'statistic': jb_test[0],
            'p_value': jb_test[1]
        }
        
        return results
    
    def plot_volatility(self, actual_returns=None, title="GARCH Volatility"):
        """
        Plot the conditional volatility estimated by the model.
        
        Parameters:
        -----------
        actual_returns : array-like, optional
            Actual returns for comparison
        title : str, optional
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.h))
        ax.plot(x, np.sqrt(self.h), label='Conditional Volatility')
        
        if actual_returns is not None:
            ax.plot(x, np.abs(actual_returns), 'r.', alpha=0.3, label='|Returns|')
            
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class TimeSeriesRidge(Ridge):
    """
    Ridge regression with temporal smoothing penalty.

    The model minimizes:
    ||y - Xβ||² + α||β||² + λ₂||Dβ||²

    Where D is a differencing matrix penalizing changes between *consecutive* coefficients
    assuming features in X are ordered meaningfully (e.g., by time lag or window size).
    Note: The effectiveness depends heavily on the order of features in X.

    Accepts Polars DataFrame for X, converts to NumPy for internal calculation.
    """
    def __init__(self, alpha=1.0, lambda2=0.1, feature_order=None, **kwargs):
        """
        Parameters:
        alpha (float): L2 regularization strength (standard Ridge).
        lambda2 (float): Temporal smoothing strength.
        feature_order (list, optional): The list of feature names in the desired order
                                         for applying the differencing penalty. If None,
                                         the penalty is applied based on the column order
                                         of X passed to fit().
        kwargs: Additional arguments for Ridge.
        """
        super().__init__(alpha=alpha, **kwargs)
        self.lambda2 = lambda2
        self.feature_order = feature_order # Store the desired order
        self.feature_names_in_ = None # Store actual feature names used in fit

    def _get_differencing_matrix(self, n_features):
        """Create differencing matrix D based on consecutive features."""
        if n_features <= 1:
            # No differences to compute for 0 or 1 feature
            return np.zeros((0, n_features))

        D = np.zeros((n_features - 1, n_features))
        for i in range(n_features - 1):
            D[i, i] = 1
            D[i, i + 1] = -1
        return D

    def fit(self, X, y, sample_weight=None):
        """
        Fit model with the combined penalty.
        If self.feature_order is set, X (Polars DF) is reordered before applying the penalty.
        Converts X and y to NumPy arrays for Ridge fitting.
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame.")
        if isinstance(y, pl.Series):
            y_np = y.to_numpy()
        elif isinstance(y, np.ndarray):
            y_np = y
        else:
            raise TypeError("y must be a Polars Series or NumPy array.")

        original_X_columns = X.columns # Keep track of original order/names

        # Reorder X (Polars DF) according to feature_order if provided
        if self.feature_order is not None:
             missing_features = set(self.feature_order) - set(X.columns)
             if missing_features:
                 raise ValueError(f"Features specified in feature_order are missing from X: {missing_features}")
             extra_features = set(X.columns) - set(self.feature_order)
             if extra_features:
                 print(f"Warning: X contains features not in feature_order: {extra_features}. They will be placed at the end.")
                 # Maintain all columns, but order according to feature_order first
                 ordered_cols = self.feature_order + list(extra_features)
                 X_ordered = X.select(ordered_cols) # Select columns in Polars
             else:
                 X_ordered = X.select(self.feature_order)
             # print(f"Fitting TimeSeriesRidge with feature order: {X_ordered.columns}")
             self.feature_names_in_ = X_ordered.columns # Store the order used for D matrix
        else:
             X_ordered = X # Use the DataFrame as is
             # if isinstance(X_ordered, pl.DataFrame):
             #     print(f"Fitting TimeSeriesRidge with default feature order: {X_ordered.columns}")
             self.feature_names_in_ = X_ordered.columns

        # Convert potentially reordered Polars DataFrame to NumPy array
        try:
            # Ensure numeric types before converting to NumPy
            numeric_cols = X_ordered.columns
            X_np = X_ordered.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols]
            ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert Polars DataFrame X to NumPy: {e}. Check dtypes.")

        # Ensure y is float64 numpy array
        y_np = np.asarray(y_np, dtype=np.float64)

        n_samples, n_features = X_np.shape

        # Basic checks for NaN/inf in NumPy arrays
        if np.isnan(X_np).any() or np.isinf(X_np).any():
             nan_cols = [self.feature_names_in_[i] for i in np.where(np.isnan(X_np).any(axis=0))[0]]
             inf_cols = [self.feature_names_in_[i] for i in np.where(np.isinf(X_np).any(axis=0))[0]]
             raise ValueError(f"NaN or Inf values detected in feature matrix X before fitting. NaN cols: {nan_cols}, Inf cols: {inf_cols}. Impute first.")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
             raise ValueError("NaN or Inf values detected in target vector y before fitting.")

        # Get differencing matrix based on the number of features used for D
        D = self._get_differencing_matrix(n_features)

        # If no differencing is possible (e.g., 1 feature), fall back to standard Ridge
        if D.shape[0] == 0 or self.lambda2 == 0:
            # print("Applying standard Ridge regression (lambda2=0 or n_features<=1).")
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))
            ridge_model.fit(X_np, y_np, sample_weight)
            self.coef_ = ridge_model.coef_
            self.intercept_ = ridge_model.intercept_
            # Feature names already stored in self.feature_names_in_
            return self

        # --- Augmentation Method for Combined Penalty ---
        sqrt_lambda2_D = np.sqrt(self.lambda2) * D
        X_augmented = np.vstack([X_np, sqrt_lambda2_D])
        y_augmented = np.concatenate([y_np, np.zeros(D.shape[0])])

        # Fit standard Ridge on augmented data with the original alpha (self.alpha)
        ridge_solver = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))

        if sample_weight is not None:
             augmented_weights = np.concatenate([sample_weight, np.ones(D.shape[0])])
             ridge_solver.fit(X_augmented, y_augmented, sample_weight=augmented_weights)
        else:
             ridge_solver.fit(X_augmented, y_augmented)

        # Store the fitted coefficients and intercept
        self.coef_ = ridge_solver.coef_
        self.intercept_ = ridge_solver.intercept_

        # If features were reordered for the D matrix, we need to ensure the final
        # self.coef_ corresponds to the *original* feature order of X.
        if self.feature_order is not None:
            # self.feature_names_in_ holds the order used for D
            # self.coef_ corresponds to self.feature_names_in_
            # We need coefficients corresponding to original_X_columns
            coef_dict = dict(zip(self.feature_names_in_, self.coef_))
            # Reconstruct based on original_X_columns, filling with 0 if a column was somehow dropped (shouldn't happen here)
            original_order_coef = [coef_dict.get(col, 0) for col in original_X_columns]
            self.coef_ = np.array(original_order_coef)
            # Update feature_names_in_ to reflect the final coefficient order
            self.feature_names_in_ = original_X_columns
        # else: self.feature_names_in_ already holds the correct (original) order

        return self

    def predict(self, X):
        """Predict using the fitted model. Expects Polars DataFrame X."""
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame for prediction.")
        if self.feature_names_in_ is None:
            raise RuntimeError("Model not fitted or feature names not stored.")

        # Ensure prediction X has the columns expected by the *fitted* model
        # (self.feature_names_in_ reflects the order coef_ corresponds to)
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Prediction data missing columns used during fit: {missing_cols}")

        # Select and order columns in Polars DF
        X_ordered = X.select(self.feature_names_in_)

        # Convert Polars DF to NumPy
        try:
            X_np = X_ordered.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
            ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert prediction Polars DataFrame X to NumPy: {e}. Check dtypes.")

        # Use the underlying Ridge predict method with the NumPy array
        return super().predict(X_np)


class XGBoostDecileModel:
    """
    XGBoostDecile Ensemble Model using Polars for data handling.
    Combines an XGBoost model with decile-based TimeSeriesRidge models.

    Prediction: yᵢ = w · yᵢ,XGBoost + (1 - w) · yᵢ,Decile
    """
    def __init__(self, weight=0.5, momentum_feature='momentum_5', n_deciles=10,
                 alpha=0.1, lambda_smooth=0.1, xgb_params=None, ts_ridge_feature_order=None):
        if not 0 <= weight <= 1:
             raise ValueError("Weight must be between 0 and 1.")
        self.weight = weight
        self.momentum_feature = momentum_feature
        self.n_deciles = n_deciles
        self.alpha = alpha # For decile TimeSeriesRidge
        self.lambda_smooth = lambda_smooth # For decile TimeSeriesRidge
        self.ts_ridge_feature_order = ts_ridge_feature_order # Pass feature order to decile models

        # Default XGBoost parameters
        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'n_jobs': -1
                # 'early_stopping_rounds': 10 # Add this back if needed, handle in fit
            }
        else:
            self.xgb_params = xgb_params

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.decile_models = [None] * n_deciles
        self.decile_boundaries = None # Store NumPy array of boundaries
        self.feature_names_in_ = None # Store feature names from training X

    def _calculate_decile_boundaries(self, X: pl.DataFrame):
        """Calculate decile boundaries using Polars quantile."""
        if self.momentum_feature not in X.columns:
             raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in input data.")

        # Calculate quantiles using Polars, ignoring nulls
        quantiles = X.select(
            pl.col(self.momentum_feature).quantile(q).alias(f"q_{q}")
            for q in np.linspace(0, 1, self.n_deciles + 1)[1:-1] # Exclude 0 and 1
        ).row(0) # Get quantiles as a tuple

        self.decile_boundaries = np.array(quantiles, dtype=np.float64)
        # print(f"Calculated {len(self.decile_boundaries)} decile boundaries based on '{self.momentum_feature}'.")


    def _assign_deciles(self, X: pl.DataFrame) -> pl.DataFrame:
        """Assign observations to deciles based on momentum feature using WHEN/THEN."""
        if self.decile_boundaries is None:
             raise RuntimeError("Decile boundaries have not been calculated. Call fit first.")
        if self.momentum_feature not in X.columns:
             raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in input data.")

        mom_col = pl.col(self.momentum_feature)
        boundaries = self.decile_boundaries # NumPy array

        # Handle NaNs: assign to decile 0
        # Build chained when/then expression
        decile_expr = pl.when(mom_col.is_nan()).then(pl.lit(0, dtype=pl.Int32))
        # Decile 0: less than first boundary
        decile_expr = decile_expr.when(mom_col < boundaries[0]).then(pl.lit(0, dtype=pl.Int32))
        # Intermediate deciles
        for i in range(len(boundaries) - 1):
            decile_expr = decile_expr.when(
                (mom_col >= boundaries[i]) & (mom_col < boundaries[i+1])
            ).then(pl.lit(i + 1, dtype=pl.Int32))
        # Last decile: greater than or equal to last boundary
        decile_expr = decile_expr.when(mom_col >= boundaries[-1]).then(pl.lit(self.n_deciles - 1, dtype=pl.Int32))
        # Fallback (should ideally not be reached if logic is complete)
        decile_expr = decile_expr.otherwise(pl.lit(0, dtype=pl.Int32)) # Assign unexpected to 0

        return X.with_columns(decile_expr.alias("decile_assignment"))

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Fit both the XGBoost model and the decile-based TimeSeriesRidge models.
        Converts data to NumPy for model fitting.
        """
        # print("Fitting XGBoostDecileModel (Polars input)...")
        if not isinstance(X, pl.DataFrame): raise TypeError("X must be a Polars DataFrame.")
        if not isinstance(y, pl.Series): raise TypeError("y must be a Polars Series.")
        if X.height != y.height: raise ValueError("X and y must have the same height.")

        self.feature_names_in_ = X.columns # Store feature names

        # --- Convert to NumPy for XGBoost and Ridge ---
        try:
            X_np = X.select( # Ensure float type for models
                [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
            ).to_numpy()
            y_np = y.cast(pl.Float64).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert Polars data to NumPy for fitting: {e}")

        # Check for NaNs/Infs in NumPy arrays (should be handled by FeatureEngineer ideally)
        if np.isnan(X_np).any() or np.isinf(X_np).any():
             print("Warning: NaNs or Infs detected in X_np before fitting XGBoostDecile.")
             # Optionally raise error or impute here as a fallback
        if np.isnan(y_np).any() or np.isinf(y_np).any():
             raise ValueError("NaN or Inf values detected in target vector y before fitting.")

        # --- Fit XGBoost Component ---
        # print(f"Fitting XGBoost component (Weight: {self.weight})...")
        try:
            # XGBoost sklearn API generally expects NumPy
            # Handle potential early stopping parameter
            fit_params = {}
            if 'early_stopping_rounds' in self.xgb_params:
                 # Need eval_set for early stopping - use the same data for simplicity
                 eval_set = [(X_np, y_np)]
                 fit_params['eval_set'] = eval_set
                 fit_params['early_stopping_rounds'] = self.xgb_params['early_stopping_rounds']
                 fit_params['verbose'] = False # Suppress verbose output

            self.xgb_model.fit(X_np, y_np, **fit_params)
            # print("XGBoost fitting complete.")
        except TypeError as e:
             if "unexpected keyword argument 'early_stopping_rounds'" in str(e) or \
                "got multiple values for keyword argument 'verbose'" in str(e):
                 # Older XGBoost or parameter conflict, try without early stopping
                 print("Warning: XGBoost parameter issue (e.g., early stopping). Retrying without it.")
                 xgb_params_fallback = self.xgb_params.copy()
                 xgb_params_fallback.pop('early_stopping_rounds', None)
                 self.xgb_model = xgb.XGBRegressor(**xgb_params_fallback)
                 self.xgb_model.fit(X_np, y_np)
                 # print("XGBoost fitting complete (without early stopping).")
             else: raise e
        except Exception as e: print(f"Error during XGBoost fit: {e}"); raise e


        # --- Fit Decile Components ---
        if self.weight < 1.0:
             # print(f"Fitting Decile TimeSeriesRidge components (Weight: {1 - self.weight})...")
             # Calculate decile boundaries based on the training Polars data
             self._calculate_decile_boundaries(X)

             # Assign training data points to deciles using Polars
             X_with_deciles = self._assign_deciles(X)

             # Fit a separate TimeSeriesRidge model for each decile
             for d in range(self.n_deciles):
                 # Filter Polars DataFrame for the current decile
                 decile_mask = pl.col("decile_assignment") == d
                 X_decile_pl = X_with_deciles.filter(decile_mask)

                 # Need corresponding y values. Get indices from Polars filter maybe?
                 # Easier: Add row number, filter, get y by row number. Or just filter y along with X.
                 # Let's filter y based on the same mask applied to X
                 # Note: This assumes y is aligned row-wise with X originally
                 y_decile_pl = y.filter(X_with_deciles.select(decile_mask).to_series()) # Filter y Series

                 # print(f"  Decile {d+1}/{self.n_deciles}: {X_decile_pl.height} samples.")
                 min_samples_required = max(5, len(self.feature_names_in_) + 1)
                 if X_decile_pl.height >= min_samples_required:
                     try:
                         # print(f"    Fitting TimeSeriesRidge for Decile {d+1}...")
                         decile_model = TimeSeriesRidge(
                             alpha=self.alpha,
                             lambda2=self.lambda_smooth,
                             feature_order=self.ts_ridge_feature_order # Pass feature order
                         )
                         # TimeSeriesRidge now expects Polars DF, converts internally
                         decile_model.fit(X_decile_pl.drop("decile_assignment"), y_decile_pl)
                         self.decile_models[d] = decile_model
                         # print(f"    Decile {d+1} model fitted.")
                     except Exception as e:
                         print(f"    Warning: Failed to fit model for Decile {d+1}. Reason: {e}")
                         self.decile_models[d] = None # Mark as failed
                 else:
                     # print(f"    Warning: Not enough samples ({X_decile_pl.height} < {min_samples_required}) for Decile {d+1}. Skipping.")
                     self.decile_models[d] = None
        # else: print("Skipping Decile model fitting as weight is 1.0 (XGBoost only).")

        # print("XGBoostDecileModel fitting complete.")
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Generate predictions using the ensemble. Expects Polars DataFrame X.
        Returns a NumPy array of predictions.
        """
        if self.feature_names_in_ is None:
             raise RuntimeError("Model has not been fitted yet. Call fit first.")
        if not isinstance(X, pl.DataFrame):
             raise TypeError("X must be a Polars DataFrame for prediction.")

        # Ensure prediction data has the same columns as training data, in the correct order
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
             raise ValueError(f"Missing columns in prediction data: {missing_cols}")
        # Select columns in the order used for training
        X_pred = X.select(self.feature_names_in_)

        # --- Convert to NumPy for XGBoost ---
        try:
             X_np = X_pred.select( # Ensure float type
                 [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
             ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert prediction Polars DataFrame to NumPy: {e}")

        # --- Get XGBoost Predictions ---
        xgb_preds = self.xgb_model.predict(X_np) # Returns NumPy array

        # --- Get Decile Predictions (if needed) ---
        if self.weight < 1.0:
             if self.decile_boundaries is None:
                 raise RuntimeError("Decile boundaries not set. Model needs fitting.")

             decile_preds_np = np.zeros_like(xgb_preds)

             # Assign test data to deciles using Polars
             X_with_deciles = self._assign_deciles(X_pred) # Adds 'decile_assignment' col

             for d in range(self.n_deciles):
                 decile_mask_pl = pl.col("decile_assignment") == d
                 # Create boolean mask Series to filter NumPy arrays later
                 bool_mask_np = X_with_deciles.select(decile_mask_pl).to_series().to_numpy()

                 if np.any(bool_mask_np): # If any samples fall into this decile
                     # Filter the *original* selected Polars DF (X_pred) for this decile
                     X_decile_test_pl = X_pred.filter(decile_mask_pl) # Filter based on boolean mask

                     # Use the fitted TimeSeriesRidge model for this decile
                     if self.decile_models[d] is not None:
                         try:
                             # Predict using the decile model (expects Polars DF)
                             preds_d = self.decile_models[d].predict(X_decile_test_pl) # Returns NumPy array
                             decile_preds_np[bool_mask_np] = preds_d
                         except Exception as e:
                             print(f"Warning: Error predicting with model for Decile {d+1}. Using XGBoost prediction as fallback. Error: {e}")
                             decile_preds_np[bool_mask_np] = xgb_preds[bool_mask_np] # Fallback
                     else:
                         # If no model was trained, use XGBoost prediction as fallback
                         # print(f"Warning: No model available for Decile {d+1}. Using XGBoost prediction.")
                         decile_preds_np[bool_mask_np] = xgb_preds[bool_mask_np] # Fallback

             # Combine predictions with weight
             ensemble_preds = self.weight * xgb_preds + (1 - self.weight) * decile_preds_np
        else:
             # If weight is 1, only use XGBoost predictions
             ensemble_preds = xgb_preds

        return ensemble_preds


class EventAsset:
    """
    Represents an event-related asset as described in the paper.
    
    This class implements the event-related asset with heterogeneous beliefs,
    dynamic volatility, and the two-risk uncertainty framework.
    """
    
    def __init__(self, returns=None, baseline_mu=0.001, rf_rate=0.0, 
                 risk_aversion=2.0, corr_generic=0.3, sigma_generic=0.01,
                 mu_generic=0.0005, transaction_cost_buy=0.001, transaction_cost_sell=0.0005):
        """
        Initialize the event-related asset.
        
        Parameters:
        -----------
        returns : array-like, optional
            Historical returns data
        baseline_mu : float, optional
            Baseline expected return
        rf_rate : float, optional
            Risk-free rate
        risk_aversion : float, optional
            Risk aversion coefficient
        corr_generic : float, optional
            Correlation with generic risky asset
        sigma_generic : float, optional
            Volatility of generic risky asset
        mu_generic : float, optional
            Expected return of generic risky asset
        transaction_cost_buy : float, optional
            Transaction cost for buying
        transaction_cost_sell : float, optional
            Transaction cost for selling
        """
        self.returns = returns
        self.baseline_mu = baseline_mu
        self.rf_rate = rf_rate
        self.gamma = risk_aversion
        self.rho = corr_generic
        self.sigma_g = sigma_generic
        self.mu_g = mu_generic
        self.tau_b = transaction_cost_buy
        self.tau_s = transaction_cost_sell
        
        # GARCH model for volatility
        self.garch_model = GARCH()
        self.garch_fitted = False
        
    def fit_garch(self, returns=None, model_type='garch'):
        """
        Fit GARCH model to returns data.
        
        Parameters:
        -----------
        returns : array-like, optional
            Returns data to fit
        model_type : str, optional
            Type of GARCH model ('garch' or 'gjr')
            
        Returns:
        --------
        self
        """
        if returns is not None:
            self.returns = returns
            
        self.garch_model.fit(self.returns, model_type=model_type)
        self.garch_fitted = True
        return self
    
    def optimal_weight(self, t, event_time, expected_return, w_prev=0, 
                       volatility=None, volatility_params=None):
        """
        Calculate the optimal portfolio weight for the event-related asset.
        
        Parameters:
        -----------
        t : float
            Current time
        event_time : float
            Time of the event
        expected_return : float
            Expected return of the asset
        w_prev : float, optional
            Previous weight
        volatility : float, optional
            Asset volatility (if None, computed using GARCH model)
        volatility_params : dict, optional
            Parameters for three-phase volatility calculation
            
        Returns:
        --------
        float
            Optimal weight
        """
        if not self.garch_fitted and volatility is None:
            raise ValueError("GARCH model must be fitted or volatility must be provided")
            
        # Compute volatility if not provided
        if volatility is None:
            if volatility_params is None:
                volatility_params = {}
                
            sigma_e = self.garch_model.three_phase_volatility(
                np.array([t]), 
                event_time, 
                **volatility_params
            )[0]
        else:
            sigma_e = volatility
            
        # Determine transaction cost based on buy/sell
        def compute_weight(tau):
            w_star = ((expected_return - self.rf_rate - tau * np.sign(w_star - w_prev)) / 
                     (self.gamma * (sigma_e**2 - self.rho * sigma_e * self.sigma_g)) -
                     (self.rho * sigma_e * self.sigma_g * (self.mu_g - self.rf_rate)) /
                     (sigma_e**2 * (self.sigma_g**2 - self.rho * sigma_e * self.sigma_g)))
            return w_star
            
        # First try with zero transaction cost to determine direction
        w_no_cost = compute_weight(0)
        
        # Then use appropriate transaction cost based on direction
        if w_no_cost > w_prev:
            tau = self.tau_b
        elif w_no_cost < w_prev:
            tau = self.tau_s
        else:
            tau = 0
            
        # Fixed point iteration to find optimal weight
        w = w_no_cost
        for _ in range(10):  # A few iterations should be enough for convergence
            w_prev_iter = w
            w = ((expected_return - self.rf_rate - tau * np.sign(w - w_prev)) / 
                 (self.gamma * (sigma_e**2 - self.rho * sigma_e * self.sigma_g)) -
                 (self.rho * sigma_e * self.sigma_g * (self.mu_g - self.rf_rate)) /
                 (sigma_e**2 * (self.sigma_g**2 - self.rho * sigma_e * self.sigma_g)))
                 
            if abs(w - w_prev_iter) < 1e-6:
                break
                
        return w
    
    def compute_rvr(self, t, event_time, expected_return, w_current, w_prev,
                   volatility=None, volatility_params=None):
        """
        Compute the Return-to-Variance Ratio (RVR).
        
        Parameters:
        -----------
        t : float
            Current time
        event_time : float
            Time of the event
        expected_return : float
            Expected return of the asset
        w_current : float
            Current weight
        w_prev : float
            Previous weight
        volatility : float, optional
            Asset volatility (if None, computed using GARCH model)
        volatility_params : dict, optional
            Parameters for three-phase volatility calculation
            
        Returns:
        --------
        float
            Return-to-Variance Ratio
        """
        if not self.garch_fitted and volatility is None:
            raise ValueError("GARCH model must be fitted or volatility must be provided")
            
        # Compute volatility if not provided
        if volatility is None:
            if volatility_params is None:
                volatility_params = {}
                
            sigma_e = self.garch_model.three_phase_volatility(
                np.array([t]), 
                event_time, 
                **volatility_params
            )[0]
        else:
            sigma_e = volatility
            
        # Determine transaction cost
        if w_current > w_prev:
            tau = self.tau_b
        elif w_current < w_prev:
            tau = self.tau_s
        else:
            tau = 0
            
        # Calculate RVR
        rvr = (expected_return - self.rf_rate - tau * abs(w_current - w_prev)) / (sigma_e**2)
        return rvr
    
    def simulate_prices(self, n_periods, event_time, initial_price=100, 
                        bias_params=None, volatility_params=None, seed=None):
        """
        Simulate asset prices around an event using the model.
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to simulate
        event_time : int
            Time of the event
        initial_price : float, optional
            Initial asset price
        bias_params : dict, optional
            Parameters for bias formation
        volatility_params : dict, optional
            Parameters for volatility calculation
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary with simulation results
        """
        if not self.garch_fitted:
            raise ValueError("GARCH model must be fitted before simulation")
            
        if seed is not None:
            np.random.seed(seed)
            
        # Default parameters
        if bias_params is None:
            bias_params = {
                'informed_bias': 0.0002,
                'uninformed_bias': 0.0005,
                'liquidity_bias': 0.0,
                'informed_noise': 0.001,
                'uninformed_noise': 0.003,
                'liquidity_noise': 0.005,
                'n_informed': 100,
                'n_uninformed': 200,
                'n_liquidity': 50,
                'liquidity_constraint': 0.5  # Reduction factor for pre-event purchases
            }
            
        if volatility_params is None:
            volatility_params = {
                'k1': 1.5,
                'k2': 2.0,
                'delta': 10,
                'dt1': 5,
                'dt2': 2,
                'dt3': 10
            }
            
        # Time points
        t = np.arange(n_periods)
        
        # Volatility process
        sigma = self.garch_model.three_phase_volatility(
            t,
            event_time,
            **volatility_params
        )
        
        # Generate news process (signal)
        signal = np.zeros(n_periods)
        
        # Pre-event signal (cumulative information)
        pre_event_signal = np.cumsum(np.random.normal(0, 0.01, event_time)) if event_time > 0 else np.array([])
        if len(pre_event_signal) > 0:
            signal[:event_time] = pre_event_signal
        
        # Event signal (actual news)
        event_signal = np.random.normal(0, 0.05)
        if event_time < n_periods:
            signal[event_time:] = event_signal
        
        # Investor belief formation
        informed_beliefs = signal + np.random.normal(bias_params['informed_bias'], bias_params['informed_noise'], n_periods)
        uninformed_beliefs = signal + np.random.normal(bias_params['uninformed_bias'], bias_params['uninformed_noise'], n_periods)
        liquidity_beliefs = signal + np.random.normal(bias_params['liquidity_bias'], bias_params['liquidity_noise'], n_periods)
        
        # Initialize arrays for weights, prices, returns
        w_informed = np.zeros(n_periods)
        w_uninformed = np.zeros(n_periods)
        w_liquidity = np.zeros(n_periods)
        prices = np.zeros(n_periods)
        returns = np.zeros(n_periods)
        rvr = np.zeros(n_periods)
        
        # Initial price
        prices[0] = initial_price
        
        # Initial weights (baseline allocation)
        initial_sigma = sigma[0]
        w_informed[0] = self.optimal_weight(0, event_time, self.baseline_mu + informed_beliefs[0], 0, initial_sigma)
        w_uninformed[0] = self.optimal_weight(0, event_time, self.baseline_mu + uninformed_beliefs[0], 0, initial_sigma)
        w_liquidity[0] = self.optimal_weight(0, event_time, self.baseline_mu + liquidity_beliefs[0], 0, initial_sigma)
        
        # Compute RVR for initial period
        rvr[0] = self.compute_rvr(0, event_time, self.baseline_mu, w_informed[0], 0, initial_sigma)
        
        # Simulation loop
        for i in range(1, n_periods):
            # Expected returns based on beliefs
            mu_informed = self.baseline_mu + informed_beliefs[i]
            mu_uninformed = self.baseline_mu + uninformed_beliefs[i]
            mu_liquidity = self.baseline_mu + liquidity_beliefs[i]
            
            # Calculate optimal weights
            w_informed[i] = self.optimal_weight(i, event_time, mu_informed, w_informed[i-1], sigma[i])
            w_uninformed[i] = self.optimal_weight(i, event_time, mu_uninformed, w_uninformed[i-1], sigma[i])
            
            # For liquidity traders, apply constraint before the event
            w_liquidity_unconstrained = self.optimal_weight(i, event_time, mu_liquidity, w_liquidity[i-1], sigma[i])
            if i < event_time and w_liquidity_unconstrained > w_liquidity[i-1]:
                # Reduce purchases pre-event
                w_liquidity[i] = w_liquidity[i-1] + bias_params['liquidity_constraint'] * (w_liquidity_unconstrained - w_liquidity[i-1])
            else:
                w_liquidity[i] = w_liquidity_unconstrained
                
            # Compute aggregate demand
            agg_demand = (bias_params['n_informed'] * w_informed[i] + 
                         bias_params['n_uninformed'] * w_uninformed[i] + 
                         bias_params['n_liquidity'] * w_liquidity[i])
            
            # Compute equilibrium price
            total_investors = bias_params['n_informed'] + bias_params['n_uninformed'] + bias_params['n_liquidity']
            supply_per_investor = 1.0  # Normalized supply
            
            # Simple price adjustment process
            price_adj_factor = 0.1
            prices[i] = prices[i-1] * (1 + price_adj_factor * (agg_demand - total_investors * supply_per_investor))
            
            # Compute realized return
            returns[i] = prices[i] / prices[i-1] - 1
            
            # Compute RVR
            rvr[i] = self.compute_rvr(i, event_time, self.baseline_mu, w_informed[i], w_informed[i-1], sigma[i])
            
        # Package results
        results = {
            'prices': prices,
            'returns': returns,
            'volatility': sigma,
            'w_informed': w_informed,
            'w_uninformed': w_uninformed,
            'w_liquidity': w_liquidity,
            'informed_beliefs': informed_beliefs,
            'uninformed_beliefs': uninformed_beliefs,
            'liquidity_beliefs': liquidity_beliefs,
            'rvr': rvr,
            'event_time': event_time
        }
        
        return results


class TwoRiskFramework:
    """
    Implements the two-risk framework as described in the paper.
    
    Distinguishes between directional news risk (uncertainty about event outcome)
    and impact uncertainty (uncertainty about magnitude of market response).
    """
    
    def __init__(self, directional_risk_vol=0.05, impact_uncertainty_vol=0.02,
                directional_risk_premium=0.05, impact_uncertainty_premium=0.03):
        """
        Initialize the two-risk framework.
        
        Parameters:
        -----------
        directional_risk_vol : float
            Volatility of directional news risk
        impact_uncertainty_vol : float
            Volatility of impact uncertainty
        directional_risk_premium : float
            Risk premium for directional risk
        impact_uncertainty_premium : float
            Risk premium for impact uncertainty
        """
        self.directional_risk_vol = directional_risk_vol
        self.impact_uncertainty_vol = impact_uncertainty_vol
        self.directional_risk_premium = directional_risk_premium
        self.impact_uncertainty_premium = impact_uncertainty_premium
        
    def decompose_returns(self, returns, event_time, pre_event_window=10, post_event_window=10):
        """
        Decompose returns into directional news risk and impact uncertainty components.
        
        Parameters:
        -----------
        returns : array-like
            Asset returns
        event_time : int
            Index of the event time
        pre_event_window : int, optional
            Number of periods before the event to consider
        post_event_window : int, optional
            Number of periods after the event to consider
            
        Returns:
        --------
        dict
            Dictionary with decomposed returns
        """
        n_periods = len(returns)
        
        # Initialize arrays
        directional_risk = np.zeros(n_periods)
        impact_uncertainty = np.zeros(n_periods)
        
        # Pre-event: only impact uncertainty matters
        pre_start = max(0, event_time - pre_event_window)
        for t in range(pre_start, event_time):
            impact_uncertainty[t] = self.impact_uncertainty_premium * np.random.normal(0, self.impact_uncertainty_vol)
            
        # At event: both risks resolve
        directional_risk[event_time] = self.directional_risk_premium * np.random.normal(0, self.directional_risk_vol)
        impact_uncertainty[event_time] = self.impact_uncertainty_premium * np.random.normal(0, self.impact_uncertainty_vol)
        
        # Post-event: only directional risk matters
        post_end = min(n_periods, event_time + post_event_window + 1)
        for t in range(event_time + 1, post_end):
            directional_risk[t] = self.directional_risk_premium * np.random.normal(0, self.directional_risk_vol)
            
        # Calculate total risk contribution
        total_risk = directional_risk + impact_uncertainty
        
        # Package results
        results = {
            'returns': returns,
            'directional_risk': directional_risk,
            'impact_uncertainty': impact_uncertainty,
            'total_risk': total_risk,
            'event_time': event_time
        }
        
        return results
        
    def estimate_risk_premia(self, returns, garch_model, event_time,
                           pre_event_window=10, post_event_window=10):
        """
        Estimate risk premia for directional risk and impact uncertainty.
        
        Parameters:
        -----------
        returns : array-like
            Asset returns
        garch_model : GARCH
            Fitted GARCH model
        event_time : int
            Index of the event time
        pre_event_window : int, optional
            Number of periods before the event to consider
        post_event_window : int, optional
            Number of periods after the event to consider
            
        Returns:
        --------
        dict
            Dictionary with estimated risk premia
        """
        # Calculate impact uncertainty using GARCH volatility innovations
        impact_uncertainty = garch_model.impact_uncertainty()
        
        # Pre-event and post-event windows
        pre_start = max(0, event_time - pre_event_window)
        post_end = min(len(returns), event_time + post_event_window + 1)
        
        # Isolate pre-event and post-event periods
        pre_returns = returns[pre_start:event_time]
        pre_impact_uncertainty = impact_uncertainty[pre_start:event_time]
        post_returns = returns[event_time:post_end]
        post_impact_uncertainty = impact_uncertainty[event_time:post_end]
        
        # Regression for pre-event period
        # r_t = alpha + beta * impact_uncertainty_t + epsilon_t
        pre_X = np.column_stack((np.ones(len(pre_returns)), pre_impact_uncertainty))
        pre_beta, _, _, _ = np.linalg.lstsq(pre_X, pre_returns, rcond=None)
        
        # Regression for post-event period
        # r_t = alpha + beta * directional_risk_t + epsilon_t
        # We approximate directional risk by the returns themselves
        post_X = np.column_stack((np.ones(len(post_returns)), post_returns))
        post_beta, _, _, _ = np.linalg.lstsq(post_X, post_returns, rcond=None)
        
        # Estimated risk premia
        impact_uncertainty_premium = pre_beta[1]
        directional_risk_premium = post_beta[1]
        
        # Package results
        results = {
            'impact_uncertainty_premium': impact_uncertainty_premium,
            'directional_risk_premium': directional_risk_premium,
            'pre_event_alpha': pre_beta[0],
            'post_event_alpha': post_beta[0]
        }
        
        return results


class MarketClearingModel:
    """
    Implements a market clearing model with heterogeneous investors.
    """
    
    def __init__(self, n_informed=100, n_uninformed=200, n_liquidity=50,
                risk_aversion_informed=2.0, risk_aversion_uninformed=3.0, risk_aversion_liquidity=1.5,
                bias_informed=0.0002, bias_uninformed=0.0005, bias_liquidity=0.0,
                noise_informed=0.001, noise_uninformed=0.003, noise_liquidity=0.005,
                liquidity_constraint=0.5, supply=1.0):
        """
        Initialize market clearing model.
        
        Parameters:
        -----------
        n_informed, n_uninformed, n_liquidity : int
            Number of each investor type
        risk_aversion_* : float
            Risk aversion coefficients
        bias_* : float
            Bias parameters for expectation formation
        noise_* : float
            Noise parameters for information quality
        liquidity_constraint : float
            Reduction factor for pre-event purchases by liquidity traders
        supply : float
            Supply of the asset per investor
        """
        self.n_informed = n_informed
        self.n_uninformed = n_uninformed
        self.n_liquidity = n_liquidity
        self.gamma_i = risk_aversion_informed
        self.gamma_u = risk_aversion_uninformed
        self.gamma_l = risk_aversion_liquidity
        self.bias_i = bias_informed
        self.bias_u = bias_uninformed
        self.bias_l = bias_liquidity
        self.noise_i = noise_informed
        self.noise_u = noise_uninformed
        self.noise_l = noise_liquidity
        self.liquidity_constraint = liquidity_constraint
        self.supply = supply
        
    def compute_equilibrium_price(self, event_asset, t, event_time, signal, prev_weights, 
                                  prev_price, rf_rate=0.0):
        """
        Compute equilibrium price that clears the market.
        
        Parameters:
        -----------
        event_asset : EventAsset
            The event-related asset
        t : int
            Current time
        event_time : int
            Time of the event
        signal : float
            Information signal
        prev_weights : dict
            Previous weights for each investor type
        prev_price : float
            Previous price
        rf_rate : float, optional
            Risk-free rate
            
        Returns:
        --------
        dict
            Dictionary with equilibrium results
        """
        # Generate beliefs based on signal
        informed_belief = signal + np.random.normal(self.bias_i, self.noise_i)
        uninformed_belief = signal + np.random.normal(self.bias_u, self.noise_u)
        liquidity_belief = signal + np.random.normal(self.bias_l, self.noise_l)
        
        # Get volatility for current period
        sigma_e = event_asset.garch_model.three_phase_volatility(
            np.array([t]), 
            event_time
        )[0]
        
        # Calculate optimal weights for each investor type
        w_i = event_asset.optimal_weight(
            t, event_time, event_asset.baseline_mu + informed_belief, 
            prev_weights.get('informed', 0), sigma_e
        )
        
        w_u = event_asset.optimal_weight(
            t, event_time, event_asset.baseline_mu + uninformed_belief, 
            prev_weights.get('uninformed', 0), sigma_e
        )
        
        # For liquidity traders, apply constraint before the event
        w_l_unconstrained = event_asset.optimal_weight(
            t, event_time, event_asset.baseline_mu + liquidity_belief, 
            prev_weights.get('liquidity', 0), sigma_e
        )
        
        if t < event_time and w_l_unconstrained > prev_weights.get('liquidity', 0):
            # Reduce purchases pre-event
            w_l = prev_weights.get('liquidity', 0) + self.liquidity_constraint * (w_l_unconstrained - prev_weights.get('liquidity', 0))
        else:
            w_l = w_l_unconstrained
            
        # Compute aggregate demand
        agg_demand = (self.n_informed * w_i + self.n_uninformed * w_u + self.n_liquidity * w_l)
        
        # Compute equilibrium price
        total_investors = self.n_informed + self.n_uninformed + self.n_liquidity
        total_supply = total_investors * self.supply
        
        # Use simple price adjustment process
        price_adj_factor = 0.1
        price = prev_price * (1 + price_adj_factor * (agg_demand - total_supply) / total_supply)
        
        # Compute realized return
        ret = price / prev_price - 1
        
        # Package results
        results = {
            'price': price,
            'return': ret,
            'weights': {
                'informed': w_i,
                'uninformed': w_u,
                'liquidity': w_l
            },
            'beliefs': {
                'informed': informed_belief,
                'uninformed': uninformed_belief,
                'liquidity': liquidity_belief
            },
            'volatility': sigma_e,
            'agg_demand': agg_demand,
            'excess_demand': agg_demand - total_supply
        }
        
        return results