import numpy as np
import polars as pl # Use Polars for type hints if applicable, but core logic is NumPy/Sklearn
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy.optimize import minimize
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union


pl.Config.set_engine_affinity(engine="streaming")

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

class GARCHModel:
    """
    GARCH(1,1) volatility model for event study analysis.
    
    Implements sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    
    Used for baseline volatility estimation in the event study framework.
    """
    def __init__(self, omega: float = 0.00001, alpha: float = 0.05, beta: float = 0.90):
        """
        Initialize GARCH(1,1) model with parameters.
        
        Parameters:
        -----------
        omega : float
            Long-run average variance (constant term)
        alpha : float
            ARCH parameter that measures the impact of past shocks
        beta : float
            GARCH parameter that measures the persistence of volatility
        """
        self._check_parameters(omega, alpha, beta)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.is_fitted = False
        self.variance_history = None
        self.residuals_history = None
        self.sigma2_t = None
        self.mean = 0.0
    
    def _check_parameters(self, omega, alpha, beta):
        """Validate GARCH parameters for stationarity and positivity."""
        if omega <= 0:
            raise ValueError("omega must be positive")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if alpha + beta >= 1:
            raise ValueError("alpha + beta must be less than 1 for stationarity")
    
    def _neg_log_likelihood(self, params, returns):
        """
        Calculate negative log-likelihood for GARCH(1,1) model with improved numerical stability.
        """
        omega, alpha, beta = params

        # Safety check for parameters
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return np.inf

        T = len(returns)
        sigma2 = np.zeros(T)

        # Initialize with unconditional variance with safety floor
        sigma2[0] = max(1e-6, np.var(returns))

        # Calculate variance series with safety checks
        for t in range(1, T):
            # Calculate next variance
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            # Add lower bound to prevent numerical issues
            sigma2[t] = max(1e-6, sigma2[t])

        # Calculate log-likelihood with safety checks
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)

        # If we get an invalid value, return infinity
        if not np.isfinite(log_likelihood):
            return np.inf

        return -log_likelihood
    
    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'SLSQP', 
            max_iter: int = 1000) -> 'GARCHModel':
        """
        Fit GARCH model to return series.

        Parameters:
        -----------
        returns : array-like
            Return series for fitting
        method : str
            Optimization method for scipy.optimize.minimize
            Must be a method that supports bounds ('SLSQP', 'L-BFGS-B', 'trust-constr')
        max_iter : int
            Maximum iterations for optimization

        Returns:
        --------
        self : GARCHModel
            Fitted model
        """
        # Convert input to numpy array
        if isinstance(returns, pl.Series):
            returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame):
            if returns.width != 1:
                raise ValueError("If returns is a DataFrame, it must have only one column")
            returns_np = returns.to_numpy().flatten()
        else:
            returns_np = np.asarray(returns)

        # Remove NaN values
        returns_np = returns_np[~np.isnan(returns_np)]

        # Clip extreme values to prevent numerical issues
        std_dev = np.std(returns_np)
        clip_threshold = 10 * std_dev
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)

        # Demean returns
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean

        # Initial parameters
        initial_params = [self.omega, self.alpha, self.beta]

        # Set optimization method that supports bounds
        # BFGS doesn't support bounds, so ensure we use SLSQP, L-BFGS-B, or trust-constr
        if method in ['BFGS', 'CG', 'Newton-CG', 'Nelder-Mead']:
            method = 'SLSQP'  # Fall back to SLSQP which handles bounds well

        # Fit GARCH parameters using MLE
        try:
            result = minimize(
                self._neg_log_likelihood,
                initial_params,
                args=(returns_centered,),
                method=method,
                bounds=[(1e-6, 1), (0, 0.5), (0.5, 0.999)],
                options={'maxiter': max_iter}
            )

            if result.success:
                self.omega, self.alpha, self.beta = result.x
                print(f"Fitted GARCH parameters: omega={self.omega:.6f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
            else:
                print(f"Warning: GARCH fitting did not converge: {result.message}")
                # Fall back to initial parameters
                print("Using initial parameters instead")
        except Exception as e:
            print(f"Error fitting GARCH model: {e}")
            print("Using initial parameters instead")

        # Calculate fitted variance series
        T = len(returns_centered)
        sigma2 = np.zeros(T)
        sigma2[0] = max(1e-6, np.var(returns_centered))

        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * returns_centered[t-1]**2 + self.beta * sigma2[t-1]
            # Add safety floor
            sigma2[t] = max(1e-6, sigma2[t])

        # Store history
        self.variance_history = sigma2
        self.residuals_history = returns_centered
        self.sigma2_t = sigma2[-1]  # Most recent variance
        self.is_fitted = True

        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """
        Forecast volatility n steps ahead.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to forecast
            
        Returns:
        --------
        forecasts : np.ndarray
            Array of variance forecasts
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        forecasts = np.zeros(n_steps)
        forecasts[0] = self.omega + self.alpha * self.residuals_history[-1]**2 + self.beta * self.sigma2_t
        
        # Multi-step forecasts
        for h in range(1, n_steps):
            forecasts[h] = self.omega + (self.alpha + self.beta) * forecasts[h-1]
        
        return forecasts
    
    def conditional_volatility(self) -> np.ndarray:
        """Return the conditional volatility (std dev) series from the model fit."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing volatility")
        
        return np.sqrt(self.variance_history)
    
    def volatility_innovations(self) -> np.ndarray:
        """
        Calculate volatility innovations (impact uncertainty measure).
        
        Returns the difference between realized and expected conditional variance.
        
        Returns:
        --------
        innovations : np.ndarray
            Array of volatility innovations
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calculating innovations")
        
        T = len(self.variance_history)
        innovations = np.zeros(T-1)
        
        for t in range(1, T):
            expected_var_t = self.omega + self.alpha * self.residuals_history[t-2]**2 + self.beta * self.variance_history[t-2]
            realized_var_t = self.variance_history[t-1]
            innovations[t-1] = realized_var_t - expected_var_t
        
        return innovations


class GJRGARCHModel(GARCHModel):
    """
    GJR-GARCH model for asymmetric volatility response.
    
    Implements sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1} + gamma * I_{t-1} * epsilon^2_{t-1}
    
    Where I_{t-1} = 1 if epsilon_{t-1} < 0 and 0 otherwise, accounting for leverage effect
    where negative returns have larger impacts on volatility than positive returns.
    """
    def __init__(self, omega: float = 0.00001, alpha: float = 0.03, beta: float = 0.90, gamma: float = 0.04):
        """
        Initialize GJR-GARCH model with parameters.
        
        Parameters:
        -----------
        omega : float
            Long-run average variance (constant term)
        alpha : float
            ARCH parameter
        beta : float
            GARCH parameter
        gamma : float
            Asymmetry parameter (leverage effect)
        """
        # Call parent initialization
        super().__init__(omega, alpha, beta)
        self.gamma = gamma
        # Check extended parameter constraint
        self._check_gjr_parameters()
    
    def _check_gjr_parameters(self):
        """Validate GJR-GARCH parameters for stationarity and positivity."""
        if self.alpha + self.beta + 0.5 * self.gamma >= 1:
            raise ValueError("alpha + beta + 0.5*gamma must be less than 1 for stationarity")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
    
    def _neg_log_likelihood(self, params, returns):
        """
        Calculate negative log-likelihood for GJR-GARCH model with improved numerical stability.
        """
        omega, alpha, beta, gamma = params

        # Safety check for parameters
        if (omega <= 0 or alpha < 0 or beta < 0 or gamma < 0 or 
            alpha + beta + 0.5 * gamma >= 1):
            return np.inf

        T = len(returns)
        sigma2 = np.zeros(T)

        # Initialize with unconditional variance with safety floor
        sigma2[0] = max(1e-6, np.var(returns))

        # Calculate variance series with safety checks
        for t in range(1, T):
            # Indicator for negative return
            I_t_minus_1 = 1.0 if returns[t-1] < 0 else 0.0

            # Calculate next variance
            sigma2[t] = (omega + alpha * returns[t-1]**2 + 
                         beta * sigma2[t-1] + 
                         gamma * I_t_minus_1 * returns[t-1]**2)

            # Add lower bound to prevent numerical issues
            sigma2[t] = max(1e-6, sigma2[t])

        # Calculate log-likelihood with safety checks
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns**2 / sigma2)

        # If we get an invalid value, return infinity
        if not np.isfinite(log_likelihood):
            return np.inf

        return -log_likelihood

    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'SLSQP',  # Changed from 'BFGS' to 'SLSQP'
            max_iter: int = 1000) -> 'GJRGARCHModel':
        """
        Fit GJR-GARCH model to return series with improved optimization settings.

        Parameters:
        -----------
        returns : array-like
            Return series for fitting
        method : str
            Optimization method for scipy.optimize.minimize (defaults to SLSQP which supports bounds)
        max_iter : int
            Maximum iterations for optimization

        Returns:
        --------
        self : GJRGARCHModel
            Fitted model
        """
        # Convert input to numpy array
        if isinstance(returns, pl.Series):
            returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame):
            if returns.width != 1:
                raise ValueError("If returns is a DataFrame, it must have only one column")
            returns_np = returns.to_numpy().flatten()
        else:
            returns_np = np.asarray(returns)

        # Remove NaN values
        returns_np = returns_np[~np.isnan(returns_np)]

        # Clip extreme values more aggressively 
        std_dev = np.std(returns_np)
        clip_threshold = 5 * std_dev  # More aggressive clipping (changed from 10*std_dev)
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)

        # Demean returns
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean

        # Initial parameters
        initial_params = [self.omega, self.alpha, self.beta, self.gamma]

        # Set optimization options with more robust settings
        options = {
            'maxiter': max_iter,
            'ftol': 1e-8,         # Tighter function tolerance
            'disp': False         # Don't display convergence messages
        }

        # Define parameter bounds more carefully
        # (omega, alpha, beta, gamma)
        bounds = [
            (1e-7, 1e-3),         # Omega: very small positive values
            (0.01, 0.2),          # Alpha: small positive values
            (0.7, 0.98),          # Beta: high persistence but <1
            (0.01, 0.2)           # Gamma: small positive values
        ]

        # Ensure bounds satisfy GARCH stationarity constraint
        # alpha + beta + 0.5*gamma < 1

        # Try multiple optimization methods if first fails
        methods_to_try = [method, 'SLSQP', 'L-BFGS-B', 'trust-constr']
        converged = False

        for current_method in methods_to_try:
            try:
                result = minimize(
                    self._neg_log_likelihood,
                    initial_params,
                    args=(returns_centered,),
                    method=current_method,
                    bounds=bounds,
                    options=options
                )

                if result.success:
                    self.omega, self.alpha, self.beta, self.gamma = result.x
                    converged = True
                    break
                else:
                    print(f"Warning: {current_method} did not converge: {result.message}")
            except Exception as e:
                print(f"Error with {current_method} method: {e}")

        if not converged:
            print("All optimization methods failed. Using refined initial parameters...")
            # Use more typical values if optimization failed
            self.omega = 1e-5
            self.alpha = 0.05
            self.beta = 0.85
            self.gamma = 0.1

        # Calculate fitted variance series
        T = len(returns_centered)
        sigma2 = np.zeros(T)
        sigma2[0] = max(1e-6, np.var(returns_centered))

        for t in range(1, T):
            # Indicator for negative return
            I_t_minus_1 = 1.0 if returns_centered[t-1] < 0 else 0.0
            sigma2[t] = (self.omega + self.alpha * returns_centered[t-1]**2 + 
                         self.beta * sigma2[t-1] + 
                         self.gamma * I_t_minus_1 * returns_centered[t-1]**2)

            # Add safety floor
            sigma2[t] = max(1e-6, sigma2[t])

        # Store history
        self.variance_history = sigma2
        self.residuals_history = returns_centered
        self.sigma2_t = sigma2[-1]  # Most recent variance
        self.is_fitted = True

        return self
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        """
        Forecast volatility n steps ahead.
        
        For GJR-GARCH, we need to account for the expected value of the asymmetry term.
        For a zero-mean process, this expectation is approximately 0.5*gamma.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to forecast
            
        Returns:
        --------
        forecasts : np.ndarray
            Array of variance forecasts
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # For one-step ahead, we know the sign of the last return
        I_last = 1.0 if self.residuals_history[-1] < 0 else 0.0
        
        forecasts = np.zeros(n_steps)
        forecasts[0] = (self.omega + 
                       self.alpha * self.residuals_history[-1]**2 + 
                       self.beta * self.sigma2_t + 
                       self.gamma * I_last * self.residuals_history[-1]**2)
        
        # Multi-step forecasts - approximation using expected value of I_t
        # For zero-mean process, E[I_t] = 0.5
        for h in range(1, n_steps):
            forecasts[h] = self.omega + (self.alpha + 0.5 * self.gamma) * forecasts[h-1] + self.beta * forecasts[h-1]
        
        return forecasts


class ThreePhaseVolatilityModel:
    """
    Three-phase volatility model for event studies as described in the paper.
    
    This model combines a baseline GARCH or GJR-GARCH process with event-specific
    volatility dynamics including pre-event rise, post-event rising phase, and 
    post-event decay.
    """
    def __init__(self, baseline_model: Union[GARCHModel, GJRGARCHModel],
                 k1: float = 1.5, k2: float = 2.0, 
                 delta_t1: float = 5.0, delta_t2: float = 3.0, delta_t3: float = 10.0, 
                 delta: int = 5):
        """
        Initialize three-phase volatility model.
        
        Parameters:
        -----------
        baseline_model : GARCHModel or GJRGARCHModel
            The base volatility model for estimation
        k1 : float
            Pre-event volatility multiplier (peak at event)
        k2 : float
            Post-event volatility multiplier (peak after event)
        delta_t1 : float
            Pre-event volatility duration parameter
        delta_t2 : float
            Post-event rising phase rate parameter
        delta_t3 : float
            Post-event decay rate parameter
        delta : int
            Duration of post-event rising phase in days
        """
        self.baseline_model = baseline_model
        self.k1 = k1
        self.k2 = k2
        self.delta_t1 = delta_t1
        self.delta_t2 = delta_t2
        self.delta_t3 = delta_t3
        self.delta = delta
        
        if k1 <= 1:
            raise ValueError("k1 must be greater than 1")
        if k2 <= 1:
            raise ValueError("k2 must be greater than 1")
        if k2 <= k1:
            print("Warning: Typically k2 > k1 to reflect higher post-event peak due to secondary uncertainties")
    
    def phi1(self, t: int, t_event: int) -> float:
        """
        Pre-event rise function.
        Peaks at t_event with value (k1-1).
        
        phi1(t) = (k1-1) * exp(-((t - t_event)^2) / (2 * delta_t1^2))
        """
        return (self.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * self.delta_t1**2))
    
    def phi2(self, t: int, t_event: int) -> float:
        """
        Post-event rising phase function.
        Starts at 0 at t_event, rises to (k2-1).
        
        phi2(t) = (k2-1) * (1 - exp(-(t - t_event) / delta_t2))
        """
        return (self.k2 - 1) * (1 - np.exp(-(t - t_event) / self.delta_t2))
    
    def phi3(self, t: int, t_event: int) -> float:
        """
        Post-event decay function.
        Starts at (k2-1) at t_event + delta, decays to 0.
        
        phi3(t) = (k2-1) * exp(-(t - (t_event + delta)) / delta_t3)
        """
        return (self.k2 - 1) * np.exp(-(t - (t_event + self.delta)) / self.delta_t3)
    
    def calculate_volatility(self, t: int, t_event: int, sigma_e0: float) -> float:
        """
        Calculate volatility at time t relative to event at t_event.
        
        Parameters:
        -----------
        t : int
            Current time point
        t_event : int
            Event time point
        sigma_e0 : float
            Baseline volatility (from GARCH model)
            
        Returns:
        --------
        sigma_e : float
            Volatility adjusted for event effect
        """
        if t <= t_event:
            # Pre-event rise
            phi = self.phi1(t, t_event)
        elif t <= t_event + self.delta:
            # Post-event rising phase
            phi = self.phi2(t, t_event)
        else:
            # Post-event decay
            phi = self.phi3(t, t_event)
        
        # Adjust baseline volatility
        sigma_e = sigma_e0 * (1 + phi)
        
        return sigma_e
    
    def calculate_volatility_series(self, 
                                   days_to_event: Union[List[int], np.ndarray], 
                                   baseline_volatility: Optional[float] = None) -> np.ndarray:
        """
        Calculate volatility for a series of days relative to the event.
        
        Parameters:
        -----------
        days_to_event : array-like
            Array of days relative to event (e.g., -30 to +30)
        baseline_volatility : float, optional
            Baseline volatility. If None, use the fitted GARCH model's unconditional volatility
            
        Returns:
        --------
        volatility_series : np.ndarray
            Array of volatilities for each day
        """
        if not isinstance(days_to_event, np.ndarray):
            days_to_event = np.array(days_to_event)
        
        # Get baseline volatility if not provided
        if baseline_volatility is None:
            if not self.baseline_model.is_fitted:
                raise RuntimeError("Baseline model must be fitted or baseline_volatility provided")
            # Use unconditional volatility as baseline
            sigma_e0 = np.sqrt(self.baseline_model.omega / (1 - self.baseline_model.alpha - self.baseline_model.beta))
        else:
            sigma_e0 = baseline_volatility
        
        # Calculate volatility for each day
        volatility_series = np.zeros_like(days_to_event, dtype=float)
        t_event = 0  # Event is at day 0 in days_to_event
        
        for i, t in enumerate(days_to_event):
            volatility_series[i] = self.calculate_volatility(t, t_event, sigma_e0)
        
        return volatility_series