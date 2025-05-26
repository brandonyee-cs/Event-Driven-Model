import numpy as np
import polars as pl 
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy.optimize import minimize
import warnings
import pandas as pd # For pd.notna in GARCH/GJR for now
from typing import Tuple, List, Dict, Optional, Union


pl.Config.set_engine_affinity(engine="streaming")

class TimeSeriesRidge(Ridge):
    """
    Ridge regression with temporal smoothing penalty.
    The model minimizes: ||y - Xβ||² + α||β||² + λ₂||Dβ||²
    """
    def __init__(self, alpha=1.0, lambda2=0.1, feature_order=None, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.lambda2 = lambda2
        self.feature_order = feature_order 
        self.feature_names_in_ = None 

    def _get_differencing_matrix(self, n_features):
        if n_features <= 1:
            return np.zeros((0, n_features))
        D = np.zeros((n_features - 1, n_features))
        for i in range(n_features - 1):
            D[i, i] = 1
            D[i, i + 1] = -1
        return D

    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame.")
        if isinstance(y, pl.Series):
            y_np = y.to_numpy()
        elif isinstance(y, np.ndarray):
            y_np = y
        else:
            raise TypeError("y must be a Polars Series or NumPy array.")

        original_X_columns = X.columns 

        if self.feature_order is not None:
             missing_features = set(self.feature_order) - set(X.columns)
             if missing_features:
                 raise ValueError(f"Features specified in feature_order are missing from X: {missing_features}")
             extra_features = set(X.columns) - set(self.feature_order)
             if extra_features:
                 ordered_cols = self.feature_order + list(extra_features)
                 X_ordered = X.select(ordered_cols) 
             else:
                 X_ordered = X.select(self.feature_order)
             self.feature_names_in_ = X_ordered.columns 
        else:
             X_ordered = X 
             self.feature_names_in_ = X_ordered.columns

        try:
            numeric_cols = X_ordered.columns
            X_np = X_ordered.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols]
            ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert Polars DataFrame X to NumPy: {e}. Check dtypes.")

        y_np = np.asarray(y_np, dtype=np.float64)
        n_samples, n_features = X_np.shape

        if np.isnan(X_np).any() or np.isinf(X_np).any():
             nan_cols = [self.feature_names_in_[i] for i in np.where(np.isnan(X_np).any(axis=0))[0]]
             inf_cols = [self.feature_names_in_[i] for i in np.where(np.isinf(X_np).any(axis=0))[0]]
             warnings.warn(f"NaN or Inf values detected in feature matrix X before fitting. NaN cols: {nan_cols}, Inf cols: {inf_cols}. Impute first.")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
             raise ValueError("NaN or Inf values detected in target vector y before fitting.")

        D = self._get_differencing_matrix(n_features)

        if D.shape[0] == 0 or self.lambda2 == 0:
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))
            ridge_model.fit(X_np, y_np, sample_weight)
            self.coef_ = ridge_model.coef_
            self.intercept_ = ridge_model.intercept_
            return self

        sqrt_lambda2_D = np.sqrt(self.lambda2) * D
        X_augmented = np.vstack([X_np, sqrt_lambda2_D])
        y_augmented = np.concatenate([y_np, np.zeros(D.shape[0])])
        ridge_solver = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))

        if sample_weight is not None:
             augmented_weights = np.concatenate([sample_weight, np.ones(D.shape[0])])
             ridge_solver.fit(X_augmented, y_augmented, sample_weight=augmented_weights)
        else:
             ridge_solver.fit(X_augmented, y_augmented)

        self.coef_ = ridge_solver.coef_
        self.intercept_ = ridge_solver.intercept_

        if self.feature_order is not None:
            coef_dict = dict(zip(self.feature_names_in_, self.coef_))
            original_order_coef = [coef_dict.get(col, 0) for col in original_X_columns]
            self.coef_ = np.array(original_order_coef)
            self.feature_names_in_ = original_X_columns
        return self

    def predict(self, X):
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame for prediction.")
        if self.feature_names_in_ is None:
            raise RuntimeError("Model not fitted or feature names not stored.")

        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Prediction data missing columns used during fit: {missing_cols}")
        X_ordered = X.select(self.feature_names_in_)
        try:
            X_np = X_ordered.select(
                [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
            ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert prediction Polars DataFrame X to NumPy: {e}. Check dtypes.")
        return super().predict(X_np)


class XGBoostDecileModel:
    def __init__(self, weight=0.5, momentum_feature='momentum_5', n_deciles=10,
                 alpha=0.1, lambda_smooth=0.1, xgb_params=None, ts_ridge_feature_order=None):
        if not 0 <= weight <= 1:
             raise ValueError("Weight must be between 0 and 1.")
        self.weight = weight
        self.momentum_feature = momentum_feature
        self.n_deciles = n_deciles
        self.alpha = alpha 
        self.lambda_smooth = lambda_smooth 
        self.ts_ridge_feature_order = ts_ridge_feature_order 

        if xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
                'objective': 'reg:squarederror', 'subsample': 0.8, 'colsample_bytree': 0.8,
                'random_state': 42, 'n_jobs': -1
            }
        else:
            self.xgb_params = xgb_params

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.decile_models = [None] * n_deciles
        self.decile_boundaries = None 
        self.feature_names_in_ = None 

    def _calculate_decile_boundaries(self, X: pl.DataFrame):
        if self.momentum_feature not in X.columns:
             raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in input data.")
        quantiles_pl = X.select(
            pl.col(self.momentum_feature).drop_nulls().quantile(q).alias(f"q_{q}")
            for q in np.linspace(0, 1, self.n_deciles + 1)[1:-1] 
        )
        if quantiles_pl.is_empty() or quantiles_pl.height == 0: 
             raise ValueError("Could not calculate decile boundaries, possibly due to all-null momentum feature.")
        quantiles_values = quantiles_pl.row(0) 
        self.decile_boundaries = np.array(quantiles_values, dtype=np.float64)

    def _assign_deciles(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.decile_boundaries is None:
             raise RuntimeError("Decile boundaries have not been calculated. Call fit first.")
        if self.momentum_feature not in X.columns:
             raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in input data.")
        if len(self.decile_boundaries) == 0: 
            return X.with_columns(pl.lit(0, dtype=pl.Int32).alias("decile_assignment"))

        mom_col = pl.col(self.momentum_feature)
        boundaries = self.decile_boundaries 
        decile_expr = pl.when(mom_col.is_nan()).then(pl.lit(0, dtype=pl.Int32))
        decile_expr = decile_expr.when(mom_col < boundaries[0]).then(pl.lit(0, dtype=pl.Int32))
        for i in range(len(boundaries) - 1):
            decile_expr = decile_expr.when(
                (mom_col >= boundaries[i]) & (mom_col < boundaries[i+1])
            ).then(pl.lit(i + 1, dtype=pl.Int32))
        decile_expr = decile_expr.when(mom_col >= boundaries[-1]).then(pl.lit(self.n_deciles - 1, dtype=pl.Int32))
        decile_expr = decile_expr.otherwise(pl.lit(0, dtype=pl.Int32)) 
        return X.with_columns(decile_expr.alias("decile_assignment"))

    def fit(self, X: pl.DataFrame, y: pl.Series):
        if not isinstance(X, pl.DataFrame): raise TypeError("X must be a Polars DataFrame.")
        if not isinstance(y, pl.Series): raise TypeError("y must be a Polars Series.")
        if X.height != y.height: raise ValueError("X and y must have the same height.")
        self.feature_names_in_ = X.columns 
        try:
            X_np = X.select( 
                [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
            ).to_numpy()
            y_np = y.cast(pl.Float64).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert Polars data to NumPy for fitting: {e}")

        if np.isnan(X_np).any() or np.isinf(X_np).any():
             warnings.warn("NaNs or Infs detected in X_np before fitting XGBoostDecile.")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
             raise ValueError("NaN or Inf values detected in target vector y before fitting.")
        try:
            fit_params = {}
            if 'early_stopping_rounds' in self.xgb_params and 'eval_set' in self.xgb_params : 
                 fit_params['eval_set'] = self.xgb_params['eval_set']
                 fit_params['early_stopping_rounds'] = self.xgb_params['early_stopping_rounds']
                 fit_params['verbose'] = self.xgb_params.get('verbose', False) 
            self.xgb_model.fit(X_np, y_np, **fit_params)
        except TypeError as e:
             if "unexpected keyword argument 'early_stopping_rounds'" in str(e) or \
                "got multiple values for keyword argument 'verbose'" in str(e) or \
                "missing 1 required positional argument: 'eval_set'" in str(e): 
                 warnings.warn(f"XGBoost parameter issue. Retrying without it. Original error: {e}")
                 xgb_params_fallback = self.xgb_params.copy()
                 xgb_params_fallback.pop('early_stopping_rounds', None)
                 xgb_params_fallback.pop('eval_set', None) 
                 self.xgb_model = xgb.XGBRegressor(**xgb_params_fallback)
                 self.xgb_model.fit(X_np, y_np)
             else: raise e
        except Exception as e: print(f"Error during XGBoost fit: {e}"); raise e

        if self.weight < 1.0:
             self._calculate_decile_boundaries(X)
             X_with_deciles = self._assign_deciles(X)
             for d in range(self.n_deciles):
                 decile_mask = pl.col("decile_assignment") == d
                 X_decile_pl = X_with_deciles.filter(decile_mask)
                 y_decile_pl = y.filter(X_with_deciles.select(decile_mask).to_series()) 
                 min_samples_required = max(5, len(self.feature_names_in_) + 1 if self.feature_names_in_ else 5)
                 if X_decile_pl.height >= min_samples_required:
                     try:
                         decile_model = TimeSeriesRidge(
                             alpha=self.alpha,
                             lambda2=self.lambda_smooth,
                             feature_order=self.ts_ridge_feature_order 
                         )
                         decile_model.fit(X_decile_pl.drop("decile_assignment"), y_decile_pl)
                         self.decile_models[d] = decile_model
                     except Exception as e:
                         warnings.warn(f"Warning: Failed to fit model for Decile {d+1}. Reason: {e}")
                         self.decile_models[d] = None 
                 else:
                     self.decile_models[d] = None
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.feature_names_in_ is None:
             raise RuntimeError("Model has not been fitted yet. Call fit first.")
        if not isinstance(X, pl.DataFrame):
             raise TypeError("X must be a Polars DataFrame for prediction.")
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
             raise ValueError(f"Missing columns in prediction data: {missing_cols}")
        X_pred = X.select(self.feature_names_in_)
        try:
             X_np = X_pred.select( 
                 [pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names_in_]
             ).to_numpy()
        except Exception as e:
             raise ValueError(f"Failed to convert prediction Polars DataFrame to NumPy: {e}")
        xgb_preds = self.xgb_model.predict(X_np) 
        if self.weight < 1.0:
             if self.decile_boundaries is None:
                 raise RuntimeError("Decile boundaries not set. Model needs fitting.")
             decile_preds_np = np.zeros_like(xgb_preds)
             X_with_deciles = self._assign_deciles(X_pred) 
             for d in range(self.n_deciles):
                 bool_mask_np = X_with_deciles.get_column("decile_assignment").eq(d).to_numpy()
                 if np.any(bool_mask_np): 
                     X_decile_test_pl = X_pred.filter(bool_mask_np)
                     if self.decile_models[d] is not None:
                         try:
                             preds_d = self.decile_models[d].predict(X_decile_test_pl) 
                             decile_preds_np[bool_mask_np] = preds_d
                         except Exception as e:
                             warnings.warn(f"Warning: Error predicting with model for Decile {d+1}. Error: {e}")
                             decile_preds_np[bool_mask_np] = xgb_preds[bool_mask_np] 
                     else:
                         decile_preds_np[bool_mask_np] = xgb_preds[bool_mask_np] 
             ensemble_preds = self.weight * xgb_preds + (1 - self.weight) * decile_preds_np
        else:
             ensemble_preds = xgb_preds
        return ensemble_preds

class GARCHModel:
    def __init__(self, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85):
        self.omega_init = omega # Store initial for potential fallback
        self.alpha_init = alpha
        self.beta_init = beta
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.is_fitted = False
        self.variance_history = None
        self.residuals_history = None
        self.sigma2_t = None # Last estimated variance
        self.mean = 0.0
        self.fit_success = False # Track fit success

    def _check_parameters(self, omega, alpha, beta, context="final"):
        valid = True
        if omega <= 1e-8: # Slightly more tolerant than 0 for optimizer
            if context=="final": warnings.warn(f"GARCH omega is non-positive ({omega:.2e}). Resetting to small positive.")
            omega = 1e-7
            valid = False
        if alpha < 0:
            if context=="final": warnings.warn(f"GARCH alpha is negative ({alpha:.2e}). Resetting to 0.")
            alpha = 0.0
            valid = False
        if beta < 0:
            if context=="final": warnings.warn(f"GARCH beta is negative ({beta:.2e}). Resetting to 0.")
            beta = 0.0
            valid = False
        if alpha + beta >= 0.99999: # Check for near non-stationarity
            if context=="final": warnings.warn(f"GARCH alpha+beta ({alpha+beta:.3f}) >= 1. May be non-stationary. Clamping sum.")
            # Heuristic to clamp: reduce beta first if alpha is substantial, else scale both
            if alpha > 0.01: # if alpha has some value
                beta = 0.9999 - alpha - 0.0001 # ensure alpha + beta < 1
                if beta < 0: beta = 0 # if alpha was too large
            else: # if alpha is tiny, scale both down proportionally
                total = alpha + beta
                alpha = alpha / (total + 1e-4) * 0.9998
                beta = beta / (total + 1e-4) * 0.9998
            valid = False
        return omega, alpha, beta, valid

    def _neg_log_likelihood(self, params, returns_centered):
        omega, alpha, beta = params
        omega, alpha, beta, _ = self._check_parameters(omega, alpha, beta, context="likelihood")
        if alpha + beta >= 0.99999: # Strict check within likelihood
            return np.inf

        T = len(returns_centered)
        sigma2 = np.zeros(T)
        
        # Initial variance: try unconditional, then empirical, then small constant
        if (1 - alpha - beta) > 1e-7:
            sigma2_0_uncond = omega / (1 - alpha - beta)
            sigma2[0] = max(1e-8, sigma2_0_uncond)
        else:
            sigma2[0] = max(1e-8, np.var(returns_centered))
        if sigma2[0] == 0: sigma2[0] = 1e-8 # Final fallback for zero variance series

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns_centered[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(1e-8, sigma2[t]) # Floor variance

        if np.any(sigma2 <= 1e-9): return np.inf # Ensure positive variance for log
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns_centered**2 / sigma2)
        
        return -log_likelihood if np.isfinite(log_likelihood) else np.inf
    
    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'L-BFGS-B', # Changed default
            max_iter: int = 200) -> 'GARCHModel': # Reduced max_iter for speed
        if isinstance(returns, pl.Series): returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame): returns_np = returns.to_numpy().flatten()
        else: returns_np = np.asarray(returns)

        returns_np = returns_np[~np.isnan(returns_np)]
        if len(returns_np) < 20: # Increased min length
            warnings.warn(f"GARCH: Not enough data points ({len(returns_np)}). Using initial parameters.")
            self._use_initial_params_for_history(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7: # More tolerant for very low vol series
            warnings.warn("GARCH: Return series has very low variance. Using simplified variance.")
            self.mean = np.mean(returns_np)
            self.variance_history = np.full(len(returns_np), max(1e-8, std_dev**2))
            self.residuals_history = returns_np - self.mean
            self.sigma2_t = self.variance_history[-1] if len(self.variance_history) > 0 else 1e-7
            self.is_fitted = True; self.fit_success = True # Technically not fitted by MLE but has values
            self.omega, self.alpha, self.beta, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init)
            if (1-self.alpha-self.beta) > 1e-7 : self.omega = max(1e-8, std_dev**2) * (1 - self.alpha - self.beta)
            else: self.omega = 1e-7 # Fallback omega if params non-stationary
            return self

        clip_threshold = 7 * std_dev if std_dev > 1e-7 else 0.07 # Tighter clipping
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean

        initial_params = [self.omega_init, self.alpha_init, self.beta_init]
        bounds = [(1e-8, 0.1), (1e-8, 0.99), (1e-8, 0.99)] # omega, alpha, beta bounds

        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9}
        
        # Try L-BFGS-B first as it's often good for GARCH
        methods_to_try = [method, 'SLSQP'] 
        final_result = None

        for opt_method in methods_to_try:
            try:
                current_bounds = bounds if opt_method != 'Nelder-Mead' else None # Nelder-Mead doesn't use bounds
                result = minimize(self._neg_log_likelihood, initial_params, args=(returns_centered,),
                                  method=opt_method, bounds=current_bounds, options=optimizer_options)
                
                if result.success:
                    omega_fit, alpha_fit, beta_fit = result.x
                    _, _, _, params_valid = self._check_parameters(omega_fit, alpha_fit, beta_fit, context="fit_check")
                    if params_valid and (alpha_fit + beta_fit < 0.9999): # Final check on stationarity
                        self.omega, self.alpha, self.beta = omega_fit, alpha_fit, beta_fit
                        self.fit_success = True
                        break 
                final_result = result # Store last result even if not successful for warning
            except Exception: # Catches errors within minimize too
                final_result = None # Mark that this method failed
                continue # Try next method
        
        if not self.fit_success:
            msg = final_result.message if final_result else "Optimization error"
            warnings.warn(f"GARCH optimization failed ({msg}). Using robust initial parameters.")
            self.omega, self.alpha, self.beta, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init)

        self._finalize_fit(returns_centered)
        return self

    def _use_initial_params_for_history(self, returns_np):
        self.omega, self.alpha, self.beta, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init)
        self.mean = np.mean(returns_np) if len(returns_np) > 0 else 0.0
        returns_centered = returns_np - self.mean
        self._finalize_fit(returns_centered, use_empirical_var_for_sigma2_0=True)

    def _finalize_fit(self, returns_centered, use_empirical_var_for_sigma2_0=False):
        T = len(returns_centered)
        if T == 0:
            self.variance_history = np.array([])
            self.residuals_history = np.array([])
            self.sigma2_t = self.omega / (1-self.alpha-self.beta) if (1-self.alpha-self.beta) > 1e-7 else 1e-7
            self.is_fitted = True
            return

        sigma2 = np.zeros(T)
        if use_empirical_var_for_sigma2_0 or (1 - self.alpha - self.beta) <= 1e-7:
             sigma2[0] = max(1e-8, np.var(returns_centered)) if T > 1 else 1e-7
        else:
             sigma2[0] = max(1e-8, self.omega / (1 - self.alpha - self.beta))
        if sigma2[0] == 0: sigma2[0] = 1e-8

        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * returns_centered[t-1]**2 + self.beta * sigma2[t-1]
            sigma2[t] = max(1e-8, sigma2[t])
        
        self.variance_history = sigma2
        self.residuals_history = returns_centered
        self.sigma2_t = sigma2[-1]
        self.is_fitted = True
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Model must be fitted")
        if self.residuals_history is None or len(self.residuals_history) == 0 or self.sigma2_t is None:
            uncond_var_denom = (1 - self.alpha - self.beta)
            uncond_var = self.omega / uncond_var_denom if uncond_var_denom > 1e-7 else 1e-7
            return np.full(n_steps, max(1e-8, uncond_var))

        forecasts = np.zeros(n_steps)
        last_resid_sq = self.residuals_history[-1]**2
        current_sigma2 = self.sigma2_t
        for h in range(n_steps):
            forecasts[h] = self.omega + self.alpha * last_resid_sq + self.beta * current_sigma2 if h == 0 else \
                           self.omega + (self.alpha + self.beta) * forecasts[h-1]
            forecasts[h] = max(1e-8, forecasts[h])
        return forecasts
    
    def conditional_volatility(self) -> np.ndarray:
        if not self.is_fitted or self.variance_history is None: raise RuntimeError("Model not fitted")
        return np.sqrt(self.variance_history)
    
    def volatility_innovations(self) -> np.ndarray:
        if not self.is_fitted or self.variance_history is None or len(self.variance_history) <= 1: return np.array([])
        T = len(self.variance_history)
        innovations = np.zeros(T-1)
        for t in range(1, T):
            expected_var_t = self.omega + self.alpha * self.residuals_history[t-1]**2 + self.beta * self.variance_history[t-1]
            realized_var_t = self.variance_history[t]
            innovations[t-1] = realized_var_t - expected_var_t
        if len(innovations) > 0 and (np.allclose(innovations,0) or np.var(innovations) < 1e-12):
            innovations = innovations + np.random.normal(0, 1e-7, size=len(innovations))
        return innovations

class GJRGARCHModel(GARCHModel):
    def __init__(self, omega: float = 1e-6, alpha: float = 0.08, beta: float = 0.85, gamma: float = 0.05):
        super().__init__(omega, alpha, beta)
        self.gamma_init = gamma # Store initial for fallback
        self.gamma = gamma
        self.fit_success = False # Reset for GJR

    def _check_gjr_parameters(self, omega, alpha, beta, gamma, context="final"):
        omega, alpha, beta, garch_valid = self._check_parameters(omega, alpha, beta, context) # Validate GARCH part
        valid = garch_valid
        if gamma < 0:
            if context=="final": warnings.warn(f"GJR gamma is negative ({gamma:.2e}). Resetting to 0.")
            gamma = 0.0
            valid = False
        # Stationarity for GJR: alpha + beta + 0.5*gamma < 1
        if alpha + beta + 0.5 * gamma >= 0.99999:
            if context=="final": warnings.warn(f"GJR sum condition violated ({alpha+beta+0.5*gamma:.3f}). Clamping.")
            # Complex to clamp perfectly, heuristic: if gamma pushes it over, reduce gamma
            # If still over, then reduce beta/alpha as in GARCHModel
            required_sum_alpha_beta = 0.9999 - 0.5 * gamma - 0.0001
            if alpha + beta > required_sum_alpha_beta:
                if gamma > 0.01: # if gamma is somewhat substantial, try reducing it
                    gamma = max(0, (0.9999 - (alpha+beta) - 0.0001) * 2)
                else: # if gamma is small, likely alpha/beta are too large
                    current_sum_alpha_beta = alpha + beta
                    alpha = alpha / (current_sum_alpha_beta + 1e-4) * required_sum_alpha_beta
                    beta = beta / (current_sum_alpha_beta + 1e-4) * required_sum_alpha_beta
            valid = False
        return omega, alpha, beta, gamma, valid

    def _neg_log_likelihood(self, params, returns_centered): # Override for GJR
        omega, alpha, beta, gamma = params
        omega, alpha, beta, gamma, _ = self._check_gjr_parameters(omega, alpha, beta, gamma, context="likelihood")
        if alpha + beta + 0.5 * gamma >= 0.99999: # Strict check
            return np.inf

        T = len(returns_centered)
        sigma2 = np.zeros(T)
        
        uncond_denom = (1 - alpha - beta - 0.5 * gamma)
        if uncond_denom > 1e-7:
            sigma2_0_uncond = omega / uncond_denom
            sigma2[0] = max(1e-8, sigma2_0_uncond)
        else:
            sigma2[0] = max(1e-8, np.var(returns_centered))
        if sigma2[0] == 0: sigma2[0] = 1e-8

        for t in range(1, T):
            I_tm1 = 1.0 if returns_centered[t-1] < 0 else 0.0
            sigma2[t] = omega + alpha * returns_centered[t-1]**2 + \
                        beta * sigma2[t-1] + gamma * I_tm1 * returns_centered[t-1]**2
            sigma2[t] = max(1e-8, sigma2[t])

        if np.any(sigma2 <= 1e-9): return np.inf
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns_centered**2 / sigma2)
        return -log_likelihood if np.isfinite(log_likelihood) else np.inf

    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'L-BFGS-B', max_iter: int = 200) -> 'GJRGARCHModel':
        if isinstance(returns, pl.Series): returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame): returns_np = returns.to_numpy().flatten()
        else: returns_np = np.asarray(returns)

        returns_np = returns_np[~np.isnan(returns_np)]
        if len(returns_np) < 20:
            warnings.warn(f"GJR: Not enough data points ({len(returns_np)}). Using initial parameters.")
            self._use_initial_params_for_history_gjr(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7:
            warnings.warn("GJR: Return series has very low variance. Using simplified variance.")
            self.mean = np.mean(returns_np)
            self.variance_history = np.full(len(returns_np), max(1e-8, std_dev**2))
            self.residuals_history = returns_np - self.mean
            self.sigma2_t = self.variance_history[-1] if len(self.variance_history) > 0 else 1e-7
            self.is_fitted = True; self.fit_success = True
            self.omega,self.alpha,self.beta,self.gamma,_ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init)
            denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)
            if denom > 1e-7: self.omega = max(1e-8, std_dev**2) * denom
            else: self.omega = 1e-7
            return self
            
        clip_threshold = 7 * std_dev if std_dev > 1e-7 else 0.07
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean

        initial_params = [self.omega_init, self.alpha_init, self.beta_init, self.gamma_init]
        bounds = [(1e-8, 0.1), (1e-8, 0.99), (1e-8, 0.99), (0, 0.99)] # omega, alpha, beta, gamma

        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9}
        methods_to_try = [method, 'SLSQP'] 
        final_result = None

        for opt_method in methods_to_try:
            try:
                current_bounds = bounds if opt_method != 'Nelder-Mead' else None
                result = minimize(self._neg_log_likelihood, initial_params, args=(returns_centered,),
                                  method=opt_method, bounds=current_bounds, options=optimizer_options)
                if result.success:
                    o, a, b, g = result.x
                    _, _, _, _, params_valid = self._check_gjr_parameters(o, a, b, g, context="fit_check")
                    if params_valid and (a + b + 0.5 * g < 0.9999):
                        self.omega, self.alpha, self.beta, self.gamma = o,a,b,g
                        self.fit_success = True
                        break
                final_result = result
            except Exception:
                final_result = None; continue
        
        if not self.fit_success:
            msg = final_result.message if final_result else "Optimization error"
            warnings.warn(f"GJR-GARCH optimization failed ({msg}). Using robust initial parameters.")
            self.omega,self.alpha,self.beta,self.gamma,_ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init)

        self._finalize_fit_gjr(returns_centered) # Use GJR specific finalization
        return self

    def _use_initial_params_for_history_gjr(self, returns_np):
        self.omega,self.alpha,self.beta,self.gamma,_ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init)
        self.mean = np.mean(returns_np) if len(returns_np) > 0 else 0.0
        returns_centered = returns_np - self.mean
        self._finalize_fit_gjr(returns_centered, use_empirical_var_for_sigma2_0=True)

    def _finalize_fit_gjr(self, returns_centered, use_empirical_var_for_sigma2_0=False):
        T = len(returns_centered)
        if T == 0:
            self.variance_history = np.array([])
            self.residuals_history = np.array([])
            uncond_denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)
            self.sigma2_t = self.omega / uncond_denom if uncond_denom > 1e-7 else 1e-7
            self.is_fitted = True
            return

        sigma2 = np.zeros(T)
        uncond_denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)
        if use_empirical_var_for_sigma2_0 or uncond_denom <= 1e-7:
            sigma2[0] = max(1e-8, np.var(returns_centered)) if T > 1 else 1e-7
        else:
            sigma2[0] = max(1e-8, self.omega / uncond_denom)
        if sigma2[0] == 0: sigma2[0] = 1e-8
        
        for t in range(1, T):
            I_tm1 = 1.0 if returns_centered[t-1] < 0 else 0.0
            sigma2[t] = self.omega + self.alpha * returns_centered[t-1]**2 + \
                        self.beta * sigma2[t-1] + self.gamma * I_tm1 * returns_centered[t-1]**2
            sigma2[t] = max(1e-8, sigma2[t])

        self.variance_history = sigma2
        self.residuals_history = returns_centered
        self.sigma2_t = sigma2[-1]
        self.is_fitted = True
    
    def predict(self, n_steps: int = 1) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Model must be fitted")
        if self.residuals_history is None or len(self.residuals_history) == 0 or self.sigma2_t is None:
            uncond_var_denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)
            uncond_var = self.omega / uncond_var_denom if uncond_var_denom > 1e-7 else 1e-7
            return np.full(n_steps, max(1e-8, uncond_var))

        forecasts = np.zeros(n_steps)
        last_resid_sq = self.residuals_history[-1]**2
        I_last = 1.0 if self.residuals_history[-1] < 0 else 0.0
        current_sigma2 = self.sigma2_t
        for h in range(n_steps):
            if h == 0:
                forecasts[h] = self.omega + self.alpha * last_resid_sq + \
                               self.beta * current_sigma2 + self.gamma * I_last * last_resid_sq
            else: # E[I*resid^2] = 0.5 * E[resid^2] = 0.5 * forecast[h-1]
                forecasts[h] = self.omega + (self.alpha + self.beta + 0.5 * self.gamma) * forecasts[h-1]
            forecasts[h] = max(1e-8, forecasts[h])
        return forecasts

    def volatility_innovations(self) -> np.ndarray:
        if not self.is_fitted or self.variance_history is None or len(self.variance_history) <= 1: return np.array([])
        T = len(self.variance_history)
        innovations = np.zeros(T-1)
        for t in range(1, T):
            I_tm1 = 1.0 if self.residuals_history[t-1] < 0 else 0.0
            expected_var_t = self.omega + self.alpha * self.residuals_history[t-1]**2 + \
                             self.beta * self.variance_history[t-1] + \
                             self.gamma * I_tm1 * self.residuals_history[t-1]**2
            realized_var_t = self.variance_history[t]
            innovations[t-1] = realized_var_t - expected_var_t
        if len(innovations) > 0 and (np.allclose(innovations,0) or np.var(innovations) < 1e-12):
            innovations = innovations + np.random.normal(0, 1e-7, size=len(innovations))
        return innovations

class ThreePhaseVolatilityModel:
    def __init__(self, baseline_model: Union[GARCHModel, GJRGARCHModel],
                 k1: float = 1.5, k2: float = 2.0, 
                 delta_t1: float = 5.0, delta_t2: float = 3.0, delta_t3: float = 10.0, 
                 delta: int = 5):
        self.baseline_model = baseline_model
        self.k1 = k1
        self.k2 = k2
        self.delta_t1 = delta_t1
        self.delta_t2 = delta_t2
        self.delta_t3 = delta_t3
        self.delta = delta
        
        if k1 <= 1: raise ValueError("k1 must be greater than 1")
        if k2 <= 1: raise ValueError("k2 must be greater than 1")
        # if k2 <= k1: warnings.warn("Typically k2 > k1") # Keep less verbose
    
    def phi1(self, t: int, t_event: int) -> float:
        return (self.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * self.delta_t1**2))
    
    def phi2(self, t: int, t_event: int) -> float:
        # Ensure t - t_event is non-negative for exp argument to avoid overflow with large positive delta_t2
        # delta_t2 should be positive.
        time_diff = t - t_event
        if self.delta_t2 <= 1e-6 : return (self.k2 - 1) if time_diff > 0 else 0.0 # Avoid division by zero, assume instant rise
        return (self.k2 - 1) * (1 - np.exp(-time_diff / self.delta_t2))
    
    def phi3(self, t: int, t_event: int) -> float:
        time_diff = t - (t_event + self.delta)
        if self.delta_t3 <= 1e-6: return 0.0 # Avoid division by zero, assume instant decay
        return (self.k2 - 1) * np.exp(-time_diff / self.delta_t3)
    
    def calculate_volatility(self, t: int, t_event: int, sigma_e0: float) -> float:
        if sigma_e0 < 1e-8: sigma_e0 = 1e-8 # Floor baseline vol
        if t <= t_event:
            phi = self.phi1(t, t_event)
        elif t <= t_event + self.delta:
            phi = self.phi2(t, t_event)
        else:
            phi = self.phi3(t, t_event)
        return sigma_e0 * (1 + phi)
    
    def calculate_volatility_series(self, 
                                   days_to_event: Union[List[int], np.ndarray], 
                                   baseline_conditional_vol_series: Optional[np.ndarray] = None) -> np.ndarray:
        if not isinstance(days_to_event, np.ndarray):
            days_to_event = np.array(days_to_event, dtype=float) # Ensure float for calculations
        
        if baseline_conditional_vol_series is not None:
            if len(baseline_conditional_vol_series) != len(days_to_event):
                 raise ValueError(
                     f"Length of baseline_conditional_vol_series ({len(baseline_conditional_vol_series)}) "
                     f"must match days_to_event ({len(days_to_event)}). "
                     "Alignment error in calling function."
                 )
            sigma_e0_series = np.maximum(baseline_conditional_vol_series, 1e-8) # Ensure positive baseline
        else: 
            if not self.baseline_model.is_fitted:
                raise RuntimeError("Baseline model must be fitted or baseline_conditional_vol_series provided")
            
            bm = self.baseline_model
            if isinstance(bm, GJRGARCHModel):
                denominator = (1 - bm.alpha - bm.beta - 0.5 * bm.gamma)
            else: 
                denominator = (1 - bm.alpha - bm.beta)
            
            if denominator <= 1e-7: 
                if bm.variance_history is not None and len(bm.variance_history) > 0 and bm.variance_history[-1] > 1e-8 : # Check if last variance is usable
                    uncond_var = bm.variance_history[-1] 
                else: 
                    uncond_var = 1e-7 # Absolute fallback if history is also problematic
            else:
                 uncond_var = bm.omega / denominator
            
            sigma_e0_val = np.sqrt(max(uncond_var, 1e-8)) 
            sigma_e0_series = np.full_like(days_to_event, sigma_e0_val, dtype=float)

        volatility_series = np.zeros_like(days_to_event, dtype=float)
        t_event = 0.0 # Treat event day as 0.0 for continuous functions
        
        for i, t_rel in enumerate(days_to_event): 
            volatility_series[i] = self.calculate_volatility(float(t_rel), t_event, sigma_e0_series[i])
        
        return volatility_series
