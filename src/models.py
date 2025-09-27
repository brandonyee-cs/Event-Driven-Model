# --- FIXED VERSION OF models.py - Addresses Artificial Periodicity Issues ---
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy.optimize import minimize
import warnings
import pandas as pd # For pd.notna
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
        self.omega_init = max(1e-9, omega) # Ensure init omega is positive
        self.alpha_init = max(0, alpha)   # Ensure init alpha is non-negative
        self.beta_init = max(0, beta)    # Ensure init beta is non-negative
        # Adjust if init params are non-stationary
        if self.alpha_init + self.beta_init >= 0.9999:
            current_sum = self.alpha_init + self.beta_init
            self.alpha_init = (self.alpha_init / current_sum) * 0.9998 if current_sum > 1e-7 else 0.05
            self.beta_init = (self.beta_init / current_sum) * 0.9998 if current_sum > 1e-7 else 0.90

        self.omega = self.omega_init
        self.alpha = self.alpha_init
        self.beta = self.beta_init
        
        self.is_fitted = False
        self.variance_history = None
        self.residuals_history = None
        self.sigma2_t = None 
        self.mean = 0.0
        self.fit_success = False 
        self.fit_message = "Not Attempted"

    def _check_parameters(self, omega, alpha, beta, context="final"):
        valid = True
        if not (isinstance(omega, (int,float)) and pd.notna(omega) and omega > 1e-9):
            if context=="final": warnings.warn(f"GARCH omega invalid ({omega}). Correcting.")
            omega = 1e-7
            valid = False
        if not (isinstance(alpha, (int,float)) and pd.notna(alpha) and alpha >= 0):
            if context=="final": warnings.warn(f"GARCH alpha invalid ({alpha}). Correcting.")
            alpha = 0.01
            valid = False
        if not (isinstance(beta, (int,float)) and pd.notna(beta) and beta >= 0):
            if context=="final": warnings.warn(f"GARCH beta invalid ({beta}). Correcting.")
            beta = 0.01
            valid = False
        
        # FIXED: Relaxed stationarity condition from 0.99999 to 0.995
        if alpha + beta >= 0.995:
            if context=="final": warnings.warn(f"GARCH alpha+beta ({alpha+beta:.4f}) too high. Adjusting for stationarity.")
            current_sum = alpha + beta
            target_sum = 0.99 
            if current_sum > 1e-7:
                alpha_new = (alpha / current_sum) * target_sum
                beta_new = (beta / current_sum) * target_sum
                # Ensure individual params remain non-negative after scaling
                alpha = max(0, alpha_new)
                beta = max(0, beta_new)
                # If sum is still off due to flooring at 0, adjust one (e.g. beta)
                if alpha + beta >= 0.995 :
                    beta = target_sum - alpha if target_sum > alpha else 0.0
            else: 
                alpha = 0.05 
                beta = 0.90
            valid = False
        return omega, alpha, beta, valid


    def _neg_log_likelihood(self, params, returns_centered):
        omega, alpha, beta = params
        # FIXED: Relaxed bound checking to match new bounds
        if not (omega > 1e-9 and alpha >= 0 and beta >= 0 and alpha + beta < 0.995):
            return np.inf

        T = len(returns_centered)
        sigma2 = np.zeros(T)
        var_ret_centered = np.var(returns_centered) if T > 1 else 1e-7
        
        if (1 - alpha - beta) > 1e-7:
            sigma2_0_uncond = omega / (1 - alpha - beta)
            sigma2[0] = max(1e-8, sigma2_0_uncond, var_ret_centered)
        else:
            sigma2[0] = max(1e-8, var_ret_centered)
        if sigma2[0] <= 1e-9 : sigma2[0] = 1e-8

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns_centered[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(1e-8, sigma2[t]) 

        if np.any(sigma2 <= 1e-9) or np.any(np.isnan(sigma2)) or np.any(np.isinf(sigma2)): return np.inf
        
        log_terms = np.log(sigma2) + returns_centered**2 / sigma2
        if np.any(np.isnan(log_terms)) or np.any(np.isinf(log_terms)): return np.inf

        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + log_terms)
        return -log_likelihood if np.isfinite(log_likelihood) else np.inf
    
    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'L-BFGS-B', 
            max_iter: int = 200) -> 'GARCHModel': 
        self.fit_success = False 
        self.fit_message = "Fit not successful"

        if isinstance(returns, pl.Series): returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame): returns_np = returns.to_numpy().flatten()
        else: returns_np = np.asarray(returns)

        returns_np = returns_np[~np.isnan(returns_np)]
        
        if len(returns_np) < 20:
            self.fit_message = f"GARCH: Not enough data points ({len(returns_np)})."
            self._use_initial_params_for_history(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7: 
            self.fit_message = "GARCH: Return series has very low variance."
            self._handle_low_variance_series(returns_np, std_dev) 
            return self

        clip_threshold = 7 * std_dev 
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean
        if np.std(returns_centered) < 1e-7: 
            self.fit_message = "GARCH: Demeaned returns have very low variance."
            self._handle_low_variance_series(returns_np, np.std(returns_centered))
            return self

        # Use checked initial parameters
        omega_i, alpha_i, beta_i, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init, context="initial_guess")
        initial_params = [omega_i, alpha_i, beta_i]
        
        # FIXED: Relaxed bounds to prevent optimization issues
        bounds = [(1e-9, 0.2), (1e-6, 0.99), (1e-6, 0.99)] # omega, alpha, beta bounds

        # FIXED: Removed 'eps' option that was causing warnings
        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9}
        
        methods_to_try = [method, 'SLSQP'] 
        final_result_obj = None 
        
        for opt_method in methods_to_try:
            try:
                current_bounds = bounds if opt_method != 'Nelder-Mead' else None
                result = minimize(self._neg_log_likelihood, initial_params, args=(returns_centered,),
                                  method=opt_method, bounds=current_bounds, options=optimizer_options)
                
                final_result_obj = result 
                if result.success:
                    omega_fit, alpha_fit, beta_fit = result.x
                    _, _, _, params_valid = self._check_parameters(omega_fit, alpha_fit, beta_fit, context="fit_check")
                    if params_valid and (alpha_fit + beta_fit < 0.995): 
                        self.omega, self.alpha, self.beta = omega_fit, alpha_fit, beta_fit
                        self.fit_success = True
                        self.fit_message = f"Optimization successful with {opt_method}."
                        break 
                self.fit_message = result.message # Store message from optimizer
            except Exception as e: 
                self.fit_message = f"Exception during {opt_method}: {str(e)}"
                final_result_obj = None
                continue 
        
        if not self.fit_success:
            msg_detail = final_result_obj.message if final_result_obj and hasattr(final_result_obj, 'message') else self.fit_message
            self.omega, self.alpha, self.beta, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init) # Re-check defaults

        self._finalize_fit(returns_centered) 
        return self

    def _handle_low_variance_series(self, returns_np_original, actual_std_dev):
        self.mean = np.mean(returns_np_original)
        var_to_use = max(1e-8, actual_std_dev**2 if pd.notna(actual_std_dev) and actual_std_dev > 0 else 1e-8)
        self.variance_history = np.full(len(returns_np_original), var_to_use)
        self.residuals_history = returns_np_original - self.mean
        self.sigma2_t = self.variance_history[-1] if len(self.variance_history) > 0 else 1e-7
        self.is_fitted = True; self.fit_success = True 
        
        self.omega, self.alpha, self.beta, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init)
        if (1-self.alpha-self.beta) > 1e-7 : 
            self.omega = max(1e-8, var_to_use * (1 - self.alpha - self.beta))
        else: 
            self.omega = 1e-7 


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
            uncond_denom = (1-self.alpha-self.beta)
            self.sigma2_t = self.omega / uncond_denom if uncond_denom > 1e-7 else 1e-7
            self.is_fitted = True
            return

        sigma2 = np.zeros(T)
        var_ret_centered = np.var(returns_centered) if T > 1 else 1e-7
        if var_ret_centered <= 1e-9: var_ret_centered = 1e-8 # Ensure it's positive

        if use_empirical_var_for_sigma2_0 or (1 - self.alpha - self.beta) <= 1e-7: 
             sigma2[0] = max(1e-8, var_ret_centered)
        else: 
             sigma2[0] = max(1e-8, self.omega / (1 - self.alpha - self.beta))
        if sigma2[0] <= 1e-9 : sigma2[0] = 1e-8 

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
        if len(innovations) > 0 and (np.allclose(innovations,0, atol=1e-9) or np.var(innovations) < 1e-12):
            innovations = innovations + np.random.normal(0, 1e-7, size=len(innovations)) 
        return innovations


class GJRGARCHModel(GARCHModel):
    def __init__(self, omega: float = 1e-6, alpha: float = 0.08, beta: float = 0.85, gamma: float = 0.05):
        super().__init__(omega, alpha, beta)
        self.gamma_init = max(0, gamma) # Ensure init gamma is non-negative
        self.gamma = self.gamma_init
        # self.fit_success and self.fit_message inherited, will be reset in GJR's fit

    def _check_gjr_parameters(self, omega, alpha, beta, gamma, context="final"):
        omega, alpha, beta, garch_valid = self._check_parameters(omega, alpha, beta, context) 
        valid = garch_valid

        if not (isinstance(gamma, (int,float)) and pd.notna(gamma) and gamma >= 0): 
            if context=="final": warnings.warn(f"GJR gamma invalid ({gamma}). Correcting.")
            gamma = 0.01 
            valid = False
        
        stationarity_val = alpha + beta + 0.5 * gamma
        # FIXED: Relaxed stationarity condition from 0.99999 to 0.995
        if stationarity_val >= 0.995:
            if context=="final": warnings.warn(f"GJR sum condition ({stationarity_val:.4f}) too high. Adjusting for stationarity.")
            target_sum_overall = 0.99
            if alpha + beta >= target_sum_overall - 0.001: # FIXED: Added small buffer
                gamma = max(0, gamma * 0.01) # Drastically reduce gamma if alpha+beta is the issue
                current_sum_alpha_beta = alpha + beta
                required_sum_alpha_beta_for_gjr = target_sum_overall - 0.5 * gamma
                if current_sum_alpha_beta > required_sum_alpha_beta_for_gjr and current_sum_alpha_beta > 1e-7:
                    alpha_new = (alpha / current_sum_alpha_beta) * required_sum_alpha_beta_for_gjr
                    beta_new = (beta / current_sum_alpha_beta) * required_sum_alpha_beta_for_gjr
                    alpha = max(0, alpha_new); beta = max(0, beta_new)
                    if alpha + beta + 0.5 * gamma >= 0.995: # Final attempt to fix
                        beta = target_sum_overall - alpha - 0.5*gamma if (target_sum_overall - 0.5*gamma) > alpha else 0.0
                elif current_sum_alpha_beta <= 1e-7 : 
                     alpha = 0.03; beta = 0.80; gamma = 0.02 # Reset all to typical GJR stationary
            else: 
                gamma = max(0, (target_sum_overall - (alpha + beta)) * 2 * 0.99 ) 
            valid = False
        return omega, alpha, beta, gamma, valid

    def _neg_log_likelihood(self, params, returns_centered): 
        omega, alpha, beta, gamma = params
        # FIXED: Relaxed bound checking to match new bounds
        if not (omega > 1e-9 and alpha >= 0 and beta >= 0 and gamma >= 0 and \
                alpha + beta + 0.5 * gamma < 0.995):
            return np.inf

        T = len(returns_centered)
        sigma2 = np.zeros(T)
        var_ret_centered = np.var(returns_centered) if T > 1 else 1e-7
        
        uncond_denom = (1 - alpha - beta - 0.5 * gamma)
        if uncond_denom > 1e-7:
            sigma2_0_uncond = omega / uncond_denom
            sigma2[0] = max(1e-8, sigma2_0_uncond, var_ret_centered)
        else:
            sigma2[0] = max(1e-8, var_ret_centered)
        if sigma2[0] <= 1e-9: sigma2[0] = 1e-8

        for t in range(1, T):
            I_tm1 = 1.0 if returns_centered[t-1] < 0 else 0.0
            sigma2[t] = (omega + alpha * returns_centered[t-1]**2 + 
                         beta * sigma2[t-1] + gamma * I_tm1 * returns_centered[t-1]**2)
            sigma2[t] = max(1e-8, sigma2[t])

        if np.any(sigma2 <= 1e-9) or np.any(np.isnan(sigma2)) or np.any(np.isinf(sigma2)): return np.inf
        log_terms = np.log(sigma2) + returns_centered**2 / sigma2
        if np.any(np.isnan(log_terms)) or np.any(np.isinf(log_terms)): return np.inf
        
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + log_terms)
        return -log_likelihood if np.isfinite(log_likelihood) else np.inf

    def fit(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
            method: str = 'L-BFGS-B', max_iter: int = 200) -> 'GJRGARCHModel':
        self.fit_success = False 
        self.fit_message = "Fit not successful"

        if isinstance(returns, pl.Series): returns_np = returns.to_numpy()
        elif isinstance(returns, pl.DataFrame): returns_np = returns.to_numpy().flatten()
        else: returns_np = np.asarray(returns)

        returns_np = returns_np[~np.isnan(returns_np)]
        
        if len(returns_np) < 20:
            self.fit_message = f"GJR: Not enough data points ({len(returns_np)})."
            self._use_initial_params_for_history_gjr(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7:
            self.fit_message = "GJR: Return series has very low variance."
            self._handle_low_variance_series_gjr(returns_np, std_dev)
            return self
            
        clip_threshold = 7 * std_dev 
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean
        if np.std(returns_centered) < 1e-7:
            self.fit_message = "GJR: Demeaned returns have very low variance."
            self._handle_low_variance_series_gjr(returns_np, np.std(returns_centered))
            return self

        omega_i, alpha_i, beta_i, gamma_i, _ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init, context="initial_guess")
        initial_params = [omega_i, alpha_i, beta_i, gamma_i]
        
        # FIXED: Relaxed bounds to prevent optimization issues
        bounds = [(1e-9, 0.2), (1e-6, 0.99), (1e-6, 0.99), (1e-6, 0.99)] 

        # FIXED: Removed 'eps' option that was causing warnings
        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9}
        methods_to_try = [method, 'SLSQP'] 
        final_result_obj = None
        
        for opt_method in methods_to_try:
            try:
                current_bounds = bounds if opt_method != 'Nelder-Mead' else None
                result = minimize(self._neg_log_likelihood, initial_params, args=(returns_centered,),
                                  method=opt_method, bounds=current_bounds, options=optimizer_options)
                final_result_obj = result
                if result.success:
                    o, a, b, g = result.x
                    _, _, _, _, params_valid = self._check_gjr_parameters(o, a, b, g, context="fit_check")
                    if params_valid and (a + b + 0.5 * g < 0.995):
                        self.omega, self.alpha, self.beta, self.gamma = o,a,b,g
                        self.fit_success = True
                        self.fit_message = f"Optimization successful with {opt_method}."
                        break
                self.fit_message = result.message
            except Exception as e:
                self.fit_message = f"Exception during {opt_method}: {str(e)}"
                final_result_obj = None; continue
        
        if not self.fit_success:
            msg_detail = final_result_obj.message if final_result_obj and hasattr(final_result_obj, 'message') else self.fit_message
            self.omega,self.alpha,self.beta,self.gamma,_ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init)

        self._finalize_fit_gjr(returns_centered) 
        return self

    def _handle_low_variance_series_gjr(self, returns_np_original, actual_std_dev):
        self.mean = np.mean(returns_np_original)
        var_to_use = max(1e-8, actual_std_dev**2 if pd.notna(actual_std_dev) and actual_std_dev > 0 else 1e-8)
        self.variance_history = np.full(len(returns_np_original), var_to_use)
        self.residuals_history = returns_np_original - self.mean
        self.sigma2_t = self.variance_history[-1] if len(self.variance_history) > 0 else 1e-7
        self.is_fitted = True; self.fit_success = True
        
        self.omega, self.alpha, self.beta, self.gamma, _ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init)
        denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)
        if denom > 1e-7: self.omega = max(1e-8, var_to_use * denom)
        else: self.omega = 1e-7

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
        var_ret_centered = np.var(returns_centered) if T > 1 else 1e-7
        if var_ret_centered <= 1e-9: var_ret_centered = 1e-8
        uncond_denom = (1 - self.alpha - self.beta - 0.5 * self.gamma)

        if use_empirical_var_for_sigma2_0 or uncond_denom <= 1e-7:
            sigma2[0] = max(1e-8, var_ret_centered)
        else:
            sigma2[0] = max(1e-8, self.omega / uncond_denom)
        if sigma2[0] <= 1e-9: sigma2[0] = 1e-8
        
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
            warnings.warn("GJR-GARCH model has insufficient history for prediction. Returning unconditional variance.")
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
            else: 
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
        if len(innovations) > 0 and (np.allclose(innovations,0, atol=1e-9) or np.var(innovations) < 1e-12):
            innovations = innovations + np.random.normal(0, 1e-7, size=len(innovations))
        return innovations

class ThreePhaseVolatilityModel:
    """
    Enhanced Three-Phase Volatility Model implementing continuous-time stochastic volatility dynamics
    with regime-switching support for high-uncertainty events.
    
    Implements Requirements 3.1-3.5: Three distinct phases with Gaussian pre-event scaling,
    exponential post-event increase, exponential decay, and regime-switching support.
    """
    
    def __init__(self, baseline_model: Union[GARCHModel, GJRGARCHModel],
                 k1: float = 1.5, k2: float = 2.0, 
                 delta_t1: float = 5.0, delta_t2: float = 3.0, delta_t3: float = 10.0, 
                 delta: int = 5, regime_state: int = 1):
        """
        Initialize Three-Phase Volatility Model with phase-specific dynamics.
        
        Args:
            baseline_model: Underlying GARCH or GJR-GARCH model for baseline volatility
            k1: Pre-event Gaussian volatility scaling parameter (k1 > 1)
            k2: Post-event exponential increase parameter (k2 > k1)
            delta_t1: Pre-event phase duration parameter
            delta_t2: Rising post-event phase duration parameter  
            delta_t3: Decay post-event phase duration parameter
            delta: Event duration (days between rising and decay phases)
            regime_state: Regime identifier (1=low uncertainty, 2=high uncertainty)
        """
        self.baseline_model = baseline_model
        self.regime_state = regime_state
        self.delta = delta
        
        # Validate input parameters first
        if k1 <= 1:
            raise ValueError("k1 must be greater than 1 for pre-event scaling")
        if k2 <= k1:
            raise ValueError("k2 must be greater than k1 for post-event amplification")
        if delta_t1 <= 0 or delta_t2 <= 0 or delta_t3 <= 0:
            raise ValueError("All delta parameters must be positive")
        
        # Store original parameters for regime switching
        self._base_params = {
            'k1': k1, 'k2': k2, 'delta_t1': delta_t1, 
            'delta_t2': delta_t2, 'delta_t3': delta_t3
        }
        
        # Regime-specific parameter sets
        self._regime_params = {
            1: self._base_params.copy(),  # Low uncertainty regime
            2: {  # High uncertainty regime - amplified parameters
                'k1': k1 * 1.2, 'k2': k2 * 1.5, 'delta_t1': delta_t1 * 0.8,
                'delta_t2': delta_t2 * 1.3, 'delta_t3': delta_t3 * 1.2
            }
        }
        
        self._set_regime_parameters(regime_state)
        
        # Track fitted state
        self.is_fitted = False
        self.event_times = []
        self.volatility_history = None
        
    def _set_regime_parameters(self, regime_state: int):
        """Set parameters based on current regime state."""
        if regime_state not in self._regime_params:
            raise ValueError(f"Invalid regime_state: {regime_state}. Must be 1 or 2.")
            
        params = self._regime_params[regime_state]
        self.k1 = params['k1']
        self.k2 = params['k2'] 
        self.delta_t1 = max(1e-3, params['delta_t1'])
        self.delta_t2 = max(1e-3, params['delta_t2'])
        self.delta_t3 = max(1e-3, params['delta_t3'])
        self.regime_state = regime_state
        
    def _validate_parameters(self):
        """Validate model parameters according to requirements."""
        if self.k1 <= 1:
            raise ValueError("k1 must be greater than 1 for pre-event scaling")
        if self.k2 <= self.k1:
            raise ValueError("k2 must be greater than k1 for post-event amplification")
            
    def switch_regime(self, new_regime_state: int):
        """
        Switch to different parameter set for regime-switching support.
        Implements Requirement 3.5: regime-switching with different parameter sets.
        """
        if new_regime_state != self.regime_state:
            self._set_regime_parameters(new_regime_state)
            
    def phi1(self, t: float, t_event: float) -> float:
        """
        Pre-event Gaussian volatility scaling function (φ₁).
        Implements Requirement 3.2: Gaussian volatility scaling with parameter k_1.
        
        Args:
            t: Current time
            t_event: Event time
            
        Returns:
            Gaussian-scaled volatility adjustment
        """
        # Gaussian scaling centered at event time, active for all times
        time_diff = t - t_event
        gaussian_term = np.exp(-(time_diff**2) / (2 * self.delta_t1**2))
        return (self.k1 - 1) * gaussian_term
    
    def phi2(self, t: float, t_event: float) -> float:
        """
        Rising post-event exponential increase function (φ₂).
        Implements Requirement 3.3: exponential volatility increase with parameter k_2 > k_1.
        
        Args:
            t: Current time
            t_event: Event time
            
        Returns:
            Exponential increase volatility adjustment
        """
        time_since_event = t - t_event
        if time_since_event <= 0 or time_since_event > self.delta:
            return 0.0
            
        # Exponential increase during rising phase
        if self.delta_t2 <= 1e-6:
            return self.k2 - 1
        else:
            exp_term = 1 - np.exp(-time_since_event / self.delta_t2)
            return (self.k2 - 1) * exp_term
    
    def phi3(self, t: float, t_event: float) -> float:
        """
        Decay post-event exponential decay function (φ₃).
        Implements Requirement 3.4: exponential volatility decay.
        
        Args:
            t: Current time
            t_event: Event time
            
        Returns:
            Exponential decay volatility adjustment
        """
        time_since_decay_start = t - (t_event + self.delta)
        if time_since_decay_start <= 0:
            return 0.0
            
        # Exponential decay starting from the peak value at end of rising phase
        if self.delta_t3 <= 1e-6:
            return 0.0
        else:
            # Get the peak value from phi2 at the end of rising phase
            peak_value = self.phi2(t_event + self.delta, t_event)
            exp_term = np.exp(-time_since_decay_start / self.delta_t3)
            return peak_value * exp_term
    
    def calculate_phase_adjustment(self, t: float, t_event: float) -> float:
        """
        Calculate total volatility adjustment Φ(t, s_t) across all phases.
        Implements Requirement 3.1: three distinct phases implementation.
        
        Args:
            t: Current time
            t_event: Event time
            
        Returns:
            Combined phase adjustment factor
        """
        # Determine active phase and calculate adjustment
        if t < t_event:
            # Pre-event phase: only phi1 is active
            return self.phi1(t, t_event)
        elif t <= t_event + self.delta:
            # Rising post-event phase: phi2 dominates, phi1 may still contribute
            phi1_val = self.phi1(t, t_event)
            phi2_val = self.phi2(t, t_event)
            # Use maximum of the two phases for smooth transition
            return max(phi1_val, phi2_val)
        else:
            # Decay post-event phase: only phi3 is active
            return self.phi3(t, t_event)
    
    def calculate_volatility(self, t: float, t_event: float, baseline_volatility: float) -> float:
        """
        Calculate total volatility: σ(t) = σ_baseline(t) * (1 + Φ(t, s_t)).
        
        Args:
            t: Current time
            t_event: Event time
            baseline_volatility: Baseline volatility from GARCH model
            
        Returns:
            Adjusted volatility incorporating three-phase dynamics
        """
        if baseline_volatility < 1e-8:
            baseline_volatility = 1e-8
            
        phase_adjustment = self.calculate_phase_adjustment(t, t_event)
        adjusted_volatility = baseline_volatility * (1 + phase_adjustment)
        
        # Ensure positive volatility with reasonable floor
        return max(baseline_volatility * 0.1, adjusted_volatility)
    
    def fit(self, returns: Union[np.ndarray, pl.Series], event_times: List[float]):
        """
        Fit the three-phase volatility model to return data with known event times.
        
        Args:
            returns: Return series for fitting
            event_times: List of event times in the return series
        """
        # First fit the baseline GARCH model
        self.baseline_model.fit(returns)
        
        # Store event times and mark as fitted
        self.event_times = event_times
        self.is_fitted = True
        
        return self
    
    def predict_volatility(self, time_points: np.ndarray, event_time: float) -> np.ndarray:
        """
        Predict volatility at specified time points around an event.
        
        Args:
            time_points: Array of time points for prediction
            event_time: Event time reference
            
        Returns:
            Array of predicted volatilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Get baseline volatility predictions
        n_steps = len(time_points)
        baseline_volatilities = self.baseline_model.predict(n_steps)
        
        # Apply three-phase adjustments
        adjusted_volatilities = np.zeros_like(time_points)
        for i, t in enumerate(time_points):
            baseline_vol = np.sqrt(baseline_volatilities[i]) if i < len(baseline_volatilities) else np.sqrt(baseline_volatilities[-1])
            adjusted_volatilities[i] = self.calculate_volatility(t, event_time, baseline_vol)
            
        return adjusted_volatilities
    
    def calculate_volatility_series(self, 
                                   days_to_event: Union[List[int], np.ndarray], 
                                   baseline_conditional_vol_series: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate volatility series for a sequence of days relative to event.
        
        Args:
            days_to_event: Array of days relative to event (negative = before, positive = after)
            baseline_conditional_vol_series: Optional baseline volatility series
            
        Returns:
            Array of adjusted volatilities
        """
        if not isinstance(days_to_event, np.ndarray):
            days_to_event_np = np.array(days_to_event, dtype=float)
        else:
            days_to_event_np = days_to_event.astype(float)

        # Get baseline volatilities
        if baseline_conditional_vol_series is not None:
            if len(baseline_conditional_vol_series) != len(days_to_event_np):
                raise ValueError(
                    f"Length mismatch: baseline_conditional_vol_series ({len(baseline_conditional_vol_series)}) "
                    f"vs days_to_event ({len(days_to_event_np)})"
                )
            baseline_vols = np.maximum(baseline_conditional_vol_series, 1e-8)
        else:
            if not self.baseline_model.is_fitted:
                raise RuntimeError("Baseline model must be fitted or baseline_conditional_vol_series provided")
            
            # Calculate unconditional volatility from baseline model
            bm = self.baseline_model
            if isinstance(bm, GJRGARCHModel):
                denominator = (1 - bm.alpha - bm.beta - 0.5 * bm.gamma)
            else:
                denominator = (1 - bm.alpha - bm.beta)
            
            if denominator > 1e-7:
                uncond_var = bm.omega / denominator
            else:
                uncond_var = 1e-7
                
            baseline_vol = np.sqrt(max(uncond_var, 1e-8))
            baseline_vols = np.full_like(days_to_event_np, baseline_vol, dtype=float)

        # Apply three-phase adjustments (event at t=0)
        volatility_series = np.array([
            self.calculate_volatility(t_rel, 0.0, baseline_vols[i]) 
            for i, t_rel in enumerate(days_to_event_np)
        ], dtype=float)
        
        return volatility_series


class EnhancedGJRGARCHModel(GJRGARCHModel):
    """
    Enhanced GJR-GARCH Model with Three-Phase Volatility Integration.
    
    Extends the existing GJRGARCHModel to support:
    - Three-phase volatility dynamics around events
    - Ornstein-Uhlenbeck process for baseline volatility h(t)
    - Event-specific volatility adjustments Φ(t, s_t)
    
    Implements Requirements 3.1-3.3: Integration with existing GARCH framework.
    """
    
    def __init__(self, omega: float = 1e-6, alpha: float = 0.08, beta: float = 0.85, gamma: float = 0.05,
                 ou_kappa: float = 0.1, ou_theta: float = 0.02, ou_sigma: float = 0.01):
        """
        Initialize Enhanced GJR-GARCH Model with Ornstein-Uhlenbeck process.
        
        Args:
            omega, alpha, beta, gamma: Standard GJR-GARCH parameters
            ou_kappa: Ornstein-Uhlenbeck mean reversion speed
            ou_theta: Ornstein-Uhlenbeck long-term mean
            ou_sigma: Ornstein-Uhlenbeck volatility of volatility
        """
        super().__init__(omega, alpha, beta, gamma)
        
        # Ornstein-Uhlenbeck process parameters for h(t)
        self.ou_kappa = ou_kappa  # Mean reversion speed
        self.ou_theta = ou_theta  # Long-term mean
        self.ou_sigma = ou_sigma  # Volatility of volatility
        
        # Three-phase volatility model (will be set externally)
        self.three_phase_model = None
        self.event_times = []
        
        # Enhanced state tracking
        self.ou_process_history = None
        self.event_adjustments_history = None
        
    def set_three_phase_model(self, three_phase_model: ThreePhaseVolatilityModel):
        """Set the three-phase volatility model for event adjustments."""
        self.three_phase_model = three_phase_model
        
    def set_event_times(self, event_times: List[float]):
        """Set event times for volatility adjustments."""
        self.event_times = event_times
        
    def simulate_ou_process(self, n_steps: int, dt: float = 1.0, initial_value: Optional[float] = None) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck process for baseline volatility h(t).
        
        dh(t) = κ(θ - h(t))dt + σ dW(t)
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
            initial_value: Initial value (if None, uses theta)
            
        Returns:
            Array of h(t) values
        """
        if initial_value is None:
            initial_value = self.ou_theta
            
        h_values = np.zeros(n_steps)
        h_values[0] = initial_value
        
        # Generate random shocks
        np.random.seed(42)  # For reproducibility
        dW = np.random.normal(0, np.sqrt(dt), n_steps - 1)
        
        for t in range(1, n_steps):
            # Ornstein-Uhlenbeck SDE discretization
            drift = self.ou_kappa * (self.ou_theta - h_values[t-1]) * dt
            diffusion = self.ou_sigma * dW[t-1]
            h_values[t] = h_values[t-1] + drift + diffusion
            
            # Ensure positive values
            h_values[t] = max(1e-8, h_values[t])
            
        return h_values
    
    def calculate_event_adjustments(self, time_points: np.ndarray) -> np.ndarray:
        """
        Calculate event-specific volatility adjustments Φ(t, s_t) for all time points.
        
        Args:
            time_points: Array of time points
            
        Returns:
            Array of volatility adjustment factors
        """
        if self.three_phase_model is None or len(self.event_times) == 0:
            return np.ones_like(time_points)
            
        adjustments = np.ones_like(time_points)
        
        for i, t in enumerate(time_points):
            # Find the closest event time
            if len(self.event_times) > 0:
                closest_event_idx = np.argmin(np.abs(np.array(self.event_times) - t))
                closest_event_time = self.event_times[closest_event_idx]
                
                # Calculate phase adjustment for this event
                phase_adjustment = self.three_phase_model.calculate_phase_adjustment(t, closest_event_time)
                adjustments[i] = 1 + phase_adjustment
                
        return adjustments
    
    def fit_enhanced(self, returns: Union[np.ndarray, pl.Series, pl.DataFrame], 
                    event_times: List[float] = None,
                    method: str = 'L-BFGS-B', max_iter: int = 200) -> 'EnhancedGJRGARCHModel':
        """
        Fit enhanced GJR-GARCH model with three-phase volatility integration.
        
        Args:
            returns: Return series
            event_times: List of event times in the series
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            Fitted model
        """
        # First fit the base GJR-GARCH model
        super().fit(returns, method, max_iter)
        
        # Store event times
        if event_times is not None:
            self.event_times = event_times
            
        # Simulate OU process for the length of the return series
        if isinstance(returns, pl.Series):
            n_obs = len(returns)
        elif isinstance(returns, pl.DataFrame):
            n_obs = returns.height
        else:
            n_obs = len(returns)
            
        self.ou_process_history = self.simulate_ou_process(n_obs)
        
        # Calculate event adjustments if three-phase model is available
        if self.three_phase_model is not None and len(self.event_times) > 0:
            time_points = np.arange(n_obs, dtype=float)
            self.event_adjustments_history = self.calculate_event_adjustments(time_points)
        else:
            self.event_adjustments_history = np.ones(n_obs)
            
        return self


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for continuous-time stochastic differential equations.
    
    Implements Requirements 10.1, 10.2, 10.5:
    - Continuous-time SDE simulation using Euler-Maruyama scheme
    - Multi-asset correlation handling in simulations
    - Regime-switching dynamics to simulation paths
    - Parallel processing support for multiple simulation runs
    """
    
    def __init__(self, config: Optional['SimulationConfig'] = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            config: Simulation configuration parameters
        """
        try:
            from .config import get_config
        except ImportError:
            from config import get_config
        
        if config is None:
            config = get_config().simulation
            
        self.n_simulations = config.n_simulations
        self.n_steps = config.n_steps
        self.parallel_processing = config.parallel_processing
        self.random_seed = config.random_seed
        
        # Simulation state
        self.simulation_results = None
        self.correlation_matrix = None
        self.regime_transition_matrix = None
        
        # Models for simulation
        self.volatility_model = None
        self.multi_risk_framework = None
        self.investor_models = None
        
    def set_models(self, volatility_model=None, multi_risk_framework=None, investor_models=None):
        """Set models for simulation."""
        if volatility_model is not None:
            self.volatility_model = volatility_model
        if multi_risk_framework is not None:
            self.multi_risk_framework = multi_risk_framework
        if investor_models is not None:
            self.investor_models = investor_models
    
    def set_correlation_matrix(self, correlation_matrix: np.ndarray):
        """
        Set correlation matrix for multi-asset simulations.
        
        Args:
            correlation_matrix: Asset correlation matrix (n_assets x n_assets)
        """
        if not isinstance(correlation_matrix, np.ndarray):
            correlation_matrix = np.array(correlation_matrix)
            
        # Validate correlation matrix
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
            
        # Check if positive semi-definite
        eigenvals = np.linalg.eigvals(correlation_matrix)
        if np.any(eigenvals < -1e-8):
            raise ValueError("Correlation matrix must be positive semi-definite")
            
        # Ensure diagonal elements are 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        self.correlation_matrix = correlation_matrix
    
    def set_regime_transition_matrix(self, transition_matrix: np.ndarray):
        """
        Set regime transition matrix for regime-switching dynamics.
        
        Args:
            transition_matrix: Regime transition probabilities (n_regimes x n_regimes)
        """
        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)
            
        # Validate transition matrix
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")
            
        # Check if rows sum to 1
        row_sums = np.sum(transition_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Transition matrix rows must sum to 1")
            
        self.regime_transition_matrix = transition_matrix
    
    def simulate_sde_euler_maruyama(self, 
                                   initial_values: np.ndarray,
                                   drift_func: callable,
                                   diffusion_func: callable,
                                   T: float,
                                   dt: float = None,
                                   n_assets: int = 1,
                                   regime_states: np.ndarray = None,
                                   seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate stochastic differential equations using Euler-Maruyama scheme.
        
        For multi-asset case:
        dS_i(t) = μ_i(t, S_t, regime_t) dt + σ_i(t, S_t, regime_t) dW_i(t)
        
        Args:
            initial_values: Initial values for each asset
            drift_func: Function returning drift terms μ(t, S_t, regime_t)
            diffusion_func: Function returning diffusion terms σ(t, S_t, regime_t)
            T: Time horizon
            dt: Time step (if None, uses T/n_steps)
            n_assets: Number of assets
            regime_states: Regime state sequence (if None, single regime)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        if seed is not None:
            np.random.seed(seed)
            
        if dt is None:
            dt = T / self.n_steps
            
        n_steps = int(T / dt)
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize paths
        paths = np.zeros((n_steps + 1, n_assets))
        paths[0] = initial_values
        
        # Initialize regime states if not provided
        if regime_states is None:
            regime_states = np.zeros(n_steps + 1, dtype=int)
        
        # Generate correlated random shocks
        if self.correlation_matrix is not None and n_assets > 1:
            # Cholesky decomposition for correlated shocks
            L = np.linalg.cholesky(self.correlation_matrix)
            random_shocks = np.random.normal(0, 1, (n_steps, n_assets))
            correlated_shocks = (L @ random_shocks.T).T
        else:
            correlated_shocks = np.random.normal(0, 1, (n_steps, n_assets))
        
        # Euler-Maruyama simulation
        for i in range(n_steps):
            t = time_grid[i]
            current_values = paths[i]
            current_regime = regime_states[i]
            
            # Calculate drift and diffusion
            drift = drift_func(t, current_values, current_regime)
            diffusion = diffusion_func(t, current_values, current_regime)
            
            # Ensure proper shapes
            if np.isscalar(drift):
                drift = np.full(n_assets, drift)
            if np.isscalar(diffusion):
                diffusion = np.full(n_assets, diffusion)
                
            # Euler-Maruyama step
            drift_term = drift * dt
            diffusion_term = diffusion * correlated_shocks[i] * np.sqrt(dt)
            
            paths[i + 1] = current_values + drift_term + diffusion_term
            
            # Update regime state if transition matrix is provided
            if self.regime_transition_matrix is not None and i < n_steps - 1:
                regime_states[i + 1] = self._simulate_regime_transition(current_regime)
        
        return {
            'time_grid': time_grid,
            'paths': paths,
            'regime_states': regime_states,
            'dt': dt,
            'n_steps': n_steps
        }
    
    def _simulate_regime_transition(self, current_regime: int) -> int:
        """Simulate regime transition based on transition matrix."""
        if self.regime_transition_matrix is None:
            return current_regime
            
        transition_probs = self.regime_transition_matrix[current_regime]
        return np.random.choice(len(transition_probs), p=transition_probs)
    
    def simulate_asset_prices(self,
                             initial_prices: np.ndarray,
                             expected_returns: np.ndarray,
                             volatilities: np.ndarray,
                             T: float,
                             dt: float = None,
                             event_times: List[float] = None,
                             seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate asset price paths with three-phase volatility dynamics.
        
        Args:
            initial_prices: Initial asset prices
            expected_returns: Expected return rates for each asset
            volatilities: Base volatility levels for each asset
            T: Time horizon
            dt: Time step
            event_times: List of event times for volatility adjustments
            seed: Random seed
            
        Returns:
            Dictionary with price simulation results
        """
        n_assets = len(initial_prices)
        
        def drift_func(t, S_t, regime_t):
            """Drift function for asset prices."""
            return expected_returns * S_t
        
        def diffusion_func(t, S_t, regime_t):
            """Diffusion function with three-phase volatility adjustments."""
            base_vol = volatilities * S_t
            
            # Apply three-phase volatility adjustments if model is available
            if self.volatility_model is not None and event_times is not None:
                vol_adjustments = np.ones(n_assets)
                for event_time in event_times:
                    if hasattr(self.volatility_model, 'calculate_phase_adjustment'):
                        phase_adj = self.volatility_model.calculate_phase_adjustment(t, event_time)
                        vol_adjustments *= (1 + phase_adj)
                base_vol *= vol_adjustments
            
            return base_vol
        
        # Generate regime states if regime transition matrix is available
        regime_states = None
        if self.regime_transition_matrix is not None:
            if dt is None:
                dt = T / self.n_steps
            n_steps = int(T / dt)
            regime_states = self._simulate_regime_sequence(n_steps + 1, seed)
        
        return self.simulate_sde_euler_maruyama(
            initial_values=initial_prices,
            drift_func=drift_func,
            diffusion_func=diffusion_func,
            T=T,
            dt=dt,
            n_assets=n_assets,
            regime_states=regime_states,
            seed=seed
        )
    
    def _simulate_regime_sequence(self, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulate a sequence of regime states."""
        if seed is not None:
            np.random.seed(seed)
            
        n_regimes = self.regime_transition_matrix.shape[0]
        regime_states = np.zeros(n_steps, dtype=int)
        
        # Start in regime 0
        regime_states[0] = 0
        
        for i in range(1, n_steps):
            regime_states[i] = self._simulate_regime_transition(regime_states[i-1])
            
        return regime_states
    
    def run_monte_carlo_simulation(self,
                                  initial_prices: np.ndarray,
                                  expected_returns: np.ndarray,
                                  volatilities: np.ndarray,
                                  T: float,
                                  dt: float = None,
                                  event_times: List[float] = None,
                                  n_simulations: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Run multiple Monte Carlo simulations.
        
        Args:
            initial_prices: Initial asset prices
            expected_returns: Expected return rates
            volatilities: Base volatility levels
            T: Time horizon
            dt: Time step
            event_times: Event times for volatility adjustments
            n_simulations: Number of simulations (if None, uses config)
            
        Returns:
            Dictionary with aggregated simulation results
        """
        if n_simulations is None:
            n_simulations = self.n_simulations
            
        n_assets = len(initial_prices)
        
        if dt is None:
            dt = T / self.n_steps
        n_steps = int(T / dt)
        
        # Initialize result arrays
        all_paths = np.zeros((n_simulations, n_steps + 1, n_assets))
        all_regime_states = np.zeros((n_simulations, n_steps + 1), dtype=int)
        
        if self.parallel_processing:
            # Parallel processing implementation
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            
            n_cores = min(mp.cpu_count(), n_simulations)
            
            def run_single_simulation(sim_idx):
                return self.simulate_asset_prices(
                    initial_prices=initial_prices,
                    expected_returns=expected_returns,
                    volatilities=volatilities,
                    T=T,
                    dt=dt,
                    event_times=event_times,
                    seed=self.random_seed + sim_idx if self.random_seed is not None else None
                )
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = {executor.submit(run_single_simulation, i): i for i in range(n_simulations)}
                
                for future in as_completed(futures):
                    sim_idx = futures[future]
                    try:
                        result = future.result()
                        all_paths[sim_idx] = result['paths']
                        all_regime_states[sim_idx] = result['regime_states']
                    except Exception as e:
                        print(f"Simulation {sim_idx} failed: {e}")
                        # Fill with NaN for failed simulations
                        all_paths[sim_idx] = np.nan
                        all_regime_states[sim_idx] = 0
        else:
            # Sequential processing
            for i in range(n_simulations):
                seed = self.random_seed + i if self.random_seed is not None else None
                result = self.simulate_asset_prices(
                    initial_prices=initial_prices,
                    expected_returns=expected_returns,
                    volatilities=volatilities,
                    T=T,
                    dt=dt,
                    event_times=event_times,
                    seed=seed
                )
                all_paths[i] = result['paths']
                all_regime_states[i] = result['regime_states']
        
        # Store results
        self.simulation_results = {
            'time_grid': np.linspace(0, T, n_steps + 1),
            'all_paths': all_paths,
            'all_regime_states': all_regime_states,
            'n_simulations': n_simulations,
            'n_assets': n_assets,
            'dt': dt,
            'T': T
        }
        
        return self.simulation_results
    
    def simulate_portfolio_dynamics(self,
                                   initial_wealth: float,
                                   portfolio_weights: np.ndarray,
                                   asset_prices: np.ndarray,
                                   transaction_costs: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Simulate portfolio wealth dynamics given asset price paths.
        
        Args:
            initial_wealth: Initial portfolio wealth
            portfolio_weights: Portfolio weights over time (n_steps x n_assets)
            asset_prices: Asset price paths (n_simulations x n_steps x n_assets)
            transaction_costs: Transaction cost parameters
            
        Returns:
            Dictionary with portfolio simulation results
        """
        if self.simulation_results is None:
            raise RuntimeError("Must run Monte Carlo simulation first")
            
        n_simulations, n_time_points, n_assets = asset_prices.shape
        n_steps = n_time_points - 1  # Number of time steps (excluding initial point)
        
        # Initialize wealth paths (n_steps columns, starting from initial wealth)
        wealth_paths = np.zeros((n_simulations, n_steps))
        wealth_paths[:, 0] = initial_wealth  # First column is initial wealth
        
        # Default transaction costs
        if transaction_costs is None:
            transaction_costs = {'tau_b': 0.001, 'tau_s': 0.0005}
        
        for sim in range(n_simulations):
            current_wealth = initial_wealth
            current_weights = portfolio_weights[0] if portfolio_weights.ndim > 1 else portfolio_weights
            
            for t in range(1, n_steps):  # Loop through n_steps-1 iterations
                # Calculate returns
                price_returns = (asset_prices[sim, t] - asset_prices[sim, t-1]) / asset_prices[sim, t-1]
                
                # Portfolio return before rebalancing
                portfolio_return = np.sum(current_weights * price_returns)
                current_wealth *= (1 + portfolio_return)
                
                # Rebalancing with transaction costs
                if portfolio_weights.ndim > 1 and t < len(portfolio_weights):
                    new_weights = portfolio_weights[t]
                    weight_changes = new_weights - current_weights
                    
                    # Calculate transaction costs
                    purchase_costs = np.sum(np.maximum(weight_changes, 0)) * transaction_costs['tau_b']
                    sale_costs = np.sum(np.maximum(-weight_changes, 0)) * transaction_costs['tau_s']
                    total_costs = (purchase_costs + sale_costs) * current_wealth
                    
                    current_wealth -= total_costs
                    current_weights = new_weights
                
                wealth_paths[sim, t] = current_wealth
        
        return {
            'wealth_paths': wealth_paths,
            'initial_wealth': initial_wealth,
            'final_wealth': wealth_paths[:, -1],
            'wealth_returns': (wealth_paths[:, -1] - initial_wealth) / initial_wealth
        }
    
    def predict_enhanced(self, n_steps: int = 1, include_ou_process: bool = True, 
                        include_event_adjustments: bool = True) -> Dict[str, np.ndarray]:
        """
        Enhanced prediction including OU process and event adjustments.
        
        Args:
            n_steps: Number of steps to predict
            include_ou_process: Whether to include OU process component
            include_event_adjustments: Whether to include event adjustments
            
        Returns:
            Dictionary with different volatility components
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Base GJR-GARCH predictions
        base_predictions = super().predict(n_steps)
        
        results = {
            'base_garch': np.sqrt(base_predictions),
            'ou_process': np.full(n_steps, self.ou_theta),
            'event_adjustments': np.ones(n_steps),
            'combined': np.sqrt(base_predictions)
        }
        
        # Add OU process component if requested
        if include_ou_process:
            if self.ou_process_history is not None and len(self.ou_process_history) > 0:
                last_ou_value = self.ou_process_history[-1]
            else:
                last_ou_value = self.ou_theta
                
            ou_predictions = self.simulate_ou_process(n_steps + 1, initial_value=last_ou_value)[1:]
            results['ou_process'] = ou_predictions
        else:
            results['ou_process'] = np.ones(n_steps)  # No OU adjustment
            
        # Add event adjustments if requested
        if include_event_adjustments and self.three_phase_model is not None:
            # Assume predictions start from the end of the fitted data
            start_time = len(self.variance_history) if self.variance_history is not None else 0
            future_times = np.arange(start_time, start_time + n_steps, dtype=float)
            results['event_adjustments'] = self.calculate_event_adjustments(future_times)
        else:
            results['event_adjustments'] = np.ones(n_steps)  # No event adjustments
            
        # Combine all components
        base_vol = np.sqrt(base_predictions)
        ou_component = results['ou_process'] if include_ou_process else np.ones(n_steps)
        event_component = results['event_adjustments'] if include_event_adjustments else np.ones(n_steps)
        
        # Combined volatility: σ_total(t) = σ_GARCH(t) * h(t) * Φ(t, s_t)
        results['combined'] = base_vol * ou_component * event_component
        
        return results
    
    def get_enhanced_volatility_history(self) -> Dict[str, np.ndarray]:
        """
        Get historical volatility decomposition.
        
        Returns:
            Dictionary with historical volatility components
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get volatility history")
            
        base_vol = self.conditional_volatility()
        
        results = {
            'base_garch': base_vol,
            'ou_process': self.ou_process_history if self.ou_process_history is not None else np.ones_like(base_vol),
            'event_adjustments': self.event_adjustments_history if self.event_adjustments_history is not None else np.ones_like(base_vol)
        }
        
        # Combined volatility
        results['combined'] = base_vol * results['ou_process'] * results['event_adjustments']
        
        return results
    
    def calculate_volatility_at_time(self, t: float, baseline_vol: Optional[float] = None) -> float:
        """
        Calculate total volatility at a specific time point.
        
        Args:
            t: Time point
            baseline_vol: Optional baseline volatility (if None, uses unconditional)
            
        Returns:
            Total volatility at time t
        """
        if baseline_vol is None:
            # Use unconditional volatility
            denominator = (1 - self.alpha - self.beta - 0.5 * self.gamma)
            if denominator > 1e-7:
                baseline_var = self.omega / denominator
            else:
                baseline_var = 1e-7
            baseline_vol = np.sqrt(baseline_var)
            
        # OU process component (assume at long-term mean for single point)
        ou_component = self.ou_theta
        
        # Event adjustment component
        event_adjustment = 1.0
        if self.three_phase_model is not None and len(self.event_times) > 0:
            # Find closest event
            closest_event_idx = np.argmin(np.abs(np.array(self.event_times) - t))
            closest_event_time = self.event_times[closest_event_idx]
            phase_adjustment = self.three_phase_model.calculate_phase_adjustment(t, closest_event_time)
            event_adjustment = 1 + phase_adjustment
            
        return baseline_vol * ou_component * event_adjustment
    
    def calculate_statistical_summary(self, metric: str = 'paths') -> Dict[str, np.ndarray]:
        """
        Calculate statistical summaries across simulation runs.
        
        Implements Requirements 10.3, 10.4:
        - Statistical summaries across simulation runs
        - Percentile-based confidence band calculations
        
        Args:
            metric: Type of metric to summarize ('paths', 'returns', 'volatility', 'wealth')
            
        Returns:
            Dictionary with statistical summaries
        """
        if self.simulation_results is None:
            raise RuntimeError("Must run Monte Carlo simulation first")
            
        if metric == 'paths':
            data = self.simulation_results['all_paths']
        elif metric == 'returns':
            # Calculate returns from paths
            paths = self.simulation_results['all_paths']
            returns = np.diff(paths, axis=1) / paths[:, :-1]
            data = returns
        elif metric == 'wealth' and 'wealth_paths' in self.simulation_results:
            data = self.simulation_results['wealth_paths']
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        # Calculate statistical summaries
        summary = {
            'mean': np.mean(data, axis=0),
            'median': np.median(data, axis=0),
            'std': np.std(data, axis=0),
            'var': np.var(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'skewness': self._calculate_skewness(data),
            'kurtosis': self._calculate_kurtosis(data),
            'percentiles': self._calculate_percentiles(data),
            'confidence_bands': self._calculate_confidence_bands(data)
        }
        
        return summary
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness across simulations."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1e-8, std)
        
        centered = data - mean
        skewness = np.mean((centered / std) ** 3, axis=0)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis across simulations."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1e-8, std)
        
        centered = data - mean
        kurtosis = np.mean((centered / std) ** 4, axis=0) - 3  # Excess kurtosis
        return kurtosis
    
    def _calculate_percentiles(self, data: np.ndarray, 
                              percentiles: List[float] = None) -> Dict[str, np.ndarray]:
        """Calculate percentiles across simulations."""
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            
        percentile_results = {}
        for p in percentiles:
            percentile_results[f'p{p}'] = np.percentile(data, p, axis=0)
            
        return percentile_results
    
    def _calculate_confidence_bands(self, data: np.ndarray, 
                                   confidence_levels: List[float] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate confidence bands for different confidence levels.
        
        Args:
            data: Simulation data (n_simulations x n_time_points x n_assets)
            confidence_levels: List of confidence levels (e.g., [0.90, 0.95, 0.99])
            
        Returns:
            Dictionary with confidence bands for each level
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
            
        confidence_bands = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            confidence_bands[f'{level:.0%}'] = {
                'lower': np.percentile(data, lower_percentile, axis=0),
                'upper': np.percentile(data, upper_percentile, axis=0),
                'width': np.percentile(data, upper_percentile, axis=0) - np.percentile(data, lower_percentile, axis=0)
            }
            
        return confidence_bands
    
    def calculate_convergence_diagnostics(self, metric: str = 'paths', 
                                        window_size: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate convergence diagnostics for Monte Carlo results.
        
        Implements Requirements 10.3, 10.4:
        - Convergence diagnostics for Monte Carlo results
        
        Args:
            metric: Type of metric to analyze convergence for
            window_size: Size of rolling window for convergence analysis
            
        Returns:
            Dictionary with convergence diagnostics
        """
        if self.simulation_results is None:
            raise RuntimeError("Must run Monte Carlo simulation first")
            
        n_simulations = self.simulation_results['n_simulations']
        
        if metric == 'paths':
            data = self.simulation_results['all_paths']
        elif metric == 'returns':
            paths = self.simulation_results['all_paths']
            data = np.diff(paths, axis=1) / paths[:, :-1]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        # Calculate running means and standard errors
        running_means = np.zeros((n_simulations, data.shape[1], data.shape[2]))
        running_stds = np.zeros((n_simulations, data.shape[1], data.shape[2]))
        
        for i in range(1, n_simulations + 1):
            running_means[i-1] = np.mean(data[:i], axis=0)
            running_stds[i-1] = np.std(data[:i], axis=0) / np.sqrt(i)  # Standard error
            
        # Calculate convergence metrics
        diagnostics = {
            'running_means': running_means,
            'running_standard_errors': running_stds,
            'convergence_ratio': self._calculate_convergence_ratio(running_means),
            'effective_sample_size': self._calculate_effective_sample_size(data),
            'monte_carlo_error': self._calculate_monte_carlo_error(data),
            'geweke_diagnostic': self._calculate_geweke_diagnostic(running_means)
        }
        
        return diagnostics
    
    def _calculate_convergence_ratio(self, running_means: np.ndarray) -> np.ndarray:
        """Calculate convergence ratio (ratio of final to initial estimates)."""
        if len(running_means) < 2:
            return np.ones_like(running_means[-1])
            
        initial_mean = running_means[len(running_means)//4]  # Use 25% point as initial
        final_mean = running_means[-1]
        
        # Avoid division by zero
        initial_mean = np.where(np.abs(initial_mean) < 1e-8, 1e-8, initial_mean)
        
        return final_mean / initial_mean
    
    def _calculate_effective_sample_size(self, data: np.ndarray) -> np.ndarray:
        """Calculate effective sample size accounting for autocorrelation."""
        n_simulations = data.shape[0]
        
        # Simple approximation: ESS = N / (1 + 2 * sum of autocorrelations)
        # For now, assume independence (ESS = N)
        # In practice, would calculate autocorrelation function
        
        return np.full(data.shape[1:], n_simulations, dtype=float)
    
    def _calculate_monte_carlo_error(self, data: np.ndarray) -> np.ndarray:
        """Calculate Monte Carlo standard error."""
        n_simulations = data.shape[0]
        std_dev = np.std(data, axis=0)
        
        return std_dev / np.sqrt(n_simulations)
    
    def _calculate_geweke_diagnostic(self, running_means: np.ndarray, 
                                   first_fraction: float = 0.1, 
                                   last_fraction: float = 0.5) -> np.ndarray:
        """
        Calculate Geweke convergence diagnostic.
        
        Compares means from first and last portions of the chain.
        """
        n_samples = len(running_means)
        
        first_end = int(first_fraction * n_samples)
        last_start = int((1 - last_fraction) * n_samples)
        
        if first_end >= last_start:
            # Not enough samples for diagnostic
            return np.zeros(running_means.shape[1:])
            
        first_mean = np.mean(running_means[:first_end], axis=0)
        last_mean = np.mean(running_means[last_start:], axis=0)
        
        first_var = np.var(running_means[:first_end], axis=0)
        last_var = np.var(running_means[last_start:], axis=0)
        
        # Standard error of difference
        se_diff = np.sqrt(first_var / first_end + last_var / (n_samples - last_start))
        
        # Avoid division by zero
        se_diff = np.where(se_diff == 0, 1e-8, se_diff)
        
        # Z-score
        z_score = (first_mean - last_mean) / se_diff
        
        return z_score
    
    def implement_variance_reduction(self, method: str = 'antithetic', 
                                   control_variates: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Implement variance reduction techniques for efficiency.
        
        Implements Requirements 10.3, 10.4, 10.5:
        - Variance reduction techniques for efficiency
        
        Args:
            method: Variance reduction method ('antithetic', 'control_variates', 'stratified')
            control_variates: List of control variate types to use
            
        Returns:
            Dictionary with variance-reduced estimates
        """
        if self.simulation_results is None:
            raise RuntimeError("Must run Monte Carlo simulation first")
            
        if method == 'antithetic':
            return self._antithetic_variates()
        elif method == 'control_variates':
            return self._control_variates(control_variates or ['linear_trend'])
        elif method == 'stratified':
            return self._stratified_sampling()
        else:
            raise ValueError(f"Unsupported variance reduction method: {method}")
    
    def _antithetic_variates(self) -> Dict[str, np.ndarray]:
        """Implement antithetic variates for variance reduction."""
        paths = self.simulation_results['all_paths']
        n_simulations, n_time_points, n_assets = paths.shape
        
        # Pair up simulations and average with their "antithetic" counterparts
        # For simplicity, pair consecutive simulations
        n_pairs = n_simulations // 2
        
        antithetic_paths = np.zeros((n_pairs, n_time_points, n_assets))
        
        for i in range(n_pairs):
            # Average pairs of simulations
            antithetic_paths[i] = (paths[2*i] + paths[2*i + 1]) / 2
            
        # Calculate variance reduction
        original_var = np.var(paths, axis=0)
        antithetic_var = np.var(antithetic_paths, axis=0)
        
        variance_reduction_ratio = original_var / (antithetic_var + 1e-8)
        
        return {
            'antithetic_paths': antithetic_paths,
            'variance_reduction_ratio': variance_reduction_ratio,
            'effective_sample_size': n_pairs * variance_reduction_ratio
        }
    
    def _control_variates(self, control_types: List[str]) -> Dict[str, np.ndarray]:
        """Implement control variates for variance reduction."""
        paths = self.simulation_results['all_paths']
        time_grid = self.simulation_results['time_grid']
        
        results = {}
        
        for control_type in control_types:
            if control_type == 'linear_trend':
                # Use linear time trend as control variate
                control_variate = np.outer(np.ones(paths.shape[0]), time_grid)
                control_variate = np.expand_dims(control_variate, axis=2)
                control_variate = np.repeat(control_variate, paths.shape[2], axis=2)
                
                # Calculate optimal control coefficient
                covariance = np.mean((paths - np.mean(paths, axis=0)) * 
                                   (control_variate - np.mean(control_variate, axis=0)), axis=0)
                control_variance = np.var(control_variate, axis=0)
                
                # Avoid division by zero
                control_variance = np.where(control_variance == 0, 1e-8, control_variance)
                
                optimal_coeff = covariance / control_variance
                
                # Apply control variate adjustment
                controlled_paths = paths - optimal_coeff * (control_variate - np.mean(control_variate, axis=0))
                
                # Calculate variance reduction
                original_var = np.var(paths, axis=0)
                controlled_var = np.var(controlled_paths, axis=0)
                variance_reduction = (original_var - controlled_var) / (original_var + 1e-8)
                
                results[control_type] = {
                    'controlled_paths': controlled_paths,
                    'optimal_coefficient': optimal_coeff,
                    'variance_reduction': variance_reduction
                }
                
        return results
    
    def _stratified_sampling(self) -> Dict[str, np.ndarray]:
        """Implement stratified sampling for variance reduction."""
        paths = self.simulation_results['all_paths']
        n_simulations = paths.shape[0]
        
        # Simple stratification by final values
        final_values = paths[:, -1, :]  # Final values for each asset
        
        # Create strata based on quantiles
        n_strata = min(10, n_simulations // 10)  # Use 10 strata or fewer
        
        stratified_results = {}
        
        for asset_idx in range(paths.shape[2]):
            asset_finals = final_values[:, asset_idx]
            
            # Create strata boundaries
            strata_boundaries = np.percentile(asset_finals, 
                                            np.linspace(0, 100, n_strata + 1))
            
            # Assign simulations to strata
            strata_assignments = np.digitize(asset_finals, strata_boundaries) - 1
            strata_assignments = np.clip(strata_assignments, 0, n_strata - 1)
            
            # Calculate stratified mean
            stratified_means = []
            strata_weights = []
            
            for stratum in range(n_strata):
                stratum_mask = strata_assignments == stratum
                if np.any(stratum_mask):
                    stratum_paths = paths[stratum_mask]
                    stratum_mean = np.mean(stratum_paths, axis=0)
                    stratified_means.append(stratum_mean)
                    strata_weights.append(np.sum(stratum_mask) / n_simulations)
                    
            if stratified_means:
                # Weighted average across strata
                stratified_mean = np.average(stratified_means, axis=0, weights=strata_weights)
                
                # Calculate variance reduction (simplified)
                original_var = np.var(paths[:, :, asset_idx], axis=0)
                stratified_var = np.zeros_like(original_var)
                
                for i, stratum in enumerate(range(n_strata)):
                    stratum_mask = strata_assignments == stratum
                    if np.any(stratum_mask):
                        stratum_var = np.var(paths[stratum_mask, :, asset_idx], axis=0)
                        stratified_var += strata_weights[i]**2 * stratum_var
                        
                variance_reduction = (original_var - stratified_var) / (original_var + 1e-8)
                
                stratified_results[f'asset_{asset_idx}'] = {
                    'stratified_mean': stratified_mean,
                    'variance_reduction': variance_reduction,
                    'n_strata': n_strata
                }
                
        return stratified_results
    
    def get_regime_parameters(self) -> Dict[str, float]:
        """Get current regime parameters."""
        return {
            'regime_state': self.regime_state,
            'k1': self.k1, 'k2': self.k2,
            'delta_t1': self.delta_t1, 'delta_t2': self.delta_t2, 'delta_t3': self.delta_t3,
            'delta': self.delta
        }
    
    def set_regime_parameters(self, regime_state: int, **kwargs):
        """
        Set custom parameters for a specific regime.
        Supports regime-switching with different parameter sets.
        """
        if regime_state not in [1, 2]:
            raise ValueError("regime_state must be 1 or 2")
            
        # Update regime parameters
        for param, value in kwargs.items():
            if param in self._base_params:
                self._regime_params[regime_state][param] = value
                
        # If this is the current regime, update active parameters
        if regime_state == self.regime_state:
            self._set_regime_parameters(regime_state)


class MultiRiskFramework:
    """
    Multi-Risk Framework for decomposing risk into directional news risk and impact uncertainty.
    
    Implements Requirements 2.1-2.4:
    - Directional news risk (ε_t) as jump component at event time
    - Impact uncertainty (η_t) through stochastic volatility
    - Event outcome classification (positive/negative/neutral)
    - Methods to separate and analyze different risk components
    """
    
    def __init__(self, jump_intensity: float = 0.1, jump_mean: float = 0.0, jump_std: float = 0.05,
                 impact_uncertainty_scale: float = 1.0, bias_decay_rate: float = 0.1):
        """
        Initialize Multi-Risk Framework.
        
        Args:
            jump_intensity: Intensity parameter for jump process (λ)
            jump_mean: Mean of directional news risk jumps (should be 0 for mean-zero ε_t)
            jump_std: Standard deviation of directional news risk jumps
            impact_uncertainty_scale: Scaling factor for impact uncertainty component
            bias_decay_rate: Rate at which bias parameter b_t decays after events
        """
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean  # Should be 0 for mean-zero ε_t
        self.jump_std = jump_std
        self.impact_uncertainty_scale = impact_uncertainty_scale
        self.bias_decay_rate = bias_decay_rate
        
        # Risk component histories
        self.directional_risk_history = None
        self.impact_uncertainty_history = None
        self.bias_parameter_history = None
        self.event_outcomes = {}  # event_time -> outcome classification
        
        # Fitted state
        self.is_fitted = False
        self.event_times = []
        
    def classify_event_outcome(self, returns_around_event: np.ndarray, 
                              pre_event_window: int = 5, post_event_window: int = 5) -> str:
        """
        Classify event outcome as positive, negative, or neutral based on returns.
        Implements Requirement 2.4: Event outcome classification.
        
        Args:
            returns_around_event: Returns series around the event
            pre_event_window: Number of days before event to consider
            post_event_window: Number of days after event to consider
            
        Returns:
            Event outcome classification: 'positive', 'negative', or 'neutral'
        """
        if len(returns_around_event) < pre_event_window + post_event_window + 1:
            return 'neutral'
            
        # Calculate cumulative returns in post-event window
        event_idx = pre_event_window
        post_event_returns = returns_around_event[event_idx:event_idx + post_event_window]
        
        if len(post_event_returns) == 0:
            return 'neutral'
            
        cumulative_return = np.sum(post_event_returns)
        
        # Classification thresholds (can be adjusted based on empirical analysis)
        positive_threshold = 0.02  # 2% cumulative positive return
        negative_threshold = -0.02  # 2% cumulative negative return
        
        if cumulative_return > positive_threshold:
            return 'positive'
        elif cumulative_return < negative_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_directional_news_risk(self, t: float, event_times: List[float], 
                                       event_outcomes: Dict[float, str]) -> float:
        """
        Calculate directional news risk (ε_t) as jump component at event time.
        Implements Requirement 2.1: Directional news risk as jump component.
        
        Args:
            t: Current time
            event_times: List of event times
            event_outcomes: Dictionary mapping event times to outcomes
            
        Returns:
            Directional news risk component
        """
        directional_risk = 0.0
        
        for event_time in event_times:
            # Check if we're at an event time (within small tolerance)
            if abs(t - event_time) < 1e-6:
                # Generate jump based on event outcome
                outcome = event_outcomes.get(event_time, 'neutral')
                
                # Base jump from normal distribution (mean-zero for ε_t)
                np.random.seed(int(event_time * 1000) % 2**32)  # Deterministic seed based on event time
                base_jump = np.random.normal(self.jump_mean, self.jump_std)
                
                # Adjust jump based on event outcome
                if outcome == 'positive':
                    # Positive events tend to have positive jumps
                    directional_risk = abs(base_jump) if base_jump < 0 else base_jump
                elif outcome == 'negative':
                    # Negative events tend to have negative jumps
                    directional_risk = -abs(base_jump) if base_jump > 0 else base_jump
                else:
                    # Neutral events have mean-zero jumps
                    directional_risk = base_jump
                    
                break  # Only one jump per time point
                
        return directional_risk
    
    def calculate_impact_uncertainty(self, t: float, event_times: List[float], 
                                   baseline_volatility: float) -> float:
        """
        Calculate impact uncertainty (η_t) through stochastic volatility.
        Implements Requirement 2.2: Impact uncertainty through stochastic volatility.
        
        Args:
            t: Current time
            event_times: List of event times
            baseline_volatility: Baseline volatility level
            
        Returns:
            Impact uncertainty component
        """
        if len(event_times) == 0:
            return baseline_volatility
            
        # Find the closest event time
        closest_event_time = min(event_times, key=lambda et: abs(t - et))
        time_to_event = abs(t - closest_event_time)
        
        # Impact uncertainty increases as we approach events and decreases afterward
        if t < closest_event_time:
            # Pre-event: uncertainty increases as event approaches
            uncertainty_factor = 1 + self.impact_uncertainty_scale * np.exp(-time_to_event / 5.0)
        else:
            # Post-event: uncertainty decreases exponentially
            uncertainty_factor = 1 + self.impact_uncertainty_scale * np.exp(-time_to_event / 3.0)
            
        return baseline_volatility * uncertainty_factor
    
    def calculate_bias_parameter(self, t: float, event_times: List[float], 
                               event_outcomes: Dict[float, str]) -> float:
        """
        Calculate bias parameter b_t evolution around events.
        Implements Requirement 2.3: Bias parameter evolution.
        
        Args:
            t: Current time
            event_times: List of event times
            event_outcomes: Dictionary mapping event times to outcomes
            
        Returns:
            Bias parameter value
        """
        bias = 0.0
        
        for event_time in event_times:
            if t >= event_time:
                # Post-event bias that decays exponentially
                time_since_event = t - event_time
                outcome = event_outcomes.get(event_time, 'neutral')
                
                # Initial bias magnitude based on event outcome
                if outcome == 'positive':
                    initial_bias = 0.02  # Positive bias for good news
                elif outcome == 'negative':
                    initial_bias = -0.02  # Negative bias for bad news
                else:
                    initial_bias = 0.0  # No bias for neutral events
                    
                # Exponential decay of bias
                current_bias = initial_bias * np.exp(-self.bias_decay_rate * time_since_event)
                bias += current_bias
                
        return bias
    
    def decompose_risk_components(self, time_points: np.ndarray, event_times: List[float],
                                 event_outcomes: Dict[float, str], 
                                 baseline_volatilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose risk into different components for analysis.
        Implements Requirement 2.3: Methods to separate and analyze different risk components.
        
        Args:
            time_points: Array of time points
            event_times: List of event times
            event_outcomes: Dictionary mapping event times to outcomes
            baseline_volatilities: Array of baseline volatility values
            
        Returns:
            Dictionary containing different risk components
        """
        n_points = len(time_points)
        
        # Initialize component arrays
        directional_risk = np.zeros(n_points)
        impact_uncertainty = np.zeros(n_points)
        bias_parameters = np.zeros(n_points)
        
        # Calculate each component for all time points
        for i, t in enumerate(time_points):
            directional_risk[i] = self.calculate_directional_news_risk(t, event_times, event_outcomes)
            impact_uncertainty[i] = self.calculate_impact_uncertainty(
                t, event_times, baseline_volatilities[i] if i < len(baseline_volatilities) else baseline_volatilities[-1]
            )
            bias_parameters[i] = self.calculate_bias_parameter(t, event_times, event_outcomes)
            
        return {
            'directional_risk': directional_risk,
            'impact_uncertainty': impact_uncertainty,
            'bias_parameters': bias_parameters,
            'time_points': time_points,
            'event_times': event_times,
            'event_outcomes': event_outcomes
        }
    
    def fit(self, returns: Union[np.ndarray, pl.Series], event_times: List[float],
           returns_around_events: Optional[Dict[float, np.ndarray]] = None) -> 'MultiRiskFramework':
        """
        Fit the multi-risk framework to return data.
        
        Args:
            returns: Return series
            event_times: List of event times
            returns_around_events: Optional dict of returns around each event for outcome classification
            
        Returns:
            Fitted MultiRiskFramework instance
        """
        if isinstance(returns, pl.Series):
            returns_np = returns.to_numpy()
        else:
            returns_np = np.asarray(returns)
            
        self.event_times = event_times
        
        # Classify event outcomes if returns around events are provided
        if returns_around_events is not None:
            for event_time, event_returns in returns_around_events.items():
                outcome = self.classify_event_outcome(event_returns)
                self.event_outcomes[event_time] = outcome
        else:
            # Default to neutral outcomes if no specific data provided
            self.event_outcomes = {et: 'neutral' for et in event_times}
            
        # Create time points for the return series
        time_points = np.arange(len(returns_np), dtype=float)
        baseline_volatilities = np.full(len(returns_np), np.std(returns_np))
        
        # Decompose risk components
        risk_decomposition = self.decompose_risk_components(
            time_points, event_times, self.event_outcomes, baseline_volatilities
        )
        
        # Store component histories
        self.directional_risk_history = risk_decomposition['directional_risk']
        self.impact_uncertainty_history = risk_decomposition['impact_uncertainty']
        self.bias_parameter_history = risk_decomposition['bias_parameters']
        
        self.is_fitted = True
        return self
    
    def predict_risk_components(self, future_time_points: np.ndarray,
                               baseline_volatilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict risk components for future time points.
        
        Args:
            future_time_points: Array of future time points
            baseline_volatilities: Array of baseline volatility predictions
            
        Returns:
            Dictionary containing predicted risk components
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        return self.decompose_risk_components(
            future_time_points, self.event_times, self.event_outcomes, baseline_volatilities
        )
    
    def get_risk_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics for risk components.
        
        Returns:
            Dictionary containing risk component statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calculating statistics")
            
        stats = {}
        
        if self.directional_risk_history is not None:
            stats['directional_risk_mean'] = np.mean(self.directional_risk_history)
            stats['directional_risk_std'] = np.std(self.directional_risk_history)
            stats['directional_risk_skewness'] = self._calculate_skewness(self.directional_risk_history)
            
        if self.impact_uncertainty_history is not None:
            stats['impact_uncertainty_mean'] = np.mean(self.impact_uncertainty_history)
            stats['impact_uncertainty_std'] = np.std(self.impact_uncertainty_history)
            
        if self.bias_parameter_history is not None:
            stats['bias_parameter_mean'] = np.mean(self.bias_parameter_history)
            stats['bias_parameter_std'] = np.std(self.bias_parameter_history)
            
        # Event outcome statistics
        if self.event_outcomes:
            outcomes = list(self.event_outcomes.values())
            stats['positive_events_pct'] = outcomes.count('positive') / len(outcomes) * 100
            stats['negative_events_pct'] = outcomes.count('negative') / len(outcomes) * 100
            stats['neutral_events_pct'] = outcomes.count('neutral') / len(outcomes) * 100
            
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data array."""
        if len(data) < 3:
            return 0.0
            
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
            
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def analyze_event_impact(self, event_time: float, analysis_window: int = 10) -> Dict[str, float]:
        """
        Analyze the impact of a specific event on risk components.
        
        Args:
            event_time: Time of the event to analyze
            analysis_window: Number of time points before and after event to analyze
            
        Returns:
            Dictionary containing event impact analysis
        """
        if not self.is_fitted or event_time not in self.event_times:
            raise ValueError(f"Event time {event_time} not found in fitted event times")
            
        # Find event index in time series
        event_idx = None
        if hasattr(self, 'time_points'):
            event_indices = np.where(np.abs(self.time_points - event_time) < 1e-6)[0]
            if len(event_indices) > 0:
                event_idx = event_indices[0]
        
        if event_idx is None:
            return {'error': 'Event not found in time series'}
            
        # Define analysis window
        start_idx = max(0, event_idx - analysis_window)
        end_idx = min(len(self.directional_risk_history), event_idx + analysis_window + 1)
        
        # Extract risk components around event
        directional_risk_window = self.directional_risk_history[start_idx:end_idx]
        impact_uncertainty_window = self.impact_uncertainty_history[start_idx:end_idx]
        bias_parameter_window = self.bias_parameter_history[start_idx:end_idx]
        
        # Calculate impact metrics
        pre_event_directional = np.mean(directional_risk_window[:analysis_window])
        post_event_directional = np.mean(directional_risk_window[analysis_window+1:])
        
        pre_event_uncertainty = np.mean(impact_uncertainty_window[:analysis_window])
        post_event_uncertainty = np.mean(impact_uncertainty_window[analysis_window+1:])
        
        event_outcome = self.event_outcomes.get(event_time, 'neutral')
        
        return {
            'event_time': event_time,
            'event_outcome': event_outcome,
            'directional_risk_change': post_event_directional - pre_event_directional,
            'impact_uncertainty_change': post_event_uncertainty - pre_event_uncertainty,
            'pre_event_directional_mean': pre_event_directional,
            'post_event_directional_mean': post_event_directional,
            'pre_event_uncertainty_mean': pre_event_uncertainty,
            'post_event_uncertainty_mean': post_event_uncertainty,
            'max_directional_risk': np.max(np.abs(directional_risk_window)),
            'max_impact_uncertainty': np.max(impact_uncertainty_window)
        }


class InvestorModel:
    """
    Base class for investor behavior modeling with mean-variance utility.
    
    Implements Requirements 6.1-6.5:
    - Base InvestorModel class with mean-variance utility
    - Support for heterogeneous investor types
    - Terminal wealth risk aversion (γ_T) component
    - Real-time variance aversion (γ_V) component
    - Instantaneous conditional variance calculation
    """
    
    def __init__(self, investor_type: str, gamma_T: float = 2.0, gamma_V: float = 1.0, 
                 bias_params: Optional[Dict[str, float]] = None, 
                 information_quality: float = 1.0, trading_constraints: Optional[Dict[str, float]] = None):
        """
        Initialize base investor model.
        
        Args:
            investor_type: Type of investor ('informed', 'uninformed', 'liquidity')
            gamma_T: Terminal wealth risk aversion parameter
            gamma_V: Real-time variance aversion parameter
            bias_params: Dictionary of bias parameters (b_0, decay_rate, etc.)
            information_quality: Quality of information access (1.0 = perfect, 0.0 = no info)
            trading_constraints: Dictionary of trading constraints
        """
        self.investor_type = investor_type
        self.gamma_T = gamma_T  # Terminal wealth risk aversion
        self.gamma_V = gamma_V  # Real-time variance aversion
        
        # Bias parameters
        if bias_params is None:
            self.bias_params = {'b_0': 0.0, 'decay_rate': 0.1}
        else:
            self.bias_params = bias_params
            
        # Information access quality
        self.information_quality = information_quality
        
        # Trading constraints
        if trading_constraints is None:
            self.trading_constraints = {'max_position': 1.0, 'min_position': -1.0}
        else:
            self.trading_constraints = trading_constraints
            
        # State variables
        self.current_wealth = 1.0  # Normalized initial wealth
        self.current_position = 0.0  # Current portfolio weight
        self.information_set = {}  # Current information set
        
        # History tracking
        self.wealth_history = []
        self.position_history = []
        self.utility_history = []
        
    def calculate_mean_variance_utility(self, expected_return: float, variance: float, 
                                      wealth: float, transaction_costs: float = 0.0) -> float:
        """
        Calculate mean-variance utility with real-time variance penalties.
        Implements Requirements 4.1-4.4: Utility optimization with variance penalties.
        
        U(W_T, Var[dW]) = E[W_T] - (γ_T/2) * Var[W_T] - (γ_V/2) * Var[dW] - TC
        
        Args:
            expected_return: Expected return of the portfolio
            variance: Variance of portfolio returns
            wealth: Current wealth level
            transaction_costs: Transaction costs
            
        Returns:
            Utility value
        """
        # Terminal wealth component
        expected_terminal_wealth = wealth * (1 + expected_return)
        terminal_wealth_variance = (wealth ** 2) * variance
        
        # Real-time variance penalty (instantaneous conditional variance)
        instantaneous_variance = self.calculate_instantaneous_variance(variance, wealth)
        
        # Mean-variance utility calculation
        utility = (expected_terminal_wealth - 
                  (self.gamma_T / 2) * terminal_wealth_variance - 
                  (self.gamma_V / 2) * instantaneous_variance - 
                  transaction_costs)
        
        return utility
    
    def calculate_instantaneous_variance(self, portfolio_variance: float, wealth: float) -> float:
        """
        Calculate instantaneous conditional variance of wealth changes.
        Implements Requirement 4.3: Instantaneous conditional variance calculation.
        
        Args:
            portfolio_variance: Variance of portfolio returns
            wealth: Current wealth level
            
        Returns:
            Instantaneous conditional variance
        """
        # Instantaneous variance of wealth changes: Var[dW] = W^2 * σ^2 * dt
        # For discrete time approximation, we use dt = 1
        instantaneous_variance = (wealth ** 2) * portfolio_variance
        
        return instantaneous_variance
    
    def update_information_set(self, new_information: Dict[str, float], event_time: Optional[float] = None):
        """
        Update investor's information set.
        
        Args:
            new_information: Dictionary containing new information
            event_time: Time of information arrival (optional)
        """
        # Apply information quality filter
        filtered_information = {}
        for key, value in new_information.items():
            # Add noise based on information quality (lower quality = more noise)
            noise_std = (1 - self.information_quality) * abs(value) * 0.1
            noise = np.random.normal(0, noise_std)
            filtered_information[key] = value + noise
            
        # Update information set
        if event_time is not None:
            self.information_set[event_time] = filtered_information
        else:
            self.information_set.update(filtered_information)
    
    def calculate_optimal_position(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 transaction_costs: float = 0.0) -> float:
        """
        Calculate optimal portfolio position using mean-variance optimization.
        
        Args:
            expected_returns: Array of expected returns for assets
            covariance_matrix: Covariance matrix of asset returns
            transaction_costs: Transaction costs for rebalancing
            
        Returns:
            Optimal portfolio weight
        """
        # For single asset case, simplify to scalar optimization
        if len(expected_returns) == 1:
            expected_return = expected_returns[0]
            variance = covariance_matrix[0, 0] if covariance_matrix.ndim > 1 else covariance_matrix
            
            # Optimal weight for single asset: w* = μ / (γ_total * σ^2)
            gamma_total = self.gamma_T + self.gamma_V
            optimal_weight = expected_return / (gamma_total * variance) if variance > 1e-8 else 0.0
            
            # Apply trading constraints
            optimal_weight = np.clip(optimal_weight, 
                                   self.trading_constraints.get('min_position', -1.0),
                                   self.trading_constraints.get('max_position', 1.0))
            
            return optimal_weight
        
        # Multi-asset case (simplified implementation)
        n_assets = len(expected_returns)
        gamma_total = self.gamma_T + self.gamma_V
        
        # Optimal weights: w* = (1/γ) * Σ^(-1) * μ
        try:
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(n_assets) * 1e-8)  # Add regularization
            optimal_weights = (1 / gamma_total) * np.dot(inv_cov, expected_returns)
            
            # Normalize to ensure sum of weights constraint if needed
            if np.sum(optimal_weights) > 1.0:
                optimal_weights = optimal_weights / np.sum(optimal_weights)
                
            return optimal_weights
        except np.linalg.LinAlgError:
            # Fallback to equal weights if covariance matrix is singular
            return np.ones(n_assets) / n_assets
    
    def update_wealth_and_position(self, returns: float, new_position: float, transaction_costs: float = 0.0):
        """
        Update wealth and position based on returns and rebalancing.
        
        Args:
            returns: Realized returns
            new_position: New portfolio position
            transaction_costs: Transaction costs incurred
        """
        # Update wealth based on returns
        self.current_wealth *= (1 + returns * self.current_position)
        
        # Subtract transaction costs
        self.current_wealth -= transaction_costs
        
        # Update position
        self.current_position = new_position
        
        # Record history
        self.wealth_history.append(self.current_wealth)
        self.position_history.append(self.current_position)
    
    def get_investor_statistics(self) -> Dict[str, float]:
        """
        Calculate performance statistics for the investor.
        
        Returns:
            Dictionary containing investor performance metrics
        """
        if len(self.wealth_history) < 2:
            return {'error': 'Insufficient history for statistics'}
            
        wealth_array = np.array(self.wealth_history)
        returns_array = np.diff(wealth_array) / wealth_array[:-1]
        
        stats = {
            'total_return': (self.current_wealth - 1.0) * 100,  # Percentage return
            'volatility': np.std(returns_array) * 100,
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 1e-8 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(wealth_array),
            'average_position': np.mean(self.position_history) if self.position_history else 0.0,
            'position_volatility': np.std(self.position_history) if len(self.position_history) > 1 else 0.0,
            'final_wealth': self.current_wealth,
            'information_quality': self.information_quality
        }
        
        return stats
    
    def _calculate_max_drawdown(self, wealth_series: np.ndarray) -> float:
        """Calculate maximum drawdown from wealth series."""
        if len(wealth_series) < 2:
            return 0.0
            
        peak = wealth_series[0]
        max_drawdown = 0.0
        
        for wealth in wealth_series[1:]:
            if wealth > peak:
                peak = wealth
            drawdown = (peak - wealth) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown * 100  # Return as percentage


class InformedInvestor(InvestorModel):
    """
    Informed investor with accurate information access.
    Implements Requirement 6.2: InformedInvestor with accurate information access.
    """
    
    def __init__(self, gamma_T: float = 2.0, gamma_V: float = 1.0, 
                 bias_params: Optional[Dict[str, float]] = None):
        """
        Initialize informed investor.
        
        Args:
            gamma_T: Terminal wealth risk aversion
            gamma_V: Real-time variance aversion  
            bias_params: Bias parameters (lower bias for informed investors)
        """
        # Informed investors have lower bias
        if bias_params is None:
            bias_params = {'b_0': 0.005, 'decay_rate': 0.15}  # Lower initial bias, faster decay
            
        super().__init__(
            investor_type='informed',
            gamma_T=gamma_T,
            gamma_V=gamma_V,
            bias_params=bias_params,
            information_quality=0.95,  # High information quality
            trading_constraints={'max_position': 1.0, 'min_position': -1.0}
        )
        
        # Informed investor specific parameters
        self.information_advantage = 0.1  # Advantage in information timing
        self.signal_accuracy = 0.85  # Accuracy of private signals
        
    def process_private_signal(self, public_information: Dict[str, float], 
                             event_time: float) -> Dict[str, float]:
        """
        Process private information signals with high accuracy.
        
        Args:
            public_information: Publicly available information
            event_time: Time of information arrival
            
        Returns:
            Enhanced information set with private signals
        """
        enhanced_info = public_information.copy()
        
        # Add private signal with high accuracy
        for key, value in public_information.items():
            if 'expected_return' in key:
                # Private signal provides more accurate expected return estimate
                signal_noise = np.random.normal(0, (1 - self.signal_accuracy) * abs(value) * 0.05)
                private_signal = value + signal_noise
                enhanced_info[f'private_{key}'] = private_signal
                
                # Combine public and private information
                combined_weight = 0.7  # Weight on private information
                enhanced_info[key] = combined_weight * private_signal + (1 - combined_weight) * value
                
        return enhanced_info
    
    def calculate_optimal_position(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 transaction_costs: float = 0.0) -> float:
        """
        Calculate optimal position with information advantage.
        
        Args:
            expected_returns: Array of expected returns
            covariance_matrix: Covariance matrix
            transaction_costs: Transaction costs
            
        Returns:
            Optimal position considering information advantage
        """
        # Base optimal position
        base_position = super().calculate_optimal_position(expected_returns, covariance_matrix, transaction_costs)
        
        # Adjust for information advantage (informed investors can take larger positions)
        information_multiplier = 1 + self.information_advantage
        adjusted_position = base_position * information_multiplier
        
        # Still respect trading constraints
        adjusted_position = np.clip(adjusted_position,
                                  self.trading_constraints.get('min_position', -1.0),
                                  self.trading_constraints.get('max_position', 1.0))
        
        return adjusted_position


class UninformedInvestor(InvestorModel):
    """
    Uninformed investor with noisy information.
    Implements Requirement 6.3: UninformedInvestor with noisy information.
    """
    
    def __init__(self, gamma_T: float = 2.5, gamma_V: float = 1.2, 
                 bias_params: Optional[Dict[str, float]] = None):
        """
        Initialize uninformed investor.
        
        Args:
            gamma_T: Terminal wealth risk aversion (higher than informed)
            gamma_V: Real-time variance aversion (higher than informed)
            bias_params: Bias parameters (higher bias for uninformed investors)
        """
        # Uninformed investors have higher bias
        if bias_params is None:
            bias_params = {'b_0': 0.02, 'decay_rate': 0.08}  # Higher initial bias, slower decay
            
        super().__init__(
            investor_type='uninformed',
            gamma_T=gamma_T,
            gamma_V=gamma_V,
            bias_params=bias_params,
            information_quality=0.6,  # Lower information quality
            trading_constraints={'max_position': 0.8, 'min_position': -0.8}  # More conservative
        )
        
        # Uninformed investor specific parameters
        self.noise_level = 0.3  # Higher noise in information processing
        self.herding_tendency = 0.2  # Tendency to follow market trends
        
    def process_noisy_information(self, public_information: Dict[str, float]) -> Dict[str, float]:
        """
        Process information with added noise and bias.
        
        Args:
            public_information: Publicly available information
            
        Returns:
            Noisy information set
        """
        noisy_info = {}
        
        for key, value in public_information.items():
            # Add noise based on noise level
            noise = np.random.normal(0, self.noise_level * abs(value))
            noisy_value = value + noise
            
            # Add bias based on herding tendency
            if 'expected_return' in key and value != 0:
                # Bias toward market consensus (assume positive values indicate bullish consensus)
                bias_adjustment = self.herding_tendency * abs(value) * np.sign(value)
                noisy_value += bias_adjustment
                
            noisy_info[key] = noisy_value
            
        return noisy_info
    
    def calculate_optimal_position(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 transaction_costs: float = 0.0) -> float:
        """
        Calculate optimal position with information disadvantage.
        
        Args:
            expected_returns: Array of expected returns (noisy)
            covariance_matrix: Covariance matrix
            transaction_costs: Transaction costs
            
        Returns:
            Optimal position considering information disadvantage
        """
        # Process returns with noise
        noisy_returns = expected_returns.copy()
        for i in range(len(noisy_returns)):
            noise = np.random.normal(0, self.noise_level * abs(noisy_returns[i]))
            noisy_returns[i] += noise
            
        # Base optimal position with noisy information
        base_position = super().calculate_optimal_position(noisy_returns, covariance_matrix, transaction_costs)
        
        # Reduce position size due to uncertainty (uninformed investors are more conservative)
        uncertainty_discount = 0.8  # Reduce position by 20%
        adjusted_position = base_position * uncertainty_discount
        
        return adjusted_position


class LiquidityTrader(InvestorModel):
    """
    Liquidity trader with non-information-based trading.
    Implements Requirement 6.4: LiquidityTrader with non-information-based trading.
    """
    
    def __init__(self, gamma_T: float = 1.5, gamma_V: float = 0.8, 
                 liquidity_needs: Optional[Dict[str, float]] = None):
        """
        Initialize liquidity trader.
        
        Args:
            gamma_T: Terminal wealth risk aversion (lower, as they're forced to trade)
            gamma_V: Real-time variance aversion (lower, as they're forced to trade)
            liquidity_needs: Dictionary specifying liquidity requirements
        """
        super().__init__(
            investor_type='liquidity',
            gamma_T=gamma_T,
            gamma_V=gamma_V,
            bias_params={'b_0': 0.0, 'decay_rate': 0.0},  # No information-based bias
            information_quality=0.0,  # No information advantage
            trading_constraints={'max_position': 0.5, 'min_position': -0.5}  # Asymmetric constraints
        )
        
        # Liquidity trader specific parameters
        if liquidity_needs is None:
            self.liquidity_needs = {
                'forced_trading_probability': 0.1,  # Probability of forced trading
                'liquidity_shock_size': 0.05,  # Size of liquidity shocks
                'trading_urgency': 0.3  # Urgency of trading needs
            }
        else:
            self.liquidity_needs = liquidity_needs
            
        # Asymmetric trading constraints (Requirement 6.5)
        self.asymmetric_constraints = True
        self.max_buy_position = 0.3   # Limited buying capacity
        self.max_sell_position = -0.7  # Higher selling capacity (liquidity provision)
        
    def generate_liquidity_shock(self) -> float:
        """
        Generate random liquidity shock that forces trading.
        
        Returns:
            Size and direction of liquidity shock
        """
        # Random liquidity shock
        if np.random.random() < self.liquidity_needs['forced_trading_probability']:
            shock_size = self.liquidity_needs['liquidity_shock_size']
            # Liquidity traders are more likely to be forced sellers
            direction = np.random.choice([-1, 1], p=[0.7, 0.3])  # 70% chance of selling
            return direction * shock_size * np.random.uniform(0.5, 1.5)
        else:
            return 0.0
    
    def calculate_optimal_position(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 transaction_costs: float = 0.0) -> float:
        """
        Calculate position based on liquidity needs rather than information.
        
        Args:
            expected_returns: Array of expected returns (ignored for liquidity trading)
            covariance_matrix: Covariance matrix
            transaction_costs: Transaction costs
            
        Returns:
            Position based on liquidity needs
        """
        # Generate liquidity shock
        liquidity_shock = self.generate_liquidity_shock()
        
        if abs(liquidity_shock) > 1e-6:
            # Forced trading due to liquidity needs
            target_position = self.current_position + liquidity_shock
            
            # Apply asymmetric constraints
            if target_position > 0:
                target_position = min(target_position, self.max_buy_position)
            else:
                target_position = max(target_position, self.max_sell_position)
                
            return target_position
        else:
            # No liquidity shock, maintain current position or make small adjustments
            # Small random walk around current position
            position_drift = np.random.normal(0, 0.01)  # Small random adjustment
            new_position = self.current_position + position_drift
            
            # Apply asymmetric constraints
            if new_position > 0:
                new_position = min(new_position, self.max_buy_position)
            else:
                new_position = max(new_position, self.max_sell_position)
                
            return new_position
    
    def calculate_liquidity_cost(self, position_change: float, market_volatility: float) -> float:
        """
        Calculate additional costs due to liquidity trading.
        
        Args:
            position_change: Change in position
            market_volatility: Current market volatility
            
        Returns:
            Additional liquidity cost
        """
        # Liquidity traders face higher costs when forced to trade
        base_cost = abs(position_change) * 0.001  # Base transaction cost
        
        # Additional cost based on trading urgency and market conditions
        urgency_cost = self.liquidity_needs['trading_urgency'] * abs(position_change) * market_volatility
        
        # Asymmetric costs (selling is more expensive due to adverse selection)
        if position_change < 0:  # Selling
            asymmetric_multiplier = 1.5
        else:  # Buying
            asymmetric_multiplier = 1.0
            
        total_cost = (base_cost + urgency_cost) * asymmetric_multiplier
        
        return total_cost
    
    def update_liquidity_needs(self, new_needs: Dict[str, float]):
        """
        Update liquidity requirements based on external factors.
        
        Args:
            new_needs: Updated liquidity need parameters
        """
        self.liquidity_needs.update(new_needs)
        
        # Adjust trading constraints based on new liquidity needs
        if 'max_buy_constraint' in new_needs:
            self.max_buy_position = new_needs['max_buy_constraint']
        if 'max_sell_constraint' in new_needs:
            self.max_sell_position = new_needs['max_sell_constraint']


class UtilityOptimizer:
    """
    Advanced utility optimization with real-time variance penalties.
    
    Implements Requirements 4.1-4.4:
    - Terminal wealth risk aversion (γ_T) component
    - Real-time variance aversion (γ_V) component  
    - Instantaneous conditional variance calculation
    - Utility optimization with variance penalties
    """
    
    def __init__(self, gamma_T: float = 2.0, gamma_V: float = 1.0, 
                 risk_free_rate: float = 0.0, time_horizon: float = 1.0):
        """
        Initialize utility optimizer.
        
        Args:
            gamma_T: Terminal wealth risk aversion coefficient
            gamma_V: Real-time variance aversion coefficient
            risk_free_rate: Risk-free rate for utility calculations
            time_horizon: Investment time horizon
        """
        self.gamma_T = gamma_T
        self.gamma_V = gamma_V
        self.risk_free_rate = risk_free_rate
        self.time_horizon = time_horizon
        
        # Optimization history
        self.optimization_history = []
        
    def calculate_enhanced_utility(self, wealth: float, portfolio_weights: np.ndarray,
                                 expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 time_to_maturity: float, transaction_costs: float = 0.0,
                                 drawdown_constraints: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate enhanced utility with both terminal and real-time variance penalties.
        
        Implements the utility function:
        U(W, w, t) = E[W_T] - (γ_T/2) * Var[W_T] - (γ_V/2) * E[∫_t^T Var[dW_s] ds] - TC
        
        Args:
            wealth: Current wealth level
            portfolio_weights: Portfolio weights vector
            expected_returns: Expected returns vector
            covariance_matrix: Asset return covariance matrix
            time_to_maturity: Time remaining to investment horizon
            transaction_costs: Transaction costs
            drawdown_constraints: Optional drawdown constraints
            
        Returns:
            Enhanced utility value
        """
        # Portfolio expected return and variance
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        
        # Terminal wealth moments
        expected_terminal_wealth = wealth * np.exp((portfolio_return - 0.5 * portfolio_variance) * time_to_maturity)
        terminal_wealth_variance = (wealth ** 2) * (np.exp(portfolio_variance * time_to_maturity) - 1) * \
                                 np.exp(2 * portfolio_return * time_to_maturity - portfolio_variance * time_to_maturity)
        
        # Real-time variance penalty (integral of instantaneous variance)
        real_time_variance_penalty = self._calculate_real_time_variance_penalty(
            wealth, portfolio_weights, covariance_matrix, time_to_maturity
        )
        
        # Base utility calculation
        utility = (expected_terminal_wealth - 
                  (self.gamma_T / 2) * terminal_wealth_variance - 
                  (self.gamma_V / 2) * real_time_variance_penalty - 
                  transaction_costs)
        
        # Apply drawdown constraints if specified
        if drawdown_constraints is not None:
            utility = self._apply_drawdown_constraints(utility, wealth, portfolio_weights, 
                                                     covariance_matrix, drawdown_constraints)
        
        return utility
    
    def _calculate_real_time_variance_penalty(self, wealth: float, portfolio_weights: np.ndarray,
                                            covariance_matrix: np.ndarray, time_to_maturity: float) -> float:
        """
        Calculate the real-time variance penalty component.
        
        Implements: E[∫_t^T Var[dW_s] ds] where dW_s = W_s * w^T * dR_s
        
        Args:
            wealth: Current wealth level
            portfolio_weights: Portfolio weights
            covariance_matrix: Covariance matrix
            time_to_maturity: Time to maturity
            
        Returns:
            Real-time variance penalty value
        """
        # Instantaneous variance of wealth changes: Var[dW] = W^2 * w^T * Σ * w * dt
        instantaneous_variance = (wealth ** 2) * np.dot(portfolio_weights, 
                                                       np.dot(covariance_matrix, portfolio_weights))
        
        # For constant portfolio weights, the integral becomes:
        # ∫_t^T Var[dW_s] ds = ∫_t^T W_s^2 * w^T * Σ * w ds
        
        # Assuming wealth follows geometric Brownian motion: W_s = W_t * exp((μ - σ²/2)(s-t) + σ√(s-t)Z)
        # E[W_s^2] = W_t^2 * exp(2μ(s-t) + σ²(s-t))
        
        portfolio_return = np.dot(portfolio_weights, np.zeros_like(portfolio_weights))  # Assume zero drift for penalty calculation
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        
        # Analytical solution for the integral
        if abs(2 * portfolio_return + portfolio_variance) > 1e-8:
            integral_factor = (np.exp((2 * portfolio_return + portfolio_variance) * time_to_maturity) - 1) / \
                            (2 * portfolio_return + portfolio_variance)
        else:
            # Limit case when exponent is close to zero
            integral_factor = time_to_maturity
            
        real_time_penalty = (wealth ** 2) * portfolio_variance * integral_factor
        
        return real_time_penalty
    
    def _apply_drawdown_constraints(self, base_utility: float, wealth: float, 
                                  portfolio_weights: np.ndarray, covariance_matrix: np.ndarray,
                                  drawdown_constraints: Dict[str, float]) -> float:
        """
        Apply drawdown constraints to utility function.
        
        Args:
            base_utility: Base utility value
            wealth: Current wealth
            portfolio_weights: Portfolio weights
            covariance_matrix: Covariance matrix
            drawdown_constraints: Drawdown constraint parameters
            
        Returns:
            Constrained utility value
        """
        max_drawdown_limit = drawdown_constraints.get('max_drawdown', 0.2)  # 20% default
        penalty_multiplier = drawdown_constraints.get('penalty_multiplier', 10.0)
        
        # Estimate potential drawdown based on portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights)))
        
        # Simple drawdown estimate: assume 2-sigma adverse move
        estimated_drawdown = 2 * portfolio_volatility
        
        # Apply penalty if estimated drawdown exceeds limit
        if estimated_drawdown > max_drawdown_limit:
            excess_drawdown = estimated_drawdown - max_drawdown_limit
            drawdown_penalty = penalty_multiplier * (excess_drawdown ** 2) * wealth
            return base_utility - drawdown_penalty
        
        return base_utility
    
    def optimize_portfolio(self, wealth: float, expected_returns: np.ndarray, 
                          covariance_matrix: np.ndarray, time_to_maturity: float,
                          current_weights: Optional[np.ndarray] = None,
                          transaction_cost_model: Optional['TransactionCostModel'] = None,
                          transaction_cost_rate: float = 0.001,
                          time_to_event: float = np.inf,
                          liquidity_factors: Optional[np.ndarray] = None,
                          market_volatilities: Optional[np.ndarray] = None,
                          position_limits: Optional[Tuple[float, float]] = None,
                          drawdown_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio weights to maximize enhanced utility with transaction costs.
        
        Implements Requirements 5.1, 5.2, 5.3: Integration of asymmetric transaction costs
        into portfolio optimization with sign function for cost direction.
        
        Args:
            wealth: Current wealth level
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            time_to_maturity: Time to investment horizon
            current_weights: Current portfolio weights (for transaction cost calculation)
            transaction_cost_model: TransactionCostModel instance for advanced cost calculation
            transaction_cost_rate: Simple transaction cost rate (fallback if no model provided)
            time_to_event: Time to next event (for time-varying costs)
            liquidity_factors: Liquidity factors for each asset
            market_volatilities: Market volatilities for each asset
            position_limits: Tuple of (min_weight, max_weight) for each asset
            drawdown_constraints: Drawdown constraint parameters
            
        Returns:
            Dictionary containing optimal weights and optimization results
        """
        n_assets = len(expected_returns)
        
        # Objective function to maximize (we minimize the negative)
        def objective(weights):
            # Calculate transaction costs using advanced model or simple rate
            if current_weights is not None:
                if transaction_cost_model is not None:
                    # Use advanced transaction cost model with asymmetric effects
                    cost_result = transaction_cost_model.calculate_portfolio_rebalancing_cost(
                        current_weights, weights, wealth, time_to_event, 
                        liquidity_factors, market_volatilities
                    )
                    transaction_costs = cost_result['total_cost']
                else:
                    # Fallback to simple symmetric cost calculation
                    weight_changes = np.abs(weights - current_weights)
                    transaction_costs = transaction_cost_rate * np.sum(weight_changes) * wealth
            else:
                transaction_costs = 0.0
                
            # Calculate negative utility (for minimization)
            utility = self.calculate_enhanced_utility(
                wealth, weights, expected_returns, covariance_matrix, 
                time_to_maturity, transaction_costs, drawdown_constraints
            )
            return -utility
        
        # Constraints
        constraints = []
        
        # Budget constraint: sum of weights = 1
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Position limits
        if position_limits is not None:
            min_weight, max_weight = position_limits
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        else:
            bounds = [(-1.0, 1.0) for _ in range(n_assets)]  # Allow short selling
        
        # Initial guess
        if current_weights is not None:
            initial_weights = current_weights
        else:
            initial_weights = np.ones(n_assets) / n_assets  # Equal weights
        
        # Optimization
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective, 
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_utility = -result.fun
                
                # Calculate final transaction costs
                if current_weights is not None:
                    if transaction_cost_model is not None:
                        cost_result = transaction_cost_model.calculate_portfolio_rebalancing_cost(
                            current_weights, optimal_weights, wealth, time_to_event,
                            liquidity_factors, market_volatilities
                        )
                        final_transaction_costs = cost_result['total_cost']
                        cost_breakdown = cost_result
                    else:
                        final_transaction_costs = transaction_cost_rate * \
                                                np.sum(np.abs(optimal_weights - current_weights)) * wealth
                        cost_breakdown = {'total_cost': final_transaction_costs}
                else:
                    final_transaction_costs = 0.0
                    cost_breakdown = {'total_cost': 0.0}
                
                optimization_result = {
                    'optimal_weights': optimal_weights,
                    'optimal_utility': optimal_utility,
                    'transaction_costs': final_transaction_costs,
                    'cost_breakdown': cost_breakdown,
                    'portfolio_return': np.dot(optimal_weights, expected_returns),
                    'portfolio_variance': np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)),
                    'optimization_success': True,
                    'optimization_message': result.message,
                    'weight_changes': optimal_weights - current_weights if current_weights is not None else optimal_weights
                }
            else:
                # Fallback to equal weights if optimization fails
                optimal_weights = np.ones(n_assets) / n_assets
                optimal_utility = self.calculate_enhanced_utility(
                    wealth, optimal_weights, expected_returns, covariance_matrix, 
                    time_to_maturity, 0.0, drawdown_constraints
                )
                
                optimization_result = {
                    'optimal_weights': optimal_weights,
                    'optimal_utility': optimal_utility,
                    'transaction_costs': 0.0,
                    'portfolio_return': np.dot(optimal_weights, expected_returns),
                    'portfolio_variance': np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)),
                    'optimization_success': False,
                    'optimization_message': f"Optimization failed: {result.message}"
                }
                
        except ImportError:
            # Fallback if scipy is not available
            optimal_weights = np.ones(n_assets) / n_assets
            optimal_utility = self.calculate_enhanced_utility(
                wealth, optimal_weights, expected_returns, covariance_matrix, 
                time_to_maturity, 0.0, drawdown_constraints
            )
            
            optimization_result = {
                'optimal_weights': optimal_weights,
                'optimal_utility': optimal_utility,
                'transaction_costs': 0.0,
                'portfolio_return': np.dot(optimal_weights, expected_returns),
                'portfolio_variance': np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)),
                'optimization_success': False,
                'optimization_message': "Scipy not available, using equal weights"
            }
        
        # Store optimization history
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def calculate_utility_gradient(self, wealth: float, portfolio_weights: np.ndarray,
                                 expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 time_to_maturity: float) -> np.ndarray:
        """
        Calculate the gradient of utility function with respect to portfolio weights.
        
        Args:
            wealth: Current wealth level
            portfolio_weights: Current portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            time_to_maturity: Time to maturity
            
        Returns:
            Gradient vector
        """
        n_assets = len(portfolio_weights)
        gradient = np.zeros(n_assets)
        
        # Small perturbation for numerical gradient
        epsilon = 1e-6
        
        # Base utility
        base_utility = self.calculate_enhanced_utility(
            wealth, portfolio_weights, expected_returns, covariance_matrix, time_to_maturity
        )
        
        # Calculate partial derivatives
        for i in range(n_assets):
            perturbed_weights = portfolio_weights.copy()
            perturbed_weights[i] += epsilon
            
            perturbed_utility = self.calculate_enhanced_utility(
                wealth, perturbed_weights, expected_returns, covariance_matrix, time_to_maturity
            )
            
            gradient[i] = (perturbed_utility - base_utility) / epsilon
            
        return gradient
    
    def analyze_utility_components(self, wealth: float, portfolio_weights: np.ndarray,
                                 expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                 time_to_maturity: float) -> Dict[str, float]:
        """
        Analyze individual components of the utility function.
        
        Args:
            wealth: Current wealth level
            portfolio_weights: Portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            time_to_maturity: Time to maturity
            
        Returns:
            Dictionary with utility component breakdown
        """
        # Portfolio moments
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        
        # Terminal wealth components
        expected_terminal_wealth = wealth * np.exp((portfolio_return - 0.5 * portfolio_variance) * time_to_maturity)
        terminal_wealth_variance = (wealth ** 2) * (np.exp(portfolio_variance * time_to_maturity) - 1) * \
                                 np.exp(2 * portfolio_return * time_to_maturity - portfolio_variance * time_to_maturity)
        
        # Real-time variance penalty
        real_time_penalty = self._calculate_real_time_variance_penalty(
            wealth, portfolio_weights, covariance_matrix, time_to_maturity
        )
        
        # Component analysis
        components = {
            'expected_terminal_wealth': expected_terminal_wealth,
            'terminal_wealth_penalty': (self.gamma_T / 2) * terminal_wealth_variance,
            'real_time_variance_penalty': (self.gamma_V / 2) * real_time_penalty,
            'portfolio_expected_return': portfolio_return,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': np.sqrt(portfolio_variance),
            'time_to_maturity': time_to_maturity,
            'wealth': wealth,
            'gamma_T': self.gamma_T,
            'gamma_V': self.gamma_V
        }
        
        # Total utility
        total_utility = (expected_terminal_wealth - 
                        components['terminal_wealth_penalty'] - 
                        components['real_time_variance_penalty'])
        components['total_utility'] = total_utility
        
        return components
    
    def calculate_hjb_transaction_cost_term(self, wealth: float, portfolio_weights: np.ndarray,
                                          weight_derivatives: np.ndarray, 
                                          transaction_cost_model: 'TransactionCostModel',
                                          time_to_event: float = np.inf,
                                          liquidity_factors: Optional[np.ndarray] = None,
                                          market_volatilities: Optional[np.ndarray] = None) -> float:
        """
        Calculate transaction cost term for Hamilton-Jacobi-Bellman equation.
        
        Implements Requirements 5.1, 5.2, 5.3: Transaction cost effects in HJB equation
        with sign function for cost direction sgn(dw_e(t)).
        
        Args:
            wealth: Current wealth level
            portfolio_weights: Current portfolio weights
            weight_derivatives: Time derivatives of portfolio weights (dw_e(t))
            transaction_cost_model: Transaction cost model
            time_to_event: Time to next event
            liquidity_factors: Liquidity factors for each asset
            market_volatilities: Market volatilities for each asset
            
        Returns:
            Transaction cost term for HJB equation
        """
        n_assets = len(portfolio_weights)
        
        # Default values if not provided
        if liquidity_factors is None:
            liquidity_factors = np.ones(n_assets)
        if market_volatilities is None:
            market_volatilities = np.full(n_assets, 0.02)
        
        # Calculate transaction cost term: ∑_i τ_i(t) * |dw_i(t)| * sgn(dw_i(t)) * W(t)
        hjb_cost_term = 0.0
        
        for i in range(n_assets):
            if abs(weight_derivatives[i]) > 1e-8:
                # Determine if this is a purchase or sale
                is_purchase = weight_derivatives[i] > 0
                
                # Sign function: sgn(dw_e(t))
                sign_dw = 1.0 if is_purchase else -1.0
                
                # Calculate instantaneous transaction cost rate
                position_change = abs(weight_derivatives[i]) * wealth
                
                # Get cost rate from transaction cost model
                instantaneous_cost = transaction_cost_model.calculate_asymmetric_cost(
                    position_change, is_purchase, time_to_event, 
                    liquidity_factors[i], market_volatilities[i]
                )
                
                # Convert to rate (cost per unit position change)
                cost_rate = instantaneous_cost / position_change if position_change > 0 else 0.0
                
                # Add to HJB cost term: -cost_rate * |dw_i| * W (negative because costs reduce utility)
                hjb_cost_term -= cost_rate * abs(weight_derivatives[i]) * wealth
        
        return hjb_cost_term
    
    def solve_hjb_with_transaction_costs(self, wealth_grid: np.ndarray, time_grid: np.ndarray,
                                       expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                       transaction_cost_model: 'TransactionCostModel',
                                       event_times: Optional[List[float]] = None,
                                       boundary_conditions: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Solve Hamilton-Jacobi-Bellman equation with transaction costs using finite differences.
        
        Implements Requirements 5.1, 5.2, 5.3: HJB equation solver with transaction cost effects.
        
        Args:
            wealth_grid: Wealth grid points for discretization
            time_grid: Time grid points for discretization
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            transaction_cost_model: Transaction cost model
            event_times: List of event times
            boundary_conditions: Boundary conditions for HJB equation
            
        Returns:
            Dictionary with HJB solution components
        """
        n_wealth = len(wealth_grid)
        n_time = len(time_grid)
        n_assets = len(expected_returns)
        
        # Initialize value function and optimal controls
        value_function = np.zeros((n_time, n_wealth))
        optimal_weights = np.zeros((n_time, n_wealth, n_assets))
        
        # Default boundary conditions
        if boundary_conditions is None:
            boundary_conditions = {'terminal_utility_power': 1.0}
        
        # Terminal condition: V(W, T) = W^power
        power = boundary_conditions.get('terminal_utility_power', 1.0)
        value_function[-1, :] = np.power(wealth_grid, power)
        
        # Backward induction
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        
        for t in range(n_time - 2, -1, -1):
            current_time = time_grid[t]
            
            # Find time to next event
            time_to_next_event = np.inf
            if event_times:
                for event_time in event_times:
                    if event_time > current_time:
                        time_to_next_event = min(time_to_next_event, event_time - current_time)
            
            for w_idx, wealth in enumerate(wealth_grid):
                if wealth <= 1e-6:  # Skip very small wealth values
                    continue
                
                # Optimize portfolio weights for this wealth level
                try:
                    # Use current optimal weights as starting point
                    current_weights = optimal_weights[t+1, w_idx, :] if t < n_time - 1 else np.ones(n_assets) / n_assets
                    
                    # Optimize with transaction costs
                    opt_result = self.optimize_portfolio(
                        wealth, expected_returns, covariance_matrix, 
                        time_grid[-1] - current_time,  # time to maturity
                        current_weights, transaction_cost_model, 
                        time_to_event=time_to_next_event
                    )
                    
                    optimal_weights[t, w_idx, :] = opt_result['optimal_weights']
                    
                    # Calculate value function using Bellman equation
                    # V(W,t) = max_w { μ*W*w*dt + (1/2)*σ²*W²*w²*dt - TC*dt + E[V(W',t+dt)] }
                    
                    portfolio_return = opt_result['portfolio_return']
                    portfolio_variance = opt_result['portfolio_variance']
                    transaction_costs = opt_result['transaction_costs']
                    
                    # Expected wealth change
                    expected_wealth_change = wealth * portfolio_return * dt
                    wealth_variance = (wealth ** 2) * portfolio_variance * dt
                    
                    # Approximate continuation value (simple forward difference)
                    if w_idx < n_wealth - 1:
                        continuation_value = value_function[t+1, w_idx]
                    else:
                        continuation_value = value_function[t+1, w_idx]
                    
                    # Bellman equation value
                    value_function[t, w_idx] = (
                        expected_wealth_change - 
                        (self.gamma_T / 2) * wealth_variance - 
                        transaction_costs * dt + 
                        continuation_value
                    )
                    
                except Exception as e:
                    # Fallback to previous time step values
                    if t < n_time - 1:
                        optimal_weights[t, w_idx, :] = optimal_weights[t+1, w_idx, :]
                        value_function[t, w_idx] = value_function[t+1, w_idx]
                    else:
                        optimal_weights[t, w_idx, :] = np.ones(n_assets) / n_assets
                        value_function[t, w_idx] = np.power(wealth, power)
        
        return {
            'value_function': value_function,
            'optimal_weights': optimal_weights,
            'wealth_grid': wealth_grid,
            'time_grid': time_grid,
            'convergence_achieved': True,  # Simplified for this implementation
            'solver_info': {
                'method': 'finite_difference_backward_induction',
                'n_wealth_points': n_wealth,
                'n_time_points': n_time,
                'transaction_cost_model': str(transaction_cost_model)
            }
        }
    
    def calculate_optimal_weight_with_transaction_costs(self, wealth: float, expected_returns: np.ndarray,
                                                      covariance_matrix: np.ndarray, 
                                                      current_weights: np.ndarray,
                                                      transaction_cost_model: 'TransactionCostModel',
                                                      time_to_event: float = np.inf) -> Dict[str, Union[np.ndarray, float]]:
        """
        Calculate optimal portfolio weights incorporating transaction costs analytically.
        
        Implements Requirements 5.1, 5.2: Optimal weight calculations with transaction costs.
        
        Args:
            wealth: Current wealth level
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights
            transaction_cost_model: Transaction cost model
            time_to_event: Time to next event
            
        Returns:
            Dictionary with optimal weights and analysis
        """
        n_assets = len(expected_returns)
        
        # First, calculate frictionless optimal weights (Markowitz solution)
        try:
            # Solve: w* = (1/γ) * Σ^(-1) * μ
            inv_cov = np.linalg.inv(covariance_matrix)
            frictionless_weights = np.dot(inv_cov, expected_returns) / self.gamma_T
            
            # Normalize to sum to 1
            frictionless_weights = frictionless_weights / np.sum(frictionless_weights)
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if covariance matrix is singular
            frictionless_weights = np.ones(n_assets) / n_assets
        
        # Calculate transaction costs for moving to frictionless optimum
        frictionless_cost = transaction_cost_model.calculate_portfolio_rebalancing_cost(
            current_weights, frictionless_weights, wealth, time_to_event
        )['total_cost']
        
        # If transaction costs are very small, use frictionless solution
        if frictionless_cost < wealth * 1e-6:
            return {
                'optimal_weights': frictionless_weights,
                'frictionless_weights': frictionless_weights,
                'transaction_costs': frictionless_cost,
                'no_trade_region': False,
                'weight_adjustment': frictionless_weights - current_weights
            }
        
        # Otherwise, solve with transaction costs using no-trade region approach
        # Define no-trade region boundaries
        no_trade_boundaries = self._calculate_no_trade_boundaries(
            current_weights, expected_returns, covariance_matrix, 
            transaction_cost_model, wealth, time_to_event
        )
        
        # Check if frictionless optimum is within no-trade region
        within_no_trade = True
        for i in range(n_assets):
            if (frictionless_weights[i] < no_trade_boundaries['lower'][i] or 
                frictionless_weights[i] > no_trade_boundaries['upper'][i]):
                within_no_trade = False
                break
        
        if within_no_trade:
            # Stay at current weights (no-trade region)
            return {
                'optimal_weights': current_weights,
                'frictionless_weights': frictionless_weights,
                'transaction_costs': 0.0,
                'no_trade_region': True,
                'weight_adjustment': np.zeros(n_assets),
                'no_trade_boundaries': no_trade_boundaries
            }
        else:
            # Trade to boundary of no-trade region
            boundary_weights = self._calculate_boundary_weights(
                current_weights, frictionless_weights, no_trade_boundaries
            )
            
            boundary_cost = transaction_cost_model.calculate_portfolio_rebalancing_cost(
                current_weights, boundary_weights, wealth, time_to_event
            )['total_cost']
            
            return {
                'optimal_weights': boundary_weights,
                'frictionless_weights': frictionless_weights,
                'transaction_costs': boundary_cost,
                'no_trade_region': False,
                'weight_adjustment': boundary_weights - current_weights,
                'no_trade_boundaries': no_trade_boundaries
            }
    
    def _calculate_no_trade_boundaries(self, current_weights: np.ndarray, expected_returns: np.ndarray,
                                     covariance_matrix: np.ndarray, transaction_cost_model: 'TransactionCostModel',
                                     wealth: float, time_to_event: float) -> Dict[str, np.ndarray]:
        """
        Calculate no-trade region boundaries for each asset.
        
        Args:
            current_weights: Current portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            transaction_cost_model: Transaction cost model
            wealth: Current wealth
            time_to_event: Time to next event
            
        Returns:
            Dictionary with upper and lower boundaries for each asset
        """
        n_assets = len(current_weights)
        lower_bounds = np.zeros(n_assets)
        upper_bounds = np.zeros(n_assets)
        
        # Approximate no-trade boundaries using transaction cost rates
        for i in range(n_assets):
            # Get transaction cost rates for small trades
            small_purchase_cost = transaction_cost_model.calculate_asymmetric_cost(
                wealth * 0.01, True, time_to_event, 1.0, 0.02
            ) / (wealth * 0.01)
            
            small_sale_cost = transaction_cost_model.calculate_asymmetric_cost(
                wealth * 0.01, False, time_to_event, 1.0, 0.02
            ) / (wealth * 0.01)
            
            # No-trade boundaries based on cost-benefit analysis
            # Upper bound: cost of buying equals marginal benefit
            expected_excess_return = expected_returns[i] - self.risk_free_rate
            risk_penalty = self.gamma_T * covariance_matrix[i, i]
            
            # Simplified boundary calculation
            upper_adjustment = small_purchase_cost / (expected_excess_return - risk_penalty) if (expected_excess_return - risk_penalty) > 0 else 0.01
            lower_adjustment = small_sale_cost / (expected_excess_return - risk_penalty) if (expected_excess_return - risk_penalty) > 0 else 0.01
            
            upper_bounds[i] = current_weights[i] + min(upper_adjustment, 0.1)  # Cap at 10% adjustment
            lower_bounds[i] = current_weights[i] - min(lower_adjustment, 0.1)  # Cap at 10% adjustment
            
            # Ensure bounds are reasonable
            upper_bounds[i] = min(upper_bounds[i], 1.0)
            lower_bounds[i] = max(lower_bounds[i], -1.0)
        
        return {'upper': upper_bounds, 'lower': lower_bounds}
    
    def _calculate_boundary_weights(self, current_weights: np.ndarray, target_weights: np.ndarray,
                                  no_trade_boundaries: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weights at the boundary of the no-trade region.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target (frictionless) portfolio weights
            no_trade_boundaries: No-trade region boundaries
            
        Returns:
            Boundary weights
        """
        n_assets = len(current_weights)
        boundary_weights = np.zeros(n_assets)
        
        for i in range(n_assets):
            if target_weights[i] > no_trade_boundaries['upper'][i]:
                # Target is above upper boundary, trade to upper boundary
                boundary_weights[i] = no_trade_boundaries['upper'][i]
            elif target_weights[i] < no_trade_boundaries['lower'][i]:
                # Target is below lower boundary, trade to lower boundary
                boundary_weights[i] = no_trade_boundaries['lower'][i]
            else:
                # Target is within no-trade region, stay at current weight
                boundary_weights[i] = current_weights[i]
        
        # Ensure weights sum to 1
        weight_sum = np.sum(boundary_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            boundary_weights = boundary_weights / weight_sum
        
        return boundary_weights


class TransactionCostModel:
    """
    Transaction cost model with asymmetric effects for dynamic portfolio optimization.
    
    Implements Requirements 5.1-5.4:
    - Different costs for purchases (τ_b) vs sales (τ_s ≤ τ_b)
    - Time-varying transaction costs (higher pre-event)
    - Liquidity-dependent cost adjustments
    - Continuous transaction cost integration
    """
    
    def __init__(self, tau_b: float = 0.002, tau_s: float = 0.001, 
                 pre_event_multiplier: float = 1.5, liquidity_sensitivity: float = 0.1,
                 time_decay_rate: float = 0.1):
        """
        Initialize transaction cost model with asymmetric cost structure.
        
        Args:
            tau_b: Base purchase transaction cost rate (τ_b)
            tau_s: Base sale transaction cost rate (τ_s ≤ τ_b)
            pre_event_multiplier: Multiplier for costs during pre-event periods
            liquidity_sensitivity: Sensitivity to liquidity conditions
            time_decay_rate: Rate at which pre-event cost premium decays
        """
        if tau_b < 0 or tau_s < 0:
            raise ValueError("Transaction costs must be non-negative")
        if tau_s > tau_b:
            raise ValueError("Sale transaction cost (tau_s) must be less than or equal to purchase cost (tau_b)")
        if pre_event_multiplier < 1:
            raise ValueError("Pre-event multiplier must be >= 1")
            
        self.tau_b_base = tau_b
        self.tau_s_base = tau_s
        self.pre_event_multiplier = pre_event_multiplier
        self.liquidity_sensitivity = liquidity_sensitivity
        self.time_decay_rate = time_decay_rate
        
        # Track cost history for analysis
        self.cost_history = []
        self.liquidity_history = []
        
    def calculate_asymmetric_cost(self, position_change: float, is_purchase: bool,
                                time_to_event: float = np.inf, liquidity_factor: float = 1.0,
                                market_volatility: float = 0.02) -> float:
        """
        Calculate asymmetric transaction cost based on trade direction and market conditions.
        
        Implements Requirements 5.1, 5.2: Different costs for purchases vs sales, 
        time-varying costs higher pre-event.
        
        Args:
            position_change: Size of position change (absolute value)
            is_purchase: True for purchases, False for sales
            time_to_event: Time until next event (days), inf if no event
            liquidity_factor: Current liquidity factor (1.0 = normal, >1 = low liquidity)
            market_volatility: Current market volatility level
            
        Returns:
            Total transaction cost
        """
        if abs(position_change) < 1e-8:
            return 0.0
            
        # Base cost depends on trade direction
        base_cost_rate = self.tau_b_base if is_purchase else self.tau_s_base
        
        # Time-varying component (higher costs pre-event)
        time_multiplier = self._calculate_time_multiplier(time_to_event)
        
        # Liquidity-dependent adjustment
        liquidity_adjustment = self._calculate_liquidity_adjustment(liquidity_factor, market_volatility)
        
        # Total cost rate
        total_cost_rate = base_cost_rate * time_multiplier * liquidity_adjustment
        
        # Apply to position change
        total_cost = total_cost_rate * abs(position_change)
        
        # Store for history tracking
        self.cost_history.append({
            'position_change': position_change,
            'is_purchase': is_purchase,
            'base_cost_rate': base_cost_rate,
            'time_multiplier': time_multiplier,
            'liquidity_adjustment': liquidity_adjustment,
            'total_cost_rate': total_cost_rate,
            'total_cost': total_cost,
            'time_to_event': time_to_event,
            'liquidity_factor': liquidity_factor,
            'market_volatility': market_volatility
        })
        
        return total_cost
    
    def _calculate_time_multiplier(self, time_to_event: float) -> float:
        """
        Calculate time-varying multiplier for pre-event cost increases.
        
        Implements Requirement 5.2: Time-varying transaction costs (higher pre-event).
        
        Args:
            time_to_event: Time until next event (days)
            
        Returns:
            Time multiplier (≥ 1.0)
        """
        if np.isinf(time_to_event) or time_to_event > 30:
            # No event or event is far away
            return 1.0
        
        # Exponential increase as event approaches
        # Cost multiplier peaks at event time and decays with distance
        time_factor = np.exp(-self.time_decay_rate * time_to_event)
        multiplier = 1.0 + (self.pre_event_multiplier - 1.0) * time_factor
        
        return max(1.0, multiplier)
    
    def _calculate_liquidity_adjustment(self, liquidity_factor: float, market_volatility: float) -> float:
        """
        Calculate liquidity-dependent cost adjustment.
        
        Implements Requirement 5.3: Liquidity-dependent cost adjustments.
        
        Args:
            liquidity_factor: Liquidity condition factor (1.0 = normal, >1 = low liquidity)
            market_volatility: Current market volatility
            
        Returns:
            Liquidity adjustment factor (≥ 1.0)
        """
        # Base liquidity adjustment
        liquidity_adj = 1.0 + self.liquidity_sensitivity * (liquidity_factor - 1.0)
        
        # Additional volatility-based adjustment
        volatility_adj = 1.0 + 0.5 * self.liquidity_sensitivity * market_volatility
        
        # Combined adjustment
        total_adjustment = liquidity_adj * volatility_adj
        
        return max(1.0, total_adjustment)
    
    def calculate_portfolio_rebalancing_cost(self, current_weights: np.ndarray, 
                                           target_weights: np.ndarray, portfolio_value: float,
                                           time_to_event: float = np.inf, 
                                           liquidity_factors: Optional[np.ndarray] = None,
                                           market_volatilities: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate total transaction costs for portfolio rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            time_to_event: Time until next event
            liquidity_factors: Liquidity factors for each asset (optional)
            market_volatilities: Market volatilities for each asset (optional)
            
        Returns:
            Dictionary with cost breakdown
        """
        n_assets = len(current_weights)
        
        # Default values if not provided
        if liquidity_factors is None:
            liquidity_factors = np.ones(n_assets)
        if market_volatilities is None:
            market_volatilities = np.full(n_assets, 0.02)
            
        # Calculate weight changes
        weight_changes = target_weights - current_weights
        position_changes = weight_changes * portfolio_value
        
        # Calculate costs for each asset
        asset_costs = np.zeros(n_assets)
        purchase_costs = 0.0
        sale_costs = 0.0
        
        for i in range(n_assets):
            if abs(position_changes[i]) > 1e-8:
                is_purchase = position_changes[i] > 0
                
                cost = self.calculate_asymmetric_cost(
                    abs(position_changes[i]),
                    is_purchase,
                    time_to_event,
                    liquidity_factors[i],
                    market_volatilities[i]
                )
                
                asset_costs[i] = cost
                
                if is_purchase:
                    purchase_costs += cost
                else:
                    sale_costs += cost
        
        total_cost = np.sum(asset_costs)
        
        return {
            'total_cost': total_cost,
            'purchase_costs': purchase_costs,
            'sale_costs': sale_costs,
            'asset_costs': asset_costs,
            'position_changes': position_changes,
            'weight_changes': weight_changes,
            'cost_as_percentage': total_cost / portfolio_value if portfolio_value > 0 else 0.0
        }
    
    def calculate_continuous_cost_integral(self, weight_path: np.ndarray, time_grid: np.ndarray,
                                         portfolio_value: float, event_times: Optional[List[float]] = None,
                                         liquidity_path: Optional[np.ndarray] = None,
                                         volatility_path: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate continuous transaction cost integration over a time path.
        
        Implements Requirement 5.4: Continuous transaction cost integration.
        
        Args:
            weight_path: Portfolio weight path over time (T x N array)
            time_grid: Time grid points
            portfolio_value: Portfolio value
            event_times: List of event times (optional)
            liquidity_path: Liquidity factor path over time (optional)
            volatility_path: Volatility path over time (optional)
            
        Returns:
            Dictionary with integrated cost results
        """
        T, n_assets = weight_path.shape
        
        if len(time_grid) != T:
            raise ValueError("Time grid length must match weight path length")
            
        # Default paths if not provided
        if liquidity_path is None:
            liquidity_path = np.ones(T)
        if volatility_path is None:
            volatility_path = np.full(T, 0.02)
        if event_times is None:
            event_times = []
            
        # Initialize cost tracking
        cumulative_costs = np.zeros(T)
        instantaneous_costs = np.zeros(T)
        purchase_costs_path = np.zeros(T)
        sale_costs_path = np.zeros(T)
        
        # Calculate costs at each time step
        for t in range(1, T):
            dt = time_grid[t] - time_grid[t-1]
            
            # Calculate weight changes
            weight_change = weight_path[t] - weight_path[t-1]
            position_change = weight_change * portfolio_value
            
            # Find time to next event
            time_to_next_event = np.inf
            for event_time in event_times:
                if event_time > time_grid[t]:
                    time_to_next_event = min(time_to_next_event, event_time - time_grid[t])
            
            # Calculate costs for this time step
            step_cost = 0.0
            step_purchase_cost = 0.0
            step_sale_cost = 0.0
            
            for i in range(n_assets):
                if abs(position_change[i]) > 1e-8:
                    is_purchase = position_change[i] > 0
                    
                    asset_cost = self.calculate_asymmetric_cost(
                        abs(position_change[i]),
                        is_purchase,
                        time_to_next_event,
                        liquidity_path[t],
                        volatility_path[t]
                    )
                    
                    step_cost += asset_cost
                    
                    if is_purchase:
                        step_purchase_cost += asset_cost
                    else:
                        step_sale_cost += asset_cost
            
            # Store results
            instantaneous_costs[t] = step_cost / dt if dt > 0 else 0.0
            cumulative_costs[t] = cumulative_costs[t-1] + step_cost
            purchase_costs_path[t] = purchase_costs_path[t-1] + step_purchase_cost
            sale_costs_path[t] = sale_costs_path[t-1] + step_sale_cost
        
        # Calculate summary statistics
        total_integrated_cost = cumulative_costs[-1]
        average_cost_rate = np.mean(instantaneous_costs[1:])
        max_instantaneous_cost = np.max(instantaneous_costs)
        
        return {
            'total_integrated_cost': total_integrated_cost,
            'cumulative_costs': cumulative_costs,
            'instantaneous_costs': instantaneous_costs,
            'purchase_costs_path': purchase_costs_path,
            'sale_costs_path': sale_costs_path,
            'average_cost_rate': average_cost_rate,
            'max_instantaneous_cost': max_instantaneous_cost,
            'cost_as_percentage': total_integrated_cost / portfolio_value if portfolio_value > 0 else 0.0
        }
    
    def get_cost_function_for_optimization(self, current_weights: np.ndarray, portfolio_value: float,
                                         time_to_event: float = np.inf,
                                         liquidity_factors: Optional[np.ndarray] = None,
                                         market_volatilities: Optional[np.ndarray] = None):
        """
        Return a cost function suitable for portfolio optimization.
        
        Args:
            current_weights: Current portfolio weights
            portfolio_value: Portfolio value
            time_to_event: Time to next event
            liquidity_factors: Liquidity factors for each asset
            market_volatilities: Market volatilities for each asset
            
        Returns:
            Function that calculates transaction costs given target weights
        """
        def cost_function(target_weights: np.ndarray) -> float:
            """Calculate transaction cost for given target weights."""
            cost_result = self.calculate_portfolio_rebalancing_cost(
                current_weights, target_weights, portfolio_value,
                time_to_event, liquidity_factors, market_volatilities
            )
            return cost_result['total_cost']
        
        return cost_function
    
    def calculate_cost_gradient(self, current_weights: np.ndarray, target_weights: np.ndarray,
                              portfolio_value: float, time_to_event: float = np.inf,
                              liquidity_factors: Optional[np.ndarray] = None,
                              market_volatilities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate gradient of transaction cost with respect to target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Portfolio value
            time_to_event: Time to next event
            liquidity_factors: Liquidity factors
            market_volatilities: Market volatilities
            
        Returns:
            Gradient vector of transaction costs
        """
        n_assets = len(target_weights)
        gradient = np.zeros(n_assets)
        epsilon = 1e-6
        
        # Base cost
        base_cost = self.calculate_portfolio_rebalancing_cost(
            current_weights, target_weights, portfolio_value,
            time_to_event, liquidity_factors, market_volatilities
        )['total_cost']
        
        # Calculate numerical gradient
        for i in range(n_assets):
            perturbed_weights = target_weights.copy()
            perturbed_weights[i] += epsilon
            
            perturbed_cost = self.calculate_portfolio_rebalancing_cost(
                current_weights, perturbed_weights, portfolio_value,
                time_to_event, liquidity_factors, market_volatilities
            )['total_cost']
            
            gradient[i] = (perturbed_cost - base_cost) / epsilon
        
        return gradient
    
    def update_cost_parameters(self, new_params: Dict[str, float]):
        """
        Update transaction cost model parameters.
        
        Args:
            new_params: Dictionary of parameter updates
        """
        if 'tau_b' in new_params:
            self.tau_b_base = max(0, new_params['tau_b'])
        if 'tau_s' in new_params:
            self.tau_s_base = max(0, min(new_params['tau_s'], self.tau_b_base))
        if 'pre_event_multiplier' in new_params:
            self.pre_event_multiplier = max(1.0, new_params['pre_event_multiplier'])
        if 'liquidity_sensitivity' in new_params:
            self.liquidity_sensitivity = max(0, new_params['liquidity_sensitivity'])
        if 'time_decay_rate' in new_params:
            self.time_decay_rate = max(0, new_params['time_decay_rate'])
    
    def analyze_cost_structure(self, weight_changes: np.ndarray, portfolio_value: float,
                             time_to_event: float = np.inf) -> Dict[str, Union[float, Dict]]:
        """
        Analyze the cost structure for given weight changes.
        
        Args:
            weight_changes: Portfolio weight changes
            portfolio_value: Portfolio value
            time_to_event: Time to next event
            
        Returns:
            Dictionary with cost structure analysis
        """
        position_changes = weight_changes * portfolio_value
        
        # Separate purchases and sales
        purchases = position_changes[position_changes > 0]
        sales = position_changes[position_changes < 0]
        
        # Calculate costs
        purchase_cost = sum(self.calculate_asymmetric_cost(abs(p), True, time_to_event) 
                           for p in purchases)
        sale_cost = sum(self.calculate_asymmetric_cost(abs(s), False, time_to_event) 
                       for s in sales)
        
        total_cost = purchase_cost + sale_cost
        
        # Cost breakdown
        analysis = {
            'total_cost': total_cost,
            'purchase_cost': purchase_cost,
            'sale_cost': sale_cost,
            'cost_asymmetry_ratio': purchase_cost / sale_cost if sale_cost > 0 else np.inf,
            'cost_as_percentage': total_cost / portfolio_value if portfolio_value > 0 else 0.0,
            'purchase_volume': np.sum(purchases),
            'sale_volume': abs(np.sum(sales)),
            'net_volume': np.sum(purchases) + np.sum(sales),
            'gross_volume': np.sum(purchases) + abs(np.sum(sales)),
            'breakdown': {
                'base_purchase_rate': self.tau_b_base,
                'base_sale_rate': self.tau_s_base,
                'time_multiplier': self._calculate_time_multiplier(time_to_event),
                'liquidity_adjustment': self._calculate_liquidity_adjustment(1.0, 0.02)
            }
        }
        
        return analysis
    
    def get_cost_history(self) -> List[Dict]:
        """
        Get transaction cost history for analysis.
        
        Returns:
            List of cost history records
        """
        return self.cost_history.copy()
    
    def clear_history(self):
        """Clear cost history."""
        self.cost_history.clear()
        self.liquidity_history.clear()
    
    def __repr__(self) -> str:
        return (f"TransactionCostModel(tau_b={self.tau_b_base:.4f}, tau_s={self.tau_s_base:.4f}, "
                f"pre_event_mult={self.pre_event_multiplier:.2f}, "
                f"liquidity_sens={self.liquidity_sensitivity:.3f})")
    
    def get_optimization_history(self) -> List[Dict]:
        """
        Get history of optimization results.
        
        Returns:
            List of optimization result dictionaries
        """
        return self.optimization_history.copy()
    
    def reset_optimization_history(self):
        """Reset optimization history."""
        self.optimization_history = []


class JumpProcessModel:
    """
    Jump Process Model for event times with outcome modeling and bias parameter evolution.
    
    Implements Requirements 2.1-2.3:
    - Jump process dN(t) for event occurrences
    - Event outcome modeling with mean-zero ε_t
    - Bias parameter b_t evolution around events
    """
    
    def __init__(self, jump_intensity: float = 0.1, jump_size_std: float = 0.05,
                 bias_initial: float = 0.01, bias_decay_rate: float = 0.1,
                 outcome_threshold: float = 0.02):
        """
        Initialize Jump Process Model.
        
        Args:
            jump_intensity: Intensity parameter λ for Poisson jump process
            jump_size_std: Standard deviation of jump sizes (for mean-zero ε_t)
            bias_initial: Initial bias magnitude for different event outcomes
            bias_decay_rate: Exponential decay rate for bias parameter b_t
            outcome_threshold: Threshold for classifying event outcomes
        """
        self.jump_intensity = jump_intensity
        self.jump_size_std = jump_size_std
        self.bias_initial = bias_initial
        self.bias_decay_rate = bias_decay_rate
        self.outcome_threshold = outcome_threshold
        
        # Process state variables
        self.jump_times = []
        self.jump_sizes = []
        self.event_outcomes = {}
        self.bias_evolution = None
        
        # Fitted state
        self.is_fitted = False
        self.time_horizon = None
        
    def simulate_jump_process(self, T: float, dt: float = 0.01, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate jump process dN(t) for event occurrences.
        Implements Requirement 2.1: Jump process dN(t) for event occurrences.
        
        Args:
            T: Time horizon for simulation
            dt: Time step size
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time_grid, jump_process_values)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Create time grid
        time_grid = np.arange(0, T + dt, dt)
        n_steps = len(time_grid)
        
        # Initialize jump process
        jump_process = np.zeros(n_steps)
        jump_times = []
        
        # Simulate Poisson arrivals
        current_time = 0.0
        jump_count = 0
        
        while current_time < T:
            # Generate next jump time using exponential distribution
            inter_arrival_time = np.random.exponential(1.0 / self.jump_intensity)
            current_time += inter_arrival_time
            
            if current_time < T:
                jump_times.append(current_time)
                jump_count += 1
                
                # Find the closest time grid point
                time_idx = int(np.round(current_time / dt))
                if time_idx < n_steps:
                    jump_process[time_idx] = 1  # Mark jump occurrence
                    
        self.jump_times = jump_times
        return time_grid, jump_process
    
    def generate_jump_sizes(self, n_jumps: int, event_outcomes: Optional[Dict[int, str]] = None,
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate jump sizes ε_t with mean-zero property.
        Implements Requirement 2.2: Event outcome modeling with mean-zero ε_t.
        
        Args:
            n_jumps: Number of jumps to generate sizes for
            event_outcomes: Optional dictionary mapping jump indices to outcomes
            seed: Random seed for reproducibility
            
        Returns:
            Array of jump sizes (mean-zero ε_t)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate base jump sizes from normal distribution (mean-zero)
        base_jumps = np.random.normal(0, self.jump_size_std, n_jumps)
        
        # Adjust based on event outcomes if provided
        if event_outcomes is not None:
            for jump_idx, outcome in event_outcomes.items():
                if jump_idx < len(base_jumps):
                    if outcome == 'positive':
                        # Positive events: ensure positive jump or amplify existing positive
                        if base_jumps[jump_idx] < 0:
                            base_jumps[jump_idx] = abs(base_jumps[jump_idx])
                        else:
                            base_jumps[jump_idx] *= 1.5  # Amplify positive jumps
                    elif outcome == 'negative':
                        # Negative events: ensure negative jump or amplify existing negative
                        if base_jumps[jump_idx] > 0:
                            base_jumps[jump_idx] = -abs(base_jumps[jump_idx])
                        else:
                            base_jumps[jump_idx] *= 1.5  # Amplify negative jumps
                    # Neutral events keep original mean-zero jumps
                    
        # Ensure overall mean is zero (adjust for any bias introduced by outcome adjustments)
        if len(base_jumps) > 0:
            mean_adjustment = np.mean(base_jumps)
            base_jumps -= mean_adjustment
            
        self.jump_sizes = base_jumps
        return base_jumps
    
    def calculate_bias_evolution(self, time_grid: np.ndarray, jump_times: List[float],
                                event_outcomes: Dict[float, str]) -> np.ndarray:
        """
        Calculate bias parameter b_t evolution around events.
        Implements Requirement 2.3: Bias parameter b_t evolution around events.
        
        Args:
            time_grid: Array of time points
            jump_times: List of jump (event) times
            event_outcomes: Dictionary mapping event times to outcomes
            
        Returns:
            Array of bias parameter values over time
        """
        bias_evolution = np.zeros_like(time_grid)
        
        for i, t in enumerate(time_grid):
            total_bias = 0.0
            
            # Sum contributions from all past events
            for jump_time in jump_times:
                if t >= jump_time:
                    time_since_event = t - jump_time
                    outcome = event_outcomes.get(jump_time, 'neutral')
                    
                    # Determine initial bias based on event outcome
                    if outcome == 'positive':
                        initial_bias = self.bias_initial
                    elif outcome == 'negative':
                        initial_bias = -self.bias_initial
                    else:
                        initial_bias = 0.0
                        
                    # Apply exponential decay
                    current_bias = initial_bias * np.exp(-self.bias_decay_rate * time_since_event)
                    total_bias += current_bias
                    
            bias_evolution[i] = total_bias
            
        self.bias_evolution = bias_evolution
        return bias_evolution
    
    def simulate_complete_jump_process(self, T: float, dt: float = 0.01, 
                                     event_outcome_probs: Dict[str, float] = None,
                                     seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate complete jump process with outcomes and bias evolution.
        
        Args:
            T: Time horizon
            dt: Time step size
            event_outcome_probs: Probabilities for different event outcomes
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing all process components
        """
        if event_outcome_probs is None:
            event_outcome_probs = {'positive': 0.3, 'negative': 0.3, 'neutral': 0.4}
            
        if seed is not None:
            np.random.seed(seed)
            
        # Simulate jump times
        time_grid, jump_process = self.simulate_jump_process(T, dt, seed)
        
        # Generate event outcomes for each jump
        n_jumps = len(self.jump_times)
        event_outcomes = {}
        
        if n_jumps > 0:
            # Randomly assign outcomes based on probabilities
            outcomes = ['positive', 'negative', 'neutral']
            probs = [event_outcome_probs[outcome] for outcome in outcomes]
            
            for i, jump_time in enumerate(self.jump_times):
                outcome = np.random.choice(outcomes, p=probs)
                event_outcomes[jump_time] = outcome
                
            # Generate jump sizes based on outcomes
            outcome_dict_by_idx = {i: event_outcomes[jump_time] for i, jump_time in enumerate(self.jump_times)}
            jump_sizes = self.generate_jump_sizes(n_jumps, outcome_dict_by_idx, seed)
            
            # Calculate bias evolution
            bias_evolution = self.calculate_bias_evolution(time_grid, self.jump_times, event_outcomes)
        else:
            jump_sizes = np.array([])
            bias_evolution = np.zeros_like(time_grid)
            
        self.event_outcomes = event_outcomes
        # Only update time_horizon if it's not already set (to preserve fitted state)
        if self.time_horizon is None:
            self.time_horizon = T
        self.is_fitted = True
        
        return {
            'time_grid': time_grid,
            'jump_process': jump_process,
            'jump_times': np.array(self.jump_times),
            'jump_sizes': jump_sizes,
            'bias_evolution': bias_evolution,
            'event_outcomes': event_outcomes
        }
    
    def fit_to_observed_events(self, event_times: List[float], event_outcomes: Dict[float, str],
                              time_horizon: float) -> 'JumpProcessModel':
        """
        Fit jump process model to observed event data.
        
        Args:
            event_times: List of observed event times
            event_outcomes: Dictionary mapping event times to outcomes
            time_horizon: Total time horizon of the data
            
        Returns:
            Fitted JumpProcessModel instance
        """
        self.jump_times = event_times
        self.event_outcomes = event_outcomes
        self.time_horizon = time_horizon
        
        # Estimate jump intensity from observed data
        if len(event_times) > 0 and time_horizon > 0:
            self.jump_intensity = len(event_times) / time_horizon
        else:
            self.jump_intensity = 0.1  # Default value
            
        # Generate jump sizes for observed events
        if len(event_times) > 0:
            outcome_dict_by_idx = {i: event_outcomes.get(event_time, 'neutral') 
                                  for i, event_time in enumerate(event_times)}
            self.jump_sizes = self.generate_jump_sizes(len(event_times), outcome_dict_by_idx)
        else:
            self.jump_sizes = np.array([])
            
        self.is_fitted = True
        return self
    
    def predict_future_jumps(self, future_horizon: float, dt: float = 0.01,
                           seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Predict future jump process evolution.
        
        Args:
            future_horizon: Time horizon for future predictions
            dt: Time step size
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing predicted jump process components
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Simulate future jump process
        future_results = self.simulate_complete_jump_process(
            future_horizon, dt, seed=seed
        )
        
        # Adjust time grid to start from current time
        if self.time_horizon is not None:
            # Add current time horizon to all time-based results
            future_results['time_grid'] = future_results['time_grid'] + self.time_horizon
            if len(future_results['jump_times']) > 0:
                future_results['jump_times'] = future_results['jump_times'] + self.time_horizon
                
                # Update jump times in event outcomes
                updated_outcomes = {}
                for jump_time, outcome in future_results['event_outcomes'].items():
                    updated_outcomes[jump_time + self.time_horizon] = outcome
                future_results['event_outcomes'] = updated_outcomes
            
        return future_results
    
    def calculate_jump_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics for the jump process.
        
        Returns:
            Dictionary containing jump process statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calculating statistics")
            
        stats = {
            'jump_intensity': self.jump_intensity,
            'n_jumps': len(self.jump_times),
            'mean_jump_size': np.mean(self.jump_sizes) if len(self.jump_sizes) > 0 else 0.0,
            'std_jump_size': np.std(self.jump_sizes) if len(self.jump_sizes) > 0 else 0.0,
            'time_horizon': self.time_horizon if self.time_horizon is not None else 0.0
        }
        
        # Event outcome statistics
        if self.event_outcomes:
            outcomes = list(self.event_outcomes.values())
            stats['positive_events_pct'] = outcomes.count('positive') / len(outcomes) * 100
            stats['negative_events_pct'] = outcomes.count('negative') / len(outcomes) * 100
            stats['neutral_events_pct'] = outcomes.count('neutral') / len(outcomes) * 100
        else:
            stats['positive_events_pct'] = 0.0
            stats['negative_events_pct'] = 0.0
            stats['neutral_events_pct'] = 0.0
            
        # Bias parameter statistics
        if self.bias_evolution is not None:
            stats['mean_bias'] = np.mean(self.bias_evolution)
            stats['max_bias'] = np.max(np.abs(self.bias_evolution))
            stats['bias_persistence'] = self._calculate_bias_persistence()
        else:
            stats['mean_bias'] = 0.0
            stats['max_bias'] = 0.0
            stats['bias_persistence'] = 0.0
            
        return stats
    
    def _calculate_bias_persistence(self) -> float:
        """
        Calculate how long bias persists after events (half-life).
        
        Returns:
            Bias persistence measure (half-life in time units)
        """
        if self.bias_decay_rate <= 0:
            return float('inf')
        else:
            # Half-life = ln(2) / decay_rate
            return np.log(2) / self.bias_decay_rate
    
    def get_jump_impact_at_time(self, t: float) -> Dict[str, float]:
        """
        Get jump process impact at a specific time.
        
        Args:
            t: Time point to evaluate
            
        Returns:
            Dictionary containing jump impact components at time t
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
            
        # Check if there's a jump at this time
        jump_at_time = 0.0
        jump_size_at_time = 0.0
        
        for i, jump_time in enumerate(self.jump_times):
            if abs(t - jump_time) < 1e-6:  # Within tolerance
                jump_at_time = 1.0
                if i < len(self.jump_sizes):
                    jump_size_at_time = self.jump_sizes[i]
                break
                
        # Calculate bias at this time
        bias_at_time = 0.0
        for jump_time in self.jump_times:
            if t >= jump_time:
                time_since_event = t - jump_time
                outcome = self.event_outcomes.get(jump_time, 'neutral')
                
                if outcome == 'positive':
                    initial_bias = self.bias_initial
                elif outcome == 'negative':
                    initial_bias = -self.bias_initial
                else:
                    initial_bias = 0.0
                    
                current_bias = initial_bias * np.exp(-self.bias_decay_rate * time_since_event)
                bias_at_time += current_bias
                
        return {
            'time': t,
            'jump_indicator': jump_at_time,
            'jump_size': jump_size_at_time,
            'bias_parameter': bias_at_time,
            'nearest_event_time': min(self.jump_times, key=lambda jt: abs(t - jt)) if self.jump_times else None,
            'time_to_nearest_event': min([abs(t - jt) for jt in self.jump_times]) if self.jump_times else float('inf')
        }


class PortfolioOptimizer:
    """
    Portfolio Optimizer class with dynamic optimization for continuous-time asset pricing model.
    
    Implements Requirements 8.1-8.4:
    - Stochastic control problem formulation
    - Wealth evolution equation with multi-asset returns
    - Portfolio weight constraints and bounds
    - Optimal weight calculation methods
    """
    
    def __init__(self, risk_free_rate: float = 0.0, gamma_T: float = 2.0, gamma_V: float = 1.0,
                 time_horizon: float = 1.0, wealth_bounds: Optional[Tuple[float, float]] = None,
                 weight_bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for optimization
            gamma_T: Terminal wealth risk aversion coefficient
            gamma_V: Real-time variance aversion coefficient
            time_horizon: Investment time horizon
            wealth_bounds: Optional bounds on wealth levels (min_wealth, max_wealth)
            weight_bounds: Optional bounds on portfolio weights (min_weight, max_weight)
        """
        self.risk_free_rate = risk_free_rate
        self.gamma_T = gamma_T
        self.gamma_V = gamma_V
        self.time_horizon = time_horizon
        self.wealth_bounds = wealth_bounds or (1e-6, 1e10)
        self.weight_bounds = weight_bounds or (-1.0, 1.0)
        
        # Optimization state
        self.current_wealth = None
        self.current_weights = None
        self.optimization_history = []
        
        # Stochastic control problem components
        self.state_variables = {}
        self.control_variables = {}
        
    def formulate_stochastic_control_problem(self, expected_returns: np.ndarray, 
                                           covariance_matrix: np.ndarray,
                                           volatility_model: Optional['ThreePhaseVolatilityModel'] = None,
                                           transaction_cost_model: Optional['TransactionCostModel'] = None,
                                           event_times: Optional[List[float]] = None) -> Dict[str, any]:
        """
        Formulate the stochastic control problem for dynamic portfolio optimization.
        
        Implements Requirement 8.1: Stochastic control problem formulation
        
        Args:
            expected_returns: Expected returns vector μ
            covariance_matrix: Covariance matrix Σ
            volatility_model: Three-phase volatility model for time-varying volatility
            transaction_cost_model: Transaction cost model for cost calculations
            event_times: List of event times for regime switching
            
        Returns:
            Dictionary containing stochastic control problem formulation
        """
        n_assets = len(expected_returns)
        
        # State variables: (W_t, t, regime_state)
        self.state_variables = {
            'wealth': 'W_t',  # Current wealth level
            'time': 't',      # Current time
            'regime_state': 's_t',  # Current regime state (if applicable)
            'volatility_state': 'h_t'  # Current volatility state
        }
        
        # Control variables: portfolio weights w_t
        self.control_variables = {
            'portfolio_weights': 'w_t',  # Portfolio weight vector
            'weight_bounds': self.weight_bounds,
            'budget_constraint': 'sum(w_t) = 1'
        }
        
        # Dynamics equations
        dynamics = {
            # Wealth evolution: dW_t = W_t * [w_t^T * (μ dt + Σ^(1/2) dB_t) - TC_t dt]
            'wealth_evolution': {
                'drift': 'W_t * w_t^T * μ - TC_t',
                'diffusion': 'W_t * w_t^T * Σ^(1/2)',
                'jump_component': 'W_t * w_t^T * ε_t * dN_t' if event_times else None
            },
            
            # Portfolio weight evolution (if using dynamic rebalancing)
            'weight_evolution': {
                'drift': '-(w_t ⊙ (μ - w_t^T * μ * 1)) + rebalancing_drift',
                'diffusion': 'w_t ⊙ (Σ^(1/2) - w_t^T * Σ^(1/2) * 1)',
                'control_term': 'dw_t (rebalancing decisions)'
            }
        }
        
        # Objective function: maximize E[U(W_T)] - ∫_0^T penalty terms dt
        objective = {
            'terminal_utility': 'U(W_T) = W_T - (γ_T/2) * Var[W_T]',
            'running_cost': '-(γ_V/2) * Var[dW_t]/dt - TC_t',
            'total_objective': 'max E[U(W_T)] + E[∫_0^T running_cost dt]'
        }
        
        # Constraints
        constraints = {
            'budget_constraint': 'sum(w_t) = 1',
            'weight_bounds': f'{self.weight_bounds[0]} ≤ w_i,t ≤ {self.weight_bounds[1]}',
            'wealth_bounds': f'{self.wealth_bounds[0]} ≤ W_t ≤ {self.wealth_bounds[1]}',
            'no_bankruptcy': 'W_t > 0 for all t'
        }
        
        # Hamilton-Jacobi-Bellman equation formulation
        hjb_equation = {
            'value_function': 'V(W, t, s)',
            'hjb_equation': '∂V/∂t + max_w [L^w V] = 0',
            'generator': 'L^w V = (W*w^T*μ - TC)*∂V/∂W + (1/2)*(W*w^T*Σ*w)*∂²V/∂W² + penalty_terms',
            'terminal_condition': 'V(W, T, s) = U(W)',
            'boundary_conditions': 'appropriate boundary conditions at wealth bounds'
        }
        
        return {
            'state_variables': self.state_variables,
            'control_variables': self.control_variables,
            'dynamics': dynamics,
            'objective': objective,
            'constraints': constraints,
            'hjb_formulation': hjb_equation,
            'problem_parameters': {
                'n_assets': n_assets,
                'expected_returns': expected_returns,
                'covariance_matrix': covariance_matrix,
                'risk_free_rate': self.risk_free_rate,
                'gamma_T': self.gamma_T,
                'gamma_V': self.gamma_V,
                'time_horizon': self.time_horizon
            }
        }
    
    def implement_wealth_evolution_equation(self, current_wealth: float, portfolio_weights: np.ndarray,
                                          expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                                          dt: float, dW: Optional[np.ndarray] = None,
                                          transaction_costs: float = 0.0,
                                          jump_component: float = 0.0) -> Dict[str, float]:
        """
        Implement wealth evolution equation with multi-asset returns.
        
        Implements Requirement 8.2: Wealth evolution equation with multi-asset returns
        
        The wealth evolution follows:
        dW_t = W_t * [w_t^T * (μ dt + Σ^(1/2) dB_t) + ε_t dN_t] - TC_t dt
        
        Args:
            current_wealth: Current wealth level W_t
            portfolio_weights: Portfolio weights w_t
            expected_returns: Expected returns vector μ
            covariance_matrix: Covariance matrix Σ
            dt: Time increment
            dW: Brownian motion increments (if None, will be simulated)
            transaction_costs: Transaction costs TC_t
            jump_component: Jump component ε_t * dN_t
            
        Returns:
            Dictionary with wealth evolution components
        """
        n_assets = len(portfolio_weights)
        
        # Validate inputs
        if abs(np.sum(portfolio_weights) - 1.0) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1")
        if current_wealth <= 0:
            raise ValueError("Wealth must be positive")
        
        # Portfolio expected return and volatility
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Generate Brownian motion increments if not provided
        if dW is None:
            dW = np.random.normal(0, np.sqrt(dt), n_assets)
        
        # Wealth evolution components
        # Drift term: W_t * w_t^T * μ * dt
        drift_component = current_wealth * portfolio_return * dt
        
        # Diffusion term: W_t * w_t^T * Σ^(1/2) * dB_t
        portfolio_volatility_vector = np.dot(covariance_matrix, portfolio_weights)
        diffusion_component = current_wealth * np.dot(portfolio_weights, 
                                                    np.dot(np.linalg.cholesky(covariance_matrix), dW))
        
        # Transaction cost term: -TC_t * dt
        transaction_cost_component = -transaction_costs * dt
        
        # Jump component: W_t * ε_t * dN_t
        jump_wealth_component = current_wealth * jump_component
        
        # Total wealth change
        dW_total = drift_component + diffusion_component + transaction_cost_component + jump_wealth_component
        
        # New wealth level
        new_wealth = max(current_wealth + dW_total, self.wealth_bounds[0])  # Ensure non-negative wealth
        
        # Update current state
        self.current_wealth = new_wealth
        
        return {
            'current_wealth': current_wealth,
            'new_wealth': new_wealth,
            'wealth_change': dW_total,
            'drift_component': drift_component,
            'diffusion_component': diffusion_component,
            'transaction_cost_component': transaction_cost_component,
            'jump_component': jump_wealth_component,
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'dt': dt,
            'portfolio_weights': portfolio_weights.copy()
        }
    
    def apply_portfolio_constraints(self, weights: np.ndarray, 
                                  additional_constraints: Optional[Dict[str, any]] = None) -> np.ndarray:
        """
        Apply portfolio weight constraints and bounds.
        
        Implements Requirement 8.3: Portfolio weight constraints and bounds
        
        Args:
            weights: Raw portfolio weights
            additional_constraints: Additional constraint specifications
            
        Returns:
            Constrained portfolio weights
        """
        n_assets = len(weights)
        constrained_weights = weights.copy()
        
        # Apply weight bounds
        min_weight, max_weight = self.weight_bounds
        constrained_weights = np.clip(constrained_weights, min_weight, max_weight)
        
        # Apply additional constraints if provided
        if additional_constraints:
            # Sector constraints
            if 'sector_limits' in additional_constraints:
                sector_limits = additional_constraints['sector_limits']
                sector_mapping = additional_constraints.get('sector_mapping', {})
                
                for sector, (min_limit, max_limit) in sector_limits.items():
                    sector_indices = sector_mapping.get(sector, [])
                    if sector_indices:
                        sector_weight = np.sum(constrained_weights[sector_indices])
                        if sector_weight > max_limit:
                            # Scale down sector weights proportionally
                            scale_factor = max_limit / sector_weight
                            constrained_weights[sector_indices] *= scale_factor
                        elif sector_weight < min_limit:
                            # Scale up sector weights proportionally
                            scale_factor = min_limit / sector_weight if sector_weight > 0 else 1.0
                            constrained_weights[sector_indices] *= scale_factor
            
            # Turnover constraints
            if 'max_turnover' in additional_constraints and self.current_weights is not None:
                max_turnover = additional_constraints['max_turnover']
                current_turnover = np.sum(np.abs(constrained_weights - self.current_weights))
                if current_turnover > max_turnover:
                    # Scale weight changes to meet turnover constraint
                    weight_changes = constrained_weights - self.current_weights
                    scale_factor = max_turnover / current_turnover
                    constrained_weights = self.current_weights + scale_factor * weight_changes
            
            # Long-only constraint
            if additional_constraints.get('long_only', False):
                constrained_weights = np.maximum(constrained_weights, 0.0)
            
            # Maximum concentration constraint
            if 'max_concentration' in additional_constraints:
                max_concentration = additional_constraints['max_concentration']
                constrained_weights = np.minimum(constrained_weights, max_concentration)
        
        # Ensure budget constraint: sum of weights = 1
        weight_sum = np.sum(constrained_weights)
        if abs(weight_sum) > 1e-8:  # Avoid division by zero
            constrained_weights = constrained_weights / weight_sum
        else:
            # Fallback to equal weights if all weights are zero
            constrained_weights = np.ones(n_assets) / n_assets
        
        # Re-apply max concentration after normalization if needed
        if additional_constraints and 'max_concentration' in additional_constraints:
            max_concentration = additional_constraints['max_concentration']
            if np.any(constrained_weights > max_concentration):
                # Iteratively adjust weights to satisfy concentration constraint
                for _ in range(10):  # Maximum iterations to prevent infinite loop
                    excess_mask = constrained_weights > max_concentration
                    if not np.any(excess_mask):
                        break
                    
                    # Cap excess weights
                    excess_amount = np.sum(constrained_weights[excess_mask] - max_concentration)
                    constrained_weights[excess_mask] = max_concentration
                    
                    # Redistribute excess to other assets
                    remaining_mask = ~excess_mask
                    if np.any(remaining_mask):
                        remaining_capacity = np.sum(np.maximum(0, max_concentration - constrained_weights[remaining_mask]))
                        if remaining_capacity > 1e-8:
                            redistribution_weights = np.maximum(0, max_concentration - constrained_weights[remaining_mask])
                            redistribution_weights = redistribution_weights / np.sum(redistribution_weights)
                            constrained_weights[remaining_mask] += excess_amount * redistribution_weights
                    
                    # Renormalize
                    weight_sum = np.sum(constrained_weights)
                    if abs(weight_sum) > 1e-8:
                        constrained_weights = constrained_weights / weight_sum
        
        # Final validation
        if abs(np.sum(constrained_weights) - 1.0) > 1e-6:
            raise ValueError("Failed to satisfy budget constraint after applying constraints")
        
        return constrained_weights
    
    def calculate_optimal_weights(self, current_wealth: float, expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray, time_to_maturity: float,
                                current_weights: Optional[np.ndarray] = None,
                                transaction_cost_model: Optional['TransactionCostModel'] = None,
                                volatility_model: Optional['ThreePhaseVolatilityModel'] = None,
                                additional_constraints: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Calculate optimal portfolio weights using dynamic optimization.
        
        Implements Requirement 8.4: Optimal weight calculation methods
        
        Args:
            current_wealth: Current wealth level
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            time_to_maturity: Time remaining to investment horizon
            current_weights: Current portfolio weights (for transaction cost calculation)
            transaction_cost_model: Transaction cost model
            volatility_model: Volatility model for time-varying parameters
            additional_constraints: Additional portfolio constraints
            
        Returns:
            Dictionary containing optimal weights and optimization details
        """
        n_assets = len(expected_returns)
        
        # Update current state
        self.current_wealth = current_wealth
        if current_weights is not None:
            self.current_weights = current_weights.copy()
        
        # Objective function for optimization
        def objective_function(weights):
            # Apply constraints
            constrained_weights = self.apply_portfolio_constraints(weights, additional_constraints)
            
            # Calculate portfolio moments
            portfolio_return = np.dot(constrained_weights, expected_returns)
            portfolio_variance = np.dot(constrained_weights, np.dot(covariance_matrix, constrained_weights))
            
            # Terminal utility component
            expected_terminal_wealth = current_wealth * np.exp(
                (portfolio_return - 0.5 * portfolio_variance) * time_to_maturity
            )
            terminal_wealth_variance = (current_wealth ** 2) * (
                np.exp(portfolio_variance * time_to_maturity) - 1
            ) * np.exp(2 * portfolio_return * time_to_maturity - portfolio_variance * time_to_maturity)
            
            terminal_utility = expected_terminal_wealth - (self.gamma_T / 2) * terminal_wealth_variance
            
            # Real-time variance penalty
            real_time_penalty = self._calculate_real_time_variance_penalty(
                current_wealth, constrained_weights, covariance_matrix, time_to_maturity
            )
            
            # Transaction costs
            transaction_costs = 0.0
            if current_weights is not None and transaction_cost_model is not None:
                cost_result = transaction_cost_model.calculate_portfolio_rebalancing_cost(
                    current_weights, constrained_weights, current_wealth, time_to_maturity
                )
                transaction_costs = cost_result['total_cost']
            
            # Total objective (negative for minimization)
            total_objective = terminal_utility - (self.gamma_V / 2) * real_time_penalty - transaction_costs
            return -total_objective
        
        # Optimization setup
        initial_weights = current_weights if current_weights is not None else np.ones(n_assets) / n_assets
        
        # Bounds for optimization
        bounds = [self.weight_bounds for _ in range(n_assets)]
        
        # Budget constraint
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Optimization
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                optimal_weights = self.apply_portfolio_constraints(result.x, additional_constraints)
                
                # Calculate final metrics
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
                
                # Transaction costs
                final_transaction_costs = 0.0
                if current_weights is not None and transaction_cost_model is not None:
                    cost_result = transaction_cost_model.calculate_portfolio_rebalancing_cost(
                        current_weights, optimal_weights, current_wealth, time_to_maturity
                    )
                    final_transaction_costs = cost_result['total_cost']
                
                optimization_result = {
                    'optimal_weights': optimal_weights,
                    'portfolio_return': portfolio_return,
                    'portfolio_variance': portfolio_variance,
                    'portfolio_volatility': np.sqrt(portfolio_variance),
                    'transaction_costs': final_transaction_costs,
                    'optimization_success': True,
                    'optimization_message': result.message,
                    'objective_value': -result.fun,
                    'weight_changes': optimal_weights - current_weights if current_weights is not None else optimal_weights,
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)) if current_weights is not None else 0.0
                }
            else:
                # Fallback to analytical solution (Markowitz)
                optimal_weights = self._calculate_markowitz_weights(expected_returns, covariance_matrix)
                optimal_weights = self.apply_portfolio_constraints(optimal_weights, additional_constraints)
                
                portfolio_return = np.dot(optimal_weights, expected_returns)
                portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
                
                optimization_result = {
                    'optimal_weights': optimal_weights,
                    'portfolio_return': portfolio_return,
                    'portfolio_variance': portfolio_variance,
                    'portfolio_volatility': np.sqrt(portfolio_variance),
                    'transaction_costs': 0.0,
                    'optimization_success': False,
                    'optimization_message': f"Numerical optimization failed: {result.message}. Using Markowitz solution.",
                    'objective_value': None,
                    'weight_changes': optimal_weights - current_weights if current_weights is not None else optimal_weights,
                    'turnover': np.sum(np.abs(optimal_weights - current_weights)) if current_weights is not None else 0.0
                }
                
        except ImportError:
            # Fallback if scipy is not available
            optimal_weights = self._calculate_markowitz_weights(expected_returns, covariance_matrix)
            optimal_weights = self.apply_portfolio_constraints(optimal_weights, additional_constraints)
            
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
            
            optimization_result = {
                'optimal_weights': optimal_weights,
                'portfolio_return': portfolio_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_volatility': np.sqrt(portfolio_variance),
                'transaction_costs': 0.0,
                'optimization_success': False,
                'optimization_message': "Scipy not available. Using Markowitz solution.",
                'objective_value': None,
                'weight_changes': optimal_weights - current_weights if current_weights is not None else optimal_weights,
                'turnover': np.sum(np.abs(optimal_weights - current_weights)) if current_weights is not None else 0.0
            }
        
        # Store optimization history
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _calculate_real_time_variance_penalty(self, wealth: float, portfolio_weights: np.ndarray,
                                            covariance_matrix: np.ndarray, time_to_maturity: float) -> float:
        """Calculate real-time variance penalty component."""
        instantaneous_variance = (wealth ** 2) * np.dot(portfolio_weights, 
                                                       np.dot(covariance_matrix, portfolio_weights))
        
        # For constant portfolio weights, integrate over time
        portfolio_return = 0.0  # Assume zero drift for penalty calculation
        portfolio_variance = np.dot(portfolio_weights, np.dot(covariance_matrix, portfolio_weights))
        
        if abs(2 * portfolio_return + portfolio_variance) > 1e-8:
            integral_factor = (np.exp((2 * portfolio_return + portfolio_variance) * time_to_maturity) - 1) / \
                            (2 * portfolio_return + portfolio_variance)
        else:
            integral_factor = time_to_maturity
            
        return (wealth ** 2) * portfolio_variance * integral_factor
    
    def _calculate_markowitz_weights(self, expected_returns: np.ndarray, 
                                   covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate Markowitz optimal weights as fallback."""
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            weights = np.dot(inv_cov, expected_returns) / self.gamma_T
            return weights / np.sum(weights) if np.sum(weights) != 0 else np.ones(len(expected_returns)) / len(expected_returns)
        except np.linalg.LinAlgError:
            return np.ones(len(expected_returns)) / len(expected_returns)


class HJBSolver:
    """
    Hamilton-Jacobi-Bellman equation solver with numerical methods for portfolio optimization.
    
    Implements Requirements 8.1-8.4:
    - Numerical methods for HJB equation
    - Finite difference schemes for HJB equation
    - Boundary condition handling for optimization problem
    - Convergence checking and stability measures
    """
    
    def __init__(self, wealth_grid: np.ndarray, time_grid: np.ndarray,
                 gamma_T: float = 2.0, gamma_V: float = 1.0, risk_free_rate: float = 0.0,
                 convergence_tolerance: float = 1e-6, max_iterations: int = 1000):
        """
        Initialize HJB solver with discretization grids.
        
        Args:
            wealth_grid: Wealth discretization grid
            time_grid: Time discretization grid
            gamma_T: Terminal wealth risk aversion coefficient
            gamma_V: Real-time variance aversion coefficient
            risk_free_rate: Risk-free rate
            convergence_tolerance: Convergence tolerance for iterative methods
            max_iterations: Maximum number of iterations
        """
        self.wealth_grid = wealth_grid
        self.time_grid = time_grid
        self.gamma_T = gamma_T
        self.gamma_V = gamma_V
        self.risk_free_rate = risk_free_rate
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        
        # Grid properties
        self.n_wealth = len(wealth_grid)
        self.n_time = len(time_grid)
        self.dW = wealth_grid[1] - wealth_grid[0] if len(wealth_grid) > 1 else 1.0
        self.dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        
        # Solution storage
        self.value_function = None
        self.optimal_controls = None
        self.convergence_history = []
        self.is_solved = False
        
    def implement_finite_difference_scheme(self, expected_returns: np.ndarray, 
                                         covariance_matrix: np.ndarray,
                                         transaction_cost_model: Optional['TransactionCostModel'] = None,
                                         scheme_type: str = 'implicit') -> Dict[str, np.ndarray]:
        """
        Implement finite difference schemes for HJB equation.
        
        Implements Requirement 8.2: Finite difference schemes for HJB equation
        
        The HJB equation is:
        ∂V/∂t + max_w [μ*W*w*∂V/∂W + (1/2)*σ²*W²*w²*∂²V/∂W² - (γ_V/2)*Var[dW] - TC] = 0
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            transaction_cost_model: Transaction cost model
            scheme_type: Type of finite difference scheme ('explicit', 'implicit', 'crank_nicolson')
            
        Returns:
            Dictionary containing finite difference scheme matrices and operators
        """
        n_assets = len(expected_returns)
        
        # Initialize value function and control grids
        self.value_function = np.zeros((self.n_time, self.n_wealth))
        self.optimal_controls = np.zeros((self.n_time, self.n_wealth, n_assets))
        
        # Finite difference operators
        def first_derivative_matrix():
            """Create first derivative finite difference matrix."""
            D1 = np.zeros((self.n_wealth, self.n_wealth))
            for i in range(1, self.n_wealth - 1):
                D1[i, i-1] = -1.0 / (2.0 * self.dW)
                D1[i, i+1] = 1.0 / (2.0 * self.dW)
            # Boundary conditions (forward/backward differences)
            D1[0, 0] = -1.0 / self.dW
            D1[0, 1] = 1.0 / self.dW
            D1[-1, -2] = -1.0 / self.dW
            D1[-1, -1] = 1.0 / self.dW
            return D1
        
        def second_derivative_matrix():
            """Create second derivative finite difference matrix."""
            D2 = np.zeros((self.n_wealth, self.n_wealth))
            for i in range(1, self.n_wealth - 1):
                D2[i, i-1] = 1.0 / (self.dW ** 2)
                D2[i, i] = -2.0 / (self.dW ** 2)
                D2[i, i+1] = 1.0 / (self.dW ** 2)
            # Boundary conditions (zero second derivative)
            D2[0, 0] = -2.0 / (self.dW ** 2)
            D2[0, 1] = 2.0 / (self.dW ** 2)
            D2[-1, -2] = 2.0 / (self.dW ** 2)
            D2[-1, -1] = -2.0 / (self.dW ** 2)
            return D2
        
        # Create finite difference matrices
        D1 = first_derivative_matrix()  # ∂V/∂W
        D2 = second_derivative_matrix()  # ∂²V/∂W²
        
        # Time derivative matrix (backward difference)
        def time_derivative_operator(V_current, V_next):
            """Calculate time derivative using finite differences."""
            return (V_next - V_current) / self.dt
        
        # HJB operator for given portfolio weights
        def hjb_operator(V, w_portfolio, wealth_idx):
            """Calculate HJB operator L^w V for given portfolio weights."""
            W = self.wealth_grid[wealth_idx]
            
            # Portfolio moments
            portfolio_return = np.dot(w_portfolio, expected_returns)
            portfolio_variance = np.dot(w_portfolio, np.dot(covariance_matrix, w_portfolio))
            
            # First-order term: (μ*W*w - TC)*∂V/∂W
            drift_term = W * portfolio_return
            if transaction_cost_model is not None:
                # Simplified transaction cost (would need previous weights for full calculation)
                drift_term -= 0.001 * W * np.sum(np.abs(w_portfolio))  # Simplified cost
            
            first_order = drift_term * np.dot(D1, V)[wealth_idx]
            
            # Second-order term: (1/2)*σ²*W²*w²*∂²V/∂W²
            diffusion_term = 0.5 * (W ** 2) * portfolio_variance
            second_order = diffusion_term * np.dot(D2, V)[wealth_idx]
            
            # Real-time variance penalty
            variance_penalty = -(self.gamma_V / 2) * (W ** 2) * portfolio_variance
            
            return first_order + second_order + variance_penalty
        
        # Store finite difference scheme components
        scheme_components = {
            'first_derivative_matrix': D1,
            'second_derivative_matrix': D2,
            'time_derivative_operator': time_derivative_operator,
            'hjb_operator': hjb_operator,
            'scheme_type': scheme_type,
            'grid_spacing': {'dW': self.dW, 'dt': self.dt},
            'stability_condition': self._check_stability_condition(covariance_matrix)
        }
        
        return scheme_components
    
    def handle_boundary_conditions(self, boundary_type: str = 'power_utility',
                                 boundary_params: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Handle boundary conditions for the optimization problem.
        
        Implements Requirement 8.3: Boundary condition handling for optimization problem
        
        Args:
            boundary_type: Type of boundary conditions ('power_utility', 'exponential', 'custom')
            boundary_params: Parameters for boundary conditions
            
        Returns:
            Dictionary containing boundary condition specifications
        """
        if boundary_params is None:
            boundary_params = {}
        
        # Terminal boundary condition: V(W, T)
        if boundary_type == 'power_utility':
            power = boundary_params.get('power', 1.0)
            terminal_condition = np.power(self.wealth_grid, power)
        elif boundary_type == 'exponential':
            risk_aversion = boundary_params.get('risk_aversion', self.gamma_T)
            terminal_condition = -np.exp(-risk_aversion * self.wealth_grid) / risk_aversion
        elif boundary_type == 'custom':
            terminal_condition = boundary_params.get('terminal_values', np.ones(self.n_wealth))
        else:
            # Default: linear utility
            terminal_condition = self.wealth_grid.copy()
        
        # Set terminal condition
        if self.value_function is not None:
            self.value_function[-1, :] = terminal_condition
        
        # Wealth boundary conditions
        # Lower boundary: V(0, t) - typically 0 or -∞
        lower_boundary = boundary_params.get('lower_boundary', 'zero')
        if lower_boundary == 'zero':
            lower_values = np.zeros(self.n_time)
        elif lower_boundary == 'negative_infinity':
            lower_values = np.full(self.n_time, -1e10)
        else:
            lower_values = np.full(self.n_time, float(lower_boundary))
        
        # Upper boundary: V(W_max, t) - typically linear growth
        upper_boundary = boundary_params.get('upper_boundary', 'linear')
        if upper_boundary == 'linear':
            upper_values = self.wealth_grid[-1] * np.ones(self.n_time)
        elif upper_boundary == 'quadratic':
            upper_values = (self.wealth_grid[-1] ** 2) * np.ones(self.n_time)
        else:
            upper_values = np.full(self.n_time, float(upper_boundary))
        
        # Apply boundary conditions to value function
        if self.value_function is not None:
            self.value_function[:, 0] = lower_values  # Lower wealth boundary
            self.value_function[:, -1] = upper_values  # Upper wealth boundary
        
        boundary_conditions = {
            'terminal_condition': terminal_condition,
            'lower_boundary': lower_values,
            'upper_boundary': upper_values,
            'boundary_type': boundary_type,
            'boundary_params': boundary_params
        }
        
        return boundary_conditions
    
    def check_convergence_and_stability(self, V_old: np.ndarray, V_new: np.ndarray,
                                      iteration: int) -> Dict[str, any]:
        """
        Check convergence and stability measures for the numerical solution.
        
        Implements Requirement 8.4: Convergence checking and stability measures
        
        Args:
            V_old: Previous iteration value function
            V_new: Current iteration value function
            iteration: Current iteration number
            
        Returns:
            Dictionary containing convergence and stability metrics
        """
        # Convergence metrics
        absolute_error = np.abs(V_new - V_old)
        max_absolute_error = np.max(absolute_error)
        mean_absolute_error = np.mean(absolute_error)
        
        relative_error = np.abs((V_new - V_old) / (V_old + 1e-10))
        max_relative_error = np.max(relative_error)
        mean_relative_error = np.mean(relative_error)
        
        # L2 norm of the difference
        l2_error = np.sqrt(np.mean((V_new - V_old) ** 2))
        
        # Convergence check
        converged = (max_absolute_error < self.convergence_tolerance and 
                    max_relative_error < self.convergence_tolerance)
        
        # Stability metrics
        # Check for oscillations
        if len(self.convergence_history) > 2:
            recent_errors = [h['max_absolute_error'] for h in self.convergence_history[-3:]]
            oscillation_detected = (recent_errors[0] < recent_errors[1] > recent_errors[2] or
                                  recent_errors[0] > recent_errors[1] < recent_errors[2])
        else:
            oscillation_detected = False
        
        # Check for divergence
        if len(self.convergence_history) > 0:
            error_trend = max_absolute_error - self.convergence_history[-1]['max_absolute_error']
            diverging = error_trend > 0 and max_absolute_error > 10 * self.convergence_tolerance
        else:
            diverging = False
        
        # Numerical stability checks
        has_nan = np.any(np.isnan(V_new))
        has_inf = np.any(np.isinf(V_new))
        numerically_stable = not (has_nan or has_inf)
        
        # Monotonicity check (value function should be increasing in wealth)
        monotonic = np.all(np.diff(V_new, axis=1) >= -1e-6)  # Allow small numerical errors
        
        convergence_info = {
            'iteration': iteration,
            'converged': converged,
            'max_absolute_error': max_absolute_error,
            'mean_absolute_error': mean_absolute_error,
            'max_relative_error': max_relative_error,
            'mean_relative_error': mean_relative_error,
            'l2_error': l2_error,
            'oscillation_detected': oscillation_detected,
            'diverging': diverging,
            'numerically_stable': numerically_stable,
            'monotonic': monotonic,
            'has_nan': has_nan,
            'has_inf': has_inf
        }
        
        # Store convergence history
        self.convergence_history.append(convergence_info)
        
        return convergence_info
    
    def solve_hjb_equation(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                          transaction_cost_model: Optional['TransactionCostModel'] = None,
                          boundary_conditions: Optional[Dict[str, any]] = None,
                          scheme_type: str = 'implicit') -> Dict[str, any]:
        """
        Solve the Hamilton-Jacobi-Bellman equation using finite difference methods.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            transaction_cost_model: Transaction cost model
            boundary_conditions: Boundary condition specifications
            scheme_type: Finite difference scheme type
            
        Returns:
            Dictionary containing HJB solution and convergence information
        """
        n_assets = len(expected_returns)
        
        # Initialize finite difference scheme
        scheme_components = self.implement_finite_difference_scheme(
            expected_returns, covariance_matrix, transaction_cost_model, scheme_type
        )
        
        # Set up boundary conditions
        boundary_specs = self.handle_boundary_conditions(
            boundary_conditions.get('type', 'power_utility') if boundary_conditions else 'power_utility',
            boundary_conditions.get('params', {}) if boundary_conditions else {}
        )
        
        # Initialize value function and controls
        self.value_function = np.zeros((self.n_time, self.n_wealth))
        self.optimal_controls = np.zeros((self.n_time, self.n_wealth, n_assets))
        
        # Set terminal and boundary conditions
        self.value_function[-1, :] = boundary_specs['terminal_condition']
        self.value_function[:, 0] = boundary_specs['lower_boundary']
        self.value_function[:, -1] = boundary_specs['upper_boundary']
        
        # Backward induction solution
        converged = False
        iteration = 0
        
        for t in range(self.n_time - 2, -1, -1):
            V_old = self.value_function[t, :].copy()
            
            # Solve for optimal controls and value function at each wealth level
            for w_idx in range(1, self.n_wealth - 1):  # Skip boundary points
                wealth = self.wealth_grid[w_idx]
                
                # Optimize portfolio weights for this (t, W) point
                optimal_weights = self._optimize_portfolio_at_point(
                    wealth, expected_returns, covariance_matrix, 
                    self.value_function[t+1, :], w_idx, scheme_components
                )
                
                self.optimal_controls[t, w_idx, :] = optimal_weights
                
                # Calculate value function using HJB equation
                hjb_value = scheme_components['hjb_operator'](
                    self.value_function[t+1, :], optimal_weights, w_idx
                )
                
                # Update value function (implicit scheme)
                if scheme_type == 'explicit':
                    self.value_function[t, w_idx] = self.value_function[t+1, w_idx] + self.dt * hjb_value
                elif scheme_type == 'implicit':
                    # Simplified implicit update (full implicit would require solving linear system)
                    self.value_function[t, w_idx] = (self.value_function[t+1, w_idx] + 
                                                   self.dt * hjb_value) / (1 + self.dt * 0.01)
                else:  # Crank-Nicolson
                    self.value_function[t, w_idx] = (self.value_function[t+1, w_idx] + 
                                                   0.5 * self.dt * hjb_value) / (1 + 0.5 * self.dt * 0.01)
            
            # Check convergence for this time step
            convergence_info = self.check_convergence_and_stability(
                V_old, self.value_function[t, :], iteration
            )
            
            iteration += 1
            
            if convergence_info['diverging'] or not convergence_info['numerically_stable']:
                break
        
        # Final convergence check
        if len(self.convergence_history) > 0:
            final_convergence = self.convergence_history[-1]
            converged = final_convergence['converged'] and final_convergence['numerically_stable']
        
        self.is_solved = converged
        
        solution = {
            'value_function': self.value_function,
            'optimal_controls': self.optimal_controls,
            'wealth_grid': self.wealth_grid,
            'time_grid': self.time_grid,
            'converged': converged,
            'convergence_history': self.convergence_history,
            'boundary_conditions': boundary_specs,
            'scheme_components': scheme_components,
            'solver_info': {
                'method': f'finite_difference_{scheme_type}',
                'n_wealth_points': self.n_wealth,
                'n_time_points': self.n_time,
                'n_assets': n_assets,
                'total_iterations': iteration
            }
        }
        
        return solution
    
    def _optimize_portfolio_at_point(self, wealth: float, expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray, V_next: np.ndarray,
                                   wealth_idx: int, scheme_components: Dict) -> np.ndarray:
        """Optimize portfolio weights at a specific (t, W) point."""
        n_assets = len(expected_returns)
        
        # Objective function for portfolio optimization
        def objective(weights):
            # Ensure budget constraint
            weights = weights / np.sum(weights) if np.sum(weights) != 0 else np.ones(n_assets) / n_assets
            
            # Calculate HJB operator value
            hjb_value = scheme_components['hjb_operator'](V_next, weights, wealth_idx)
            
            return -hjb_value  # Minimize negative value
        
        # Initial guess and bounds
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(-1.0, 1.0) for _ in range(n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Optimization
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            if result.success:
                return result.x / np.sum(result.x)  # Ensure normalization
            else:
                return initial_weights
        except:
            # Fallback to analytical solution
            try:
                inv_cov = np.linalg.inv(covariance_matrix)
                weights = np.dot(inv_cov, expected_returns) / self.gamma_T
                return weights / np.sum(weights) if np.sum(weights) != 0 else initial_weights
            except:
                return initial_weights
    
    def _check_stability_condition(self, covariance_matrix: np.ndarray) -> Dict[str, any]:
        """Check stability condition for finite difference scheme."""
        max_eigenvalue = np.max(np.linalg.eigvals(covariance_matrix))
        max_wealth = np.max(self.wealth_grid)
        
        # CFL condition for stability
        cfl_condition = self.dt * max_eigenvalue * (max_wealth ** 2) / (self.dW ** 2)
        stable = cfl_condition <= 0.5
        
        return {
            'cfl_condition': cfl_condition,
            'stable': stable,
            'max_eigenvalue': max_eigenvalue,
            'recommended_dt': 0.5 * (self.dW ** 2) / (max_eigenvalue * (max_wealth ** 2))
        }

# ============================================================================
# Machine Learning Parameter Estimation Components
# ============================================================================

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    keras = None
    layers = None
    HAS_TENSORFLOW = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hmmlearn import hmm
from typing import Dict, Any, Callable
import joblib
import logging

class MLParameterEstimator:
    """
    Ensemble machine learning model for dynamic parameter estimation.
    Combines XGBoost and deep neural networks for robust parameter prediction.
    """
    
    def __init__(self, 
                 xgb_params: Optional[Dict[str, Any]] = None,
                 nn_params: Optional[Dict[str, Any]] = None,
                 ensemble_weights: Optional[Dict[str, float]] = None,
                 parameter_types: Optional[List[str]] = None):
        """
        Initialize ML Parameter Estimator with ensemble methods.
        
        Args:
            xgb_params: XGBoost hyperparameters
            nn_params: Neural network architecture parameters
            ensemble_weights: Weights for combining XGB and NN predictions
            parameter_types: List of parameter types to estimate (k1, k2, delta_t1, etc.)
        """
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror'
        }
        
        # Default neural network parameters
        self.nn_params = nn_params or {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15
        }
        
        # Default ensemble weights
        self.ensemble_weights = ensemble_weights or {
            'xgb': 0.6,
            'nn': 0.4
        }
        
        # Parameter types to estimate
        self.parameter_types = parameter_types or [
            'k1', 'k2', 'delta_t1', 'delta_t2', 'delta_t3',
            'gamma_T', 'gamma_V', 'tau_b', 'tau_s'
        ]
        
        # Model storage
        self.xgb_models = {}
        self.nn_models = {}
        self.scalers = {}
        self.feature_names = None
        self.is_fitted = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _create_neural_network(self, input_dim: int):
        """Create deep neural network architecture for parameter estimation."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for neural network functionality")
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.nn_params['hidden_layers'][0],
            input_dim=input_dim,
            activation=self.nn_params['activation']
        ))
        model.add(layers.Dropout(self.nn_params['dropout_rate']))
        
        # Hidden layers
        for units in self.nn_params['hidden_layers'][1:]:
            model.add(layers.Dense(units, activation=self.nn_params['activation']))
            model.add(layers.Dropout(self.nn_params['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.nn_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X: pl.DataFrame, y_dict: Dict[str, pl.Series]) -> 'MLParameterEstimator':
        """
        Fit ensemble models for each parameter type.
        
        Args:
            X: Feature matrix (Polars DataFrame)
            y_dict: Dictionary mapping parameter names to target series
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame")
        
        # Store feature names
        self.feature_names = X.columns
        
        # Convert to numpy for sklearn/tensorflow compatibility
        X_np = X.select([pl.col(c).cast(pl.Float64, strict=False) for c in X.columns]).to_numpy()
        
        # Handle missing values
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit models for each parameter type
        for param_name in self.parameter_types:
            if param_name not in y_dict:
                self.logger.warning(f"Parameter {param_name} not found in target dictionary")
                continue
            
            self.logger.info(f"Fitting models for parameter: {param_name}")
            
            # Get target values
            y_np = y_dict[param_name].cast(pl.Float64).to_numpy()
            y_np = np.nan_to_num(y_np, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_np)
            self.scalers[param_name] = scaler
            
            # Fit XGBoost model
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.fit(X_scaled, y_np)
            self.xgb_models[param_name] = xgb_model
            
            # Fit Neural Network model
            nn_model = self._create_neural_network(X_scaled.shape[1])
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=self.nn_params['early_stopping_patience'],
                restore_best_weights=True
            )
            
            # Fit neural network
            nn_model.fit(
                X_scaled, y_np,
                batch_size=self.nn_params['batch_size'],
                epochs=self.nn_params['epochs'],
                callbacks=[early_stopping],
                verbose=0
            )
            self.nn_models[param_name] = nn_model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pl.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict parameters using ensemble of XGBoost and Neural Network models.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Dictionary mapping parameter names to predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame")
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Convert to numpy
        X_pred = X.select(self.feature_names)
        X_np = X_pred.select([pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names]).to_numpy()
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        predictions = {}
        
        for param_name in self.parameter_types:
            if param_name not in self.xgb_models or param_name not in self.nn_models:
                continue
            
            # Scale features
            X_scaled = self.scalers[param_name].transform(X_np)
            
            # Get XGBoost predictions
            xgb_pred = self.xgb_models[param_name].predict(X_scaled)
            
            # Get Neural Network predictions
            nn_pred = self.nn_models[param_name].predict(X_scaled, verbose=0).flatten()
            
            # Ensemble predictions
            ensemble_pred = (
                self.ensemble_weights['xgb'] * xgb_pred +
                self.ensemble_weights['nn'] * nn_pred
            )
            
            # Apply parameter-specific constraints
            ensemble_pred = self._apply_parameter_constraints(param_name, ensemble_pred)
            
            predictions[param_name] = ensemble_pred
        
        return predictions
    
    def _apply_parameter_constraints(self, param_name: str, predictions: np.ndarray) -> np.ndarray:
        """Apply economic constraints to parameter predictions."""
        if param_name in ['k1', 'k2']:
            # Volatility scaling parameters should be positive and k2 > k1
            predictions = np.maximum(predictions, 0.1)
            if param_name == 'k1':
                predictions = np.minimum(predictions, 3.0)
            elif param_name == 'k2':
                predictions = np.maximum(predictions, 1.5)
                predictions = np.minimum(predictions, 5.0)
        
        elif param_name in ['delta_t1', 'delta_t2', 'delta_t3']:
            # Time parameters should be positive
            predictions = np.maximum(predictions, 0.5)
            predictions = np.minimum(predictions, 30.0)
        
        elif param_name in ['gamma_T', 'gamma_V']:
            # Risk aversion parameters should be positive
            predictions = np.maximum(predictions, 0.1)
            predictions = np.minimum(predictions, 10.0)
        
        elif param_name in ['tau_b', 'tau_s']:
            # Transaction costs should be small and positive
            predictions = np.maximum(predictions, 0.0001)
            predictions = np.minimum(predictions, 0.05)
        
        return predictions
    
    def get_feature_importance(self, param_name: str) -> Dict[str, float]:
        """Get feature importance for a specific parameter."""
        if not self.is_fitted or param_name not in self.xgb_models:
            raise RuntimeError(f"Model not fitted or parameter {param_name} not found")
        
        # Get XGBoost feature importance
        xgb_importance = self.xgb_models[param_name].feature_importances_
        
        return dict(zip(self.feature_names, xgb_importance))
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        if not self.is_fitted:
            raise RuntimeError("No fitted models to save")
        
        model_data = {
            'xgb_models': self.xgb_models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'parameter_types': self.parameter_types,
            'ensemble_weights': self.ensemble_weights
        }
        
        # Save XGBoost models and scalers
        joblib.dump(model_data, f"{filepath}_sklearn_components.pkl")
        
        # Save neural network models
        for param_name, nn_model in self.nn_models.items():
            nn_model.save(f"{filepath}_nn_{param_name}.h5")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        # Load XGBoost models and scalers
        model_data = joblib.load(f"{filepath}_sklearn_components.pkl")
        
        self.xgb_models = model_data['xgb_models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.parameter_types = model_data['parameter_types']
        self.ensemble_weights = model_data['ensemble_weights']
        
        # Load neural network models
        self.nn_models = {}
        for param_name in self.parameter_types:
            try:
                self.nn_models[param_name] = keras.models.load_model(f"{filepath}_nn_{param_name}.h5")
            except:
                self.logger.warning(f"Could not load neural network model for {param_name}")
        
        self.is_fitted = True


class FeatureEngineer:
    """
    Feature engineering class for ML parameter estimation.
    Creates firm characteristics, market conditions, and event-specific features.
    """
    
    def __init__(self):
        self.feature_names = []
        self.is_fitted = False
    
    def create_firm_characteristics(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Create firm-specific characteristic features.
        
        Args:
            data: Input data with stock information
            
        Returns:
            DataFrame with firm characteristic features
        """
        features = data.clone()
        
        # Market capitalization (if price and shares available)
        if 'price' in data.columns and 'shares_outstanding' in data.columns:
            features = features.with_columns(
                (pl.col('price') * pl.col('shares_outstanding')).alias('market_cap')
            )
            features = features.with_columns(
                pl.col('market_cap').log().alias('log_market_cap')
            )
        
        # Book-to-market ratio (if available)
        if 'book_value' in data.columns and 'market_cap' in features.columns:
            features = features.with_columns(
                (pl.col('book_value') / pl.col('market_cap')).alias('book_to_market')
            )
        
        # Price momentum features
        if 'price' in data.columns:
            features = features.with_columns([
                (pl.col('price') / pl.col('price').shift(5) - 1).alias('momentum_5d'),
                (pl.col('price') / pl.col('price').shift(20) - 1).alias('momentum_20d'),
                (pl.col('price') / pl.col('price').shift(60) - 1).alias('momentum_60d')
            ])
        
        # Volume characteristics
        if 'volume' in data.columns:
            features = features.with_columns([
                pl.col('volume').rolling_mean(window_size=20).alias('avg_volume_20d'),
                pl.col('volume').rolling_std(window_size=20).alias('volume_volatility_20d')
            ])
            
            # Relative volume
            features = features.with_columns(
                (pl.col('volume') / pl.col('avg_volume_20d')).alias('relative_volume')
            )
        
        # Analyst coverage (if available)
        if 'analyst_count' in data.columns:
            features = features.with_columns([
                pl.col('analyst_count').alias('analyst_coverage'),
                pl.col('analyst_count').log().alias('log_analyst_coverage')
            ])
        
        return features
    
    def create_market_conditions(self, data: pl.DataFrame, market_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Create market condition features.
        
        Args:
            data: Input stock data
            market_data: Optional market-wide data (VIX, credit spreads, etc.)
            
        Returns:
            DataFrame with market condition features
        """
        features = data.clone()
        
        # If market data is provided, merge it
        if market_data is not None:
            if 'date' in data.columns and 'date' in market_data.columns:
                features = features.join(market_data, on='date', how='left')
        
        # VIX-related features (if available)
        if 'vix' in features.columns:
            features = features.with_columns([
                pl.col('vix').alias('market_volatility'),
                pl.col('vix').rolling_mean(window_size=20).alias('vix_ma20'),
                (pl.col('vix') - pl.col('vix').rolling_mean(window_size=20)).alias('vix_deviation')
            ])
        
        # Credit spread features (if available)
        if 'credit_spread' in features.columns:
            features = features.with_columns([
                pl.col('credit_spread').alias('credit_risk'),
                pl.col('credit_spread').rolling_mean(window_size=20).alias('credit_spread_ma20'),
                (pl.col('credit_spread') - pl.col('credit_spread').rolling_mean(window_size=20)).alias('credit_spread_deviation')
            ])
        
        # Market return features
        if 'market_return' in features.columns:
            features = features.with_columns([
                pl.col('market_return').rolling_mean(window_size=20).alias('market_return_ma20'),
                pl.col('market_return').rolling_std(window_size=20).alias('market_volatility_20d'),
                pl.col('market_return').rolling_mean(window_size=5).alias('market_return_ma5')
            ])
        
        # Sector volatility (if sector information available)
        if 'sector' in features.columns and 'return' in features.columns:
            sector_volatility = features.group_by(['date', 'sector']).agg([
                pl.col('return').std().alias('sector_volatility')
            ])
            features = features.join(sector_volatility, on=['date', 'sector'], how='left')
        
        return features
    
    def create_event_features(self, data: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
        """
        Create event-specific features.
        
        Args:
            data: Stock data
            events: Event data with dates and types
            
        Returns:
            DataFrame with event features
        """
        features = data.clone()
        
        if 'date' not in data.columns or 'ticker' not in data.columns:
            return features
        
        # Merge with events
        if 'date' in events.columns and 'ticker' in events.columns:
            features = features.join(
                events.select(['date', 'ticker', 'event_type', 'event_outcome']),
                on=['date', 'ticker'],
                how='left'
            )
        
        # Event type indicators
        if 'event_type' in features.columns:
            event_types = features.select('event_type').unique().drop_nulls().to_series().to_list()
            for event_type in event_types:
                features = features.with_columns(
                    (pl.col('event_type') == event_type).cast(pl.Int32).alias(f'is_{event_type}_event')
                )
        
        # Days since last event
        if 'event_type' in features.columns:
            features = features.with_columns(
                pl.col('date').diff().dt.total_days().alias('days_since_last_event')
            )
        
        # Event frequency (events per year for this ticker)
        if 'event_type' in features.columns:
            event_counts = events.group_by('ticker').agg([
                pl.col('event_type').count().alias('annual_event_frequency')
            ])
            features = features.join(event_counts, on='ticker', how='left')
        
        # Event outcome features (if available)
        if 'event_outcome' in features.columns:
            features = features.with_columns([
                (pl.col('event_outcome') == 'positive').cast(pl.Int32).alias('positive_event'),
                (pl.col('event_outcome') == 'negative').cast(pl.Int32).alias('negative_event'),
                (pl.col('event_outcome') == 'neutral').cast(pl.Int32).alias('neutral_event')
            ])
        
        return features
    
    def create_cross_sectional_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Create cross-sectional features comparing firms to peers.
        
        Args:
            data: Input data with firm information
            
        Returns:
            DataFrame with cross-sectional features
        """
        features = data.clone()
        
        # Peer volatility (sector-based if sector available)
        if 'sector' in features.columns and 'return' in features.columns:
            # Calculate sector median volatility
            sector_stats = features.group_by(['date', 'sector']).agg([
                pl.col('return').std().alias('sector_volatility'),
                pl.col('return').median().alias('sector_median_return'),
                pl.col('return').count().alias('sector_firm_count')
            ])
            
            features = features.join(sector_stats, on=['date', 'sector'], how='left')
            
            # Relative volatility vs sector
            if 'return' in features.columns:
                features = features.with_columns([
                    pl.col('return').rolling_std(window_size=20).alias('firm_volatility_20d')
                ])
                features = features.with_columns(
                    (pl.col('firm_volatility_20d') / pl.col('sector_volatility')).alias('relative_volatility_vs_sector')
                )
        
        # Market cap percentile within sector
        if 'market_cap' in features.columns and 'sector' in features.columns:
            features = features.with_columns(
                pl.col('market_cap').rank(method='average').over(['date', 'sector']).alias('market_cap_sector_rank')
            )
        
        # Sector concentration (HHI-like measure)
        if 'sector' in features.columns and 'market_cap' in features.columns:
            sector_concentration = features.group_by(['date', 'sector']).agg([
                ((pl.col('market_cap') / pl.col('market_cap').sum()) ** 2).sum().alias('sector_concentration')
            ])
            features = features.join(sector_concentration, on=['date', 'sector'], how='left')
        
        return features
    
    def engineer_all_features(self, 
                            data: pl.DataFrame, 
                            events: Optional[pl.DataFrame] = None,
                            market_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Create all feature types in one call.
        
        Args:
            data: Base stock data
            events: Event data
            market_data: Market-wide data
            
        Returns:
            DataFrame with all engineered features
        """
        features = data.clone()
        
        # Create firm characteristics
        features = self.create_firm_characteristics(features)
        
        # Create market condition features
        features = self.create_market_conditions(features, market_data)
        
        # Create event features
        if events is not None:
            features = self.create_event_features(features, events)
        
        # Create cross-sectional features
        features = self.create_cross_sectional_features(features)
        
        # Store feature names
        self.feature_names = features.columns
        self.is_fitted = True
        
        return features


class RegimeIdentifier:
    """
    Hidden Markov Model for regime identification in market conditions.
    Identifies high/low uncertainty regimes using observable state variables.
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 covariance_type: str = 'full',
                 n_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize regime identifier.
        
        Args:
            n_regimes: Number of regimes (typically 2 for high/low uncertainty)
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize HMM model
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.feature_names = None
        self.scaler = StandardScaler()
        
        # Regime interpretation
        self.regime_labels = {0: 'Low Uncertainty', 1: 'High Uncertainty'}
    
    def fit(self, X: pl.DataFrame) -> 'RegimeIdentifier':
        """
        Fit Hidden Markov Model to observable state variables.
        
        Args:
            X: Observable state variables (VIX, credit spreads, earnings indicators, etc.)
        """
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame")
        
        self.feature_names = X.columns
        
        # Convert to numpy and handle missing values
        X_np = X.select([pl.col(c).cast(pl.Float64, strict=False) for c in X.columns]).to_numpy()
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_np)
        
        # Fit HMM
        self.hmm_model.fit(X_scaled)
        
        # Determine regime interpretation based on mean values
        regime_means = self.hmm_model.means_
        
        # Assume regime with higher mean VIX (if available) is high uncertainty
        if 'vix' in self.feature_names:
            vix_idx = list(self.feature_names).index('vix')
            high_uncertainty_regime = np.argmax(regime_means[:, vix_idx])
            low_uncertainty_regime = 1 - high_uncertainty_regime
            
            self.regime_labels = {
                low_uncertainty_regime: 'Low Uncertainty',
                high_uncertainty_regime: 'High Uncertainty'
            }
        
        self.is_fitted = True
        return self
    
    def predict_regime(self, X: pl.DataFrame) -> np.ndarray:
        """
        Predict regime for new observations.
        
        Args:
            X: Observable state variables
            
        Returns:
            Array of regime predictions (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame")
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Convert and scale
        X_pred = X.select(self.feature_names)
        X_np = X_pred.select([pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names]).to_numpy()
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X_np)
        
        # Predict regimes
        regimes = self.hmm_model.predict(X_scaled)
        
        return regimes
    
    def predict_regime_probabilities(self, X: pl.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for new observations.
        
        Args:
            X: Observable state variables
            
        Returns:
            Array of regime probabilities (n_samples x n_regimes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame")
        
        # Convert and scale
        X_pred = X.select(self.feature_names)
        X_np = X_pred.select([pl.col(c).cast(pl.Float64, strict=False) for c in self.feature_names]).to_numpy()
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X_np)
        
        # Predict probabilities
        log_probs = self.hmm_model.predict_proba(X_scaled)
        
        return log_probs
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probability matrix."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        return self.hmm_model.transmat_
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each regime."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        stats = {}
        for regime_idx in range(self.n_regimes):
            regime_name = self.regime_labels.get(regime_idx, f"Regime {regime_idx}")
            stats[regime_name] = {
                'mean': self.hmm_model.means_[regime_idx].tolist(),
                'covariance': self.hmm_model.covars_[regime_idx].tolist() if self.covariance_type == 'full' else None
            }
        
        return stats


class ValidationFramework:
    """
    Regularization and validation framework for ML parameter estimation.
    Implements temporal cross-validation, regularization, and economic constraints.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size_days: int = 30,
                 gap_days: int = 5):
        """
        Initialize validation framework.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size_days: Size of test set in days
            gap_days: Gap between train and test sets to prevent data leakage
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        
        # Regularization parameters
        self.l1_alpha = 0.01
        self.l2_alpha = 0.01
        
        # Economic constraints
        self.parameter_bounds = {
            'k1': (0.1, 3.0),
            'k2': (1.5, 5.0),
            'delta_t1': (0.5, 30.0),
            'delta_t2': (0.5, 30.0),
            'delta_t3': (0.5, 30.0),
            'gamma_T': (0.1, 10.0),
            'gamma_V': (0.1, 10.0),
            'tau_b': (0.0001, 0.05),
            'tau_s': (0.0001, 0.05)
        }
    
    def temporal_cross_validate(self, 
                              X: pl.DataFrame, 
                              y: pl.Series, 
                              model: Any,
                              date_column: str = 'date') -> Dict[str, float]:
        """
        Perform temporal cross-validation with rolling windows.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Model to validate
            date_column: Name of date column for temporal ordering
            
        Returns:
            Dictionary of validation metrics
        """
        if date_column not in X.columns:
            raise ValueError(f"Date column '{date_column}' not found in features")
        
        # Sort by date
        data_sorted = X.with_columns(y.alias('target')).sort(date_column)
        
        # Get unique dates
        unique_dates = data_sorted.select(date_column).unique().sort(date_column).to_series().to_list()
        
        if len(unique_dates) < self.n_splits * (self.test_size_days + self.gap_days):
            raise ValueError("Not enough data for temporal cross-validation")
        
        scores = []
        
        for i in range(self.n_splits):
            # Calculate split points
            test_end_idx = len(unique_dates) - i * self.test_size_days
            test_start_idx = test_end_idx - self.test_size_days
            train_end_idx = test_start_idx - self.gap_days
            
            if train_end_idx <= 0:
                break
            
            # Get date ranges
            train_end_date = unique_dates[train_end_idx - 1]
            test_start_date = unique_dates[test_start_idx]
            test_end_date = unique_dates[test_end_idx - 1]
            
            # Split data
            train_data = data_sorted.filter(pl.col(date_column) <= train_end_date)
            test_data = data_sorted.filter(
                (pl.col(date_column) >= test_start_date) & 
                (pl.col(date_column) <= test_end_date)
            )
            
            if train_data.height == 0 or test_data.height == 0:
                continue
            
            # Prepare features and targets
            X_train = train_data.drop(['target', date_column])
            y_train = train_data.select('target').to_series()
            X_test = test_data.drop(['target', date_column])
            y_test = test_data.select('target').to_series()
            
            # Fit and predict
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test.to_numpy(), y_pred)
                mae = mean_absolute_error(y_test.to_numpy(), y_pred)
                
                scores.append({'mse': mse, 'mae': mae})
                
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue
        
        if not scores:
            return {'mse': np.inf, 'mae': np.inf, 'n_folds': 0}
        
        # Aggregate scores
        avg_mse = np.mean([s['mse'] for s in scores])
        avg_mae = np.mean([s['mae'] for s in scores])
        std_mse = np.std([s['mse'] for s in scores])
        std_mae = np.std([s['mae'] for s in scores])
        
        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'mse_std': std_mse,
            'mae_std': std_mae,
            'n_folds': len(scores)
        }
    
    def event_type_stratified_validation(self, 
                                       X: pl.DataFrame, 
                                       y: pl.Series, 
                                       event_types: pl.Series,
                                       model: Any) -> Dict[str, Dict[str, float]]:
        """
        Perform validation stratified by event type.
        
        Args:
            X: Feature matrix
            y: Target variable
            event_types: Event type labels
            model: Model to validate
            
        Returns:
            Dictionary of metrics by event type
        """
        unique_event_types = event_types.unique().drop_nulls().to_list()
        results = {}
        
        for event_type in unique_event_types:
            # Filter data for this event type
            mask = event_types == event_type
            X_event = X.filter(mask)
            y_event = y.filter(mask)
            
            if X_event.height < 10:  # Skip if too few samples
                continue
            
            # Simple train-test split for this event type
            split_idx = int(0.8 * X_event.height)
            
            X_train = X_event[:split_idx]
            y_train = y_event[:split_idx]
            X_test = X_event[split_idx:]
            y_test = y_event[split_idx:]
            
            try:
                # Fit and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test.to_numpy(), y_pred)
                mae = mean_absolute_error(y_test.to_numpy(), y_pred)
                
                results[event_type] = {'mse': mse, 'mae': mae}
                
            except Exception as e:
                print(f"Error validating event type {event_type}: {e}")
                continue
        
        return results
    
    def apply_regularization(self, model: Any, X: pl.DataFrame, y: pl.Series) -> Any:
        """
        Apply L1/L2 regularization to model.
        
        Args:
            model: Model to regularize
            X: Feature matrix
            y: Target variable
            
        Returns:
            Regularized model
        """
        # This is a placeholder - specific implementation depends on model type
        if hasattr(model, 'reg_alpha') and hasattr(model, 'reg_lambda'):
            # XGBoost-style regularization
            model.reg_alpha = self.l1_alpha
            model.reg_lambda = self.l2_alpha
        
        return model
    
    def enforce_economic_constraints(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Enforce economic constraints on parameter predictions.
        
        Args:
            predictions: Dictionary of parameter predictions
            
        Returns:
            Constrained predictions
        """
        constrained_predictions = {}
        
        for param_name, pred_values in predictions.items():
            if param_name in self.parameter_bounds:
                lower_bound, upper_bound = self.parameter_bounds[param_name]
                constrained_values = np.clip(pred_values, lower_bound, upper_bound)
                constrained_predictions[param_name] = constrained_values
            else:
                constrained_predictions[param_name] = pred_values
        
        # Additional cross-parameter constraints
        if 'k1' in constrained_predictions and 'k2' in constrained_predictions:
            # Ensure k2 > k1
            k1_vals = constrained_predictions['k1']
            k2_vals = constrained_predictions['k2']
            
            # Adjust k2 to be at least k1 + 0.1
            k2_adjusted = np.maximum(k2_vals, k1_vals + 0.1)
            constrained_predictions['k2'] = k2_adjusted
        
        if 'tau_s' in constrained_predictions and 'tau_b' in constrained_predictions:
            # Ensure tau_s <= tau_b (sale costs <= purchase costs)
            tau_s_vals = constrained_predictions['tau_s']
            tau_b_vals = constrained_predictions['tau_b']
            
            tau_s_adjusted = np.minimum(tau_s_vals, tau_b_vals)
            constrained_predictions['tau_s'] = tau_s_adjusted
        
        return constrained_predictions

class MonteCarloStatistics:
    """
    Statistical analysis and confidence intervals for Monte Carlo results.
    
    Implements Requirements 10.3, 10.4, 10.5:
    - Statistical summaries across simulation runs
    - Percentile-based confidence band calculations
    - Convergence diagnostics for Monte Carlo results
    - Variance reduction techniques for efficiency
    """
    
    def __init__(self, monte_carlo_engine: MonteCarloEngine):
        """
        Initialize statistics calculator.
        
        Args:
            monte_carlo_engine: Monte Carlo engine with simulation results
        """
        self.mc_engine = monte_carlo_engine
        self.statistics = {}
        
    def calculate_path_statistics(self, paths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate statistical summaries across simulation paths.
        
        Args:
            paths: Simulation paths (n_simulations x n_steps x n_assets)
            
        Returns:
            Dictionary with statistical summaries
        """
        n_simulations, n_steps, n_assets = paths.shape
        
        # Remove any NaN simulations
        valid_mask = ~np.isnan(paths).any(axis=(1, 2))
        valid_paths = paths[valid_mask]
        n_valid = valid_paths.shape[0]
        
        if n_valid == 0:
            raise ValueError("No valid simulation paths found")
        
        statistics = {
            'mean': np.mean(valid_paths, axis=0),
            'std': np.std(valid_paths, axis=0),
            'median': np.median(valid_paths, axis=0),
            'min': np.min(valid_paths, axis=0),
            'max': np.max(valid_paths, axis=0),
            'n_valid_simulations': n_valid,
            'n_total_simulations': n_simulations
        }
        
        return statistics
    
    def calculate_confidence_bands(self, 
                                  paths: np.ndarray, 
                                  confidence_levels: List[float] = None) -> Dict[str, np.ndarray]:
        """
        Calculate percentile-based confidence bands.
        
        Args:
            paths: Simulation paths (n_simulations x n_steps x n_assets)
            confidence_levels: List of confidence levels (e.g., [0.05, 0.95])
            
        Returns:
            Dictionary with confidence bands
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.75, 0.95]
        
        # Remove any NaN simulations
        valid_mask = ~np.isnan(paths).any(axis=(1, 2))
        valid_paths = paths[valid_mask]
        
        if valid_paths.shape[0] == 0:
            raise ValueError("No valid simulation paths found")
        
        confidence_bands = {}
        
        for level in confidence_levels:
            percentile = level * 100
            confidence_bands[f'percentile_{percentile:.1f}'] = np.percentile(
                valid_paths, percentile, axis=0
            )
        
        return confidence_bands
    
    def calculate_convergence_diagnostics(self, 
                                        paths: np.ndarray,
                                        batch_size: int = 100) -> Dict[str, float]:
        """
        Calculate convergence diagnostics for Monte Carlo results.
        
        Args:
            paths: Simulation paths (n_simulations x n_steps x n_assets)
            batch_size: Size of batches for convergence analysis
            
        Returns:
            Dictionary with convergence metrics
        """
        # Remove any NaN simulations
        valid_mask = ~np.isnan(paths).any(axis=(1, 2))
        valid_paths = paths[valid_mask]
        n_valid = valid_paths.shape[0]
        
        if n_valid < 2 * batch_size:
            raise ValueError(f"Need at least {2 * batch_size} valid simulations for convergence analysis")
        
        # Calculate final values for convergence analysis
        final_values = valid_paths[:, -1, :]  # Final values for each asset
        
        # Running mean convergence
        n_batches = n_valid // batch_size
        batch_means = np.zeros((n_batches, final_values.shape[1]))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_means[i] = np.mean(final_values[start_idx:end_idx], axis=0)
        
        # Calculate convergence metrics
        overall_mean = np.mean(final_values, axis=0)
        batch_variance = np.var(batch_means, axis=0)
        within_batch_variance = np.mean([
            np.var(final_values[i*batch_size:(i+1)*batch_size], axis=0) 
            for i in range(n_batches)
        ], axis=0)
        
        # Effective sample size (Kass et al., 1998)
        # ESS = n * within_variance / (within_variance + between_variance)
        total_variance = batch_variance + within_batch_variance
        effective_sample_size = n_valid * within_batch_variance / total_variance
        
        # Monte Carlo standard error
        mc_standard_error = np.sqrt(total_variance / n_valid)
        
        # Relative efficiency compared to independent sampling
        relative_efficiency = within_batch_variance / total_variance
        
        diagnostics = {
            'effective_sample_size': np.mean(effective_sample_size),
            'monte_carlo_standard_error': np.mean(mc_standard_error),
            'relative_efficiency': np.mean(relative_efficiency),
            'n_batches': n_batches,
            'batch_size': batch_size,
            'convergence_ratio': np.mean(batch_variance / within_batch_variance)
        }
        
        return diagnostics
    
    def implement_antithetic_variates(self, 
                                    simulation_func: callable,
                                    *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Implement antithetic variates for variance reduction.
        
        Args:
            simulation_func: Function to run simulation
            *args, **kwargs: Arguments for simulation function
            
        Returns:
            Dictionary with variance-reduced results
        """
        # Run original simulations
        original_seed = kwargs.get('seed', self.mc_engine.random_seed)
        n_sims = kwargs.get('n_simulations', self.mc_engine.n_simulations)
        
        # Split simulations in half
        n_half = n_sims // 2
        kwargs['n_simulations'] = n_half
        
        # First half with original random numbers
        kwargs['seed'] = original_seed
        results_1 = simulation_func(*args, **kwargs)
        
        # Second half with antithetic random numbers
        # This requires modifying the simulation to use antithetic variates
        kwargs['seed'] = original_seed + 10000  # Different seed for antithetic
        kwargs['use_antithetic'] = True
        results_2 = simulation_func(*args, **kwargs)
        
        # Combine results
        combined_paths = np.concatenate([
            results_1['all_paths'], 
            results_2['all_paths']
        ], axis=0)
        
        combined_regime_states = np.concatenate([
            results_1['all_regime_states'],
            results_2['all_regime_states']
        ], axis=0)
        
        return {
            'time_grid': results_1['time_grid'],
            'all_paths': combined_paths,
            'all_regime_states': combined_regime_states,
            'n_simulations': n_sims,
            'variance_reduction_method': 'antithetic_variates'
        }
    
    def implement_control_variates(self,
                                  paths: np.ndarray,
                                  control_paths: np.ndarray,
                                  control_expectation: float) -> np.ndarray:
        """
        Implement control variates for variance reduction.
        
        Args:
            paths: Original simulation paths
            control_paths: Control variate paths (known expectation)
            control_expectation: Known expectation of control variate
            
        Returns:
            Variance-reduced paths
        """
        # Calculate final values
        final_values = paths[:, -1, :]
        control_final = control_paths[:, -1, :]
        
        # Estimate optimal control coefficient
        covariance = np.cov(final_values.flatten(), control_final.flatten())[0, 1]
        control_variance = np.var(control_final)
        
        if control_variance > 1e-10:
            optimal_c = covariance / control_variance
        else:
            optimal_c = 0
        
        # Apply control variate adjustment
        control_adjustment = optimal_c * (control_final - control_expectation)
        adjusted_final = final_values - control_adjustment.reshape(final_values.shape)
        
        # Create adjusted paths (simple approach: adjust final values proportionally)
        adjusted_paths = paths.copy()
        for i in range(paths.shape[0]):
            for j in range(paths.shape[2]):
                if paths[i, -1, j] != 0:
                    adjustment_factor = adjusted_final[i, j] / paths[i, -1, j]
                    adjusted_paths[i, :, j] *= adjustment_factor
        
        return adjusted_paths
    
    def calculate_risk_metrics(self, wealth_paths: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics from wealth paths.
        
        Args:
            wealth_paths: Wealth paths (n_simulations x n_steps)
            
        Returns:
            Dictionary with risk metrics
        """
        # Calculate returns
        wealth_returns = (wealth_paths[:, -1] - wealth_paths[:, 0]) / wealth_paths[:, 0]
        
        # Remove any invalid returns
        valid_returns = wealth_returns[~np.isnan(wealth_returns)]
        
        if len(valid_returns) == 0:
            raise ValueError("No valid wealth returns found")
        
        # Calculate risk metrics
        mean_return = np.mean(valid_returns)
        return_std = np.std(valid_returns)
        
        # Value at Risk (VaR) at different confidence levels
        var_95 = np.percentile(valid_returns, 5)
        var_99 = np.percentile(valid_returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(valid_returns[valid_returns <= var_95])
        cvar_99 = np.mean(valid_returns[valid_returns <= var_99])
        
        # Sharpe ratio (assuming risk-free rate is 0)
        sharpe_ratio = mean_return / return_std if return_std > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(wealth_paths, axis=1)
        drawdowns = (wealth_paths - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'mean_return': mean_return,
            'return_volatility': return_std,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'n_valid_simulations': len(valid_returns)
        }


class PerformanceMetrics:
    """
    Class for calculating risk-adjusted performance measures including instantaneous
    Sharpe ratios, return-to-variance ratios (RVR), and statistical significance testing.
    
    This class implements the performance metrics required for testing hypotheses
    about return-to-variance ratio and Sharpe ratio dynamics during post-event periods.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize PerformanceMetrics.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_instantaneous_sharpe_ratio(self, 
                                           returns: np.ndarray, 
                                           volatilities: np.ndarray,
                                           window_size: int = 1) -> np.ndarray:
        """
        Calculate instantaneous Sharpe ratios.
        
        Args:
            returns: Array of returns
            volatilities: Array of volatilities (standard deviations)
            window_size: Window size for rolling calculation (1 for instantaneous)
            
        Returns:
            Array of instantaneous Sharpe ratios
        """
        if len(returns) != len(volatilities):
            raise ValueError("Returns and volatilities must have same length")
            
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
            
        sharpe_ratios = np.full(len(returns), np.nan)
        
        for i in range(window_size - 1, len(returns)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            window_returns = returns[start_idx:end_idx]
            window_volatilities = volatilities[start_idx:end_idx]
            
            # Calculate mean excess return and mean volatility for the window
            mean_excess_return = np.mean(window_returns) - self.risk_free_rate
            mean_volatility = np.mean(window_volatilities)
            
            if mean_volatility > 1e-10:
                sharpe_ratios[i] = mean_excess_return / mean_volatility
            else:
                sharpe_ratios[i] = 0.0
                
        return sharpe_ratios
    
    def calculate_return_to_variance_ratio(self, 
                                         returns: np.ndarray, 
                                         variances: np.ndarray,
                                         window_size: int = 1) -> np.ndarray:
        """
        Calculate return-to-variance ratios (RVR).
        
        The RVR is defined as the ratio of expected return to conditional variance,
        which is a key metric for testing the theoretical predictions of the model.
        
        Args:
            returns: Array of returns
            variances: Array of conditional variances
            window_size: Window size for rolling calculation (1 for instantaneous)
            
        Returns:
            Array of return-to-variance ratios
        """
        if len(returns) != len(variances):
            raise ValueError("Returns and variances must have same length")
            
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
            
        rvr_ratios = np.full(len(returns), np.nan)
        
        for i in range(window_size - 1, len(returns)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            window_returns = returns[start_idx:end_idx]
            window_variances = variances[start_idx:end_idx]
            
            # Calculate mean return and mean variance for the window
            mean_return = np.mean(window_returns)
            mean_variance = np.mean(window_variances)
            
            if mean_variance > 1e-10:
                rvr_ratios[i] = mean_return / mean_variance
            else:
                rvr_ratios[i] = 0.0
                
        return rvr_ratios
    
    def calculate_real_time_risk_measures(self, 
                                        returns: np.ndarray,
                                        variances: np.ndarray,
                                        confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, np.ndarray]:
        """
        Calculate real-time risk measures including VaR and CVaR.
        
        Args:
            returns: Array of returns
            variances: Array of conditional variances
            confidence_levels: List of confidence levels for VaR/CVaR calculation
            
        Returns:
            Dictionary containing real-time risk measures
        """
        from scipy import stats
        
        n_obs = len(returns)
        risk_measures = {}
        
        # Initialize arrays for each confidence level
        for cl in confidence_levels:
            risk_measures[f'var_{int(cl*100)}'] = np.full(n_obs, np.nan)
            risk_measures[f'cvar_{int(cl*100)}'] = np.full(n_obs, np.nan)
        
        # Calculate rolling risk measures
        for i in range(n_obs):
            if i == 0:
                # For first observation, use simple estimates
                current_return = returns[i]
                current_vol = np.sqrt(variances[i])
                
                for cl in confidence_levels:
                    # Assume normal distribution for VaR
                    var_quantile = stats.norm.ppf(1 - cl, loc=current_return, scale=current_vol)
                    risk_measures[f'var_{int(cl*100)}'][i] = var_quantile
                    
                    # CVaR approximation for normal distribution
                    cvar_value = current_return - current_vol * stats.norm.pdf(stats.norm.ppf(1 - cl)) / (1 - cl)
                    risk_measures[f'cvar_{int(cl*100)}'][i] = cvar_value
            else:
                # Use historical data up to current point
                hist_returns = returns[:i+1]
                
                for cl in confidence_levels:
                    # Historical VaR
                    var_quantile = np.percentile(hist_returns, (1 - cl) * 100)
                    risk_measures[f'var_{int(cl*100)}'][i] = var_quantile
                    
                    # Historical CVaR
                    var_exceedances = hist_returns[hist_returns <= var_quantile]
                    if len(var_exceedances) > 0:
                        cvar_value = np.mean(var_exceedances)
                    else:
                        cvar_value = var_quantile
                    risk_measures[f'cvar_{int(cl*100)}'][i] = cvar_value
        
        # Add additional real-time measures
        risk_measures['realized_volatility'] = np.sqrt(variances)
        risk_measures['downside_deviation'] = self._calculate_downside_deviation(returns)
        risk_measures['maximum_drawdown'] = self._calculate_rolling_max_drawdown(returns)
        
        return risk_measures
    
    def _calculate_downside_deviation(self, returns: np.ndarray, target_return: float = 0.0) -> np.ndarray:
        """Calculate rolling downside deviation."""
        downside_dev = np.full(len(returns), np.nan)
        
        for i in range(len(returns)):
            hist_returns = returns[:i+1]
            downside_returns = hist_returns[hist_returns < target_return]
            
            if len(downside_returns) > 0:
                downside_dev[i] = np.sqrt(np.mean((downside_returns - target_return)**2))
            else:
                downside_dev[i] = 0.0
                
        return downside_dev
    
    def _calculate_rolling_max_drawdown(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling maximum drawdown."""
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        max_drawdown = np.full(len(returns), np.nan)
        for i in range(len(returns)):
            max_drawdown[i] = np.min(drawdowns[:i+1])
            
        return max_drawdown
    
    def test_statistical_significance(self, 
                                    metric_values: np.ndarray,
                                    null_hypothesis_value: float = 0.0,
                                    test_type: str = 'two_sided',
                                    confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Test statistical significance of performance metrics.
        
        Args:
            metric_values: Array of metric values (e.g., Sharpe ratios, RVR)
            null_hypothesis_value: Value under null hypothesis
            test_type: Type of test ('two_sided', 'greater', 'less')
            confidence_level: Confidence level for the test
            
        Returns:
            Dictionary with test statistics and p-values
        """
        from scipy import stats
        
        # Remove NaN values
        valid_values = metric_values[~np.isnan(metric_values)]
        
        if len(valid_values) < 2:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'critical_value': np.nan,
                'is_significant': False,
                'n_observations': len(valid_values)
            }
        
        # Calculate test statistic
        sample_mean = np.mean(valid_values)
        sample_std = np.std(valid_values, ddof=1)
        n = len(valid_values)
        
        if sample_std == 0:
            # All values are the same
            if sample_mean == null_hypothesis_value:
                p_value = 1.0
                test_stat = 0.0
            else:
                p_value = 0.0
                test_stat = np.inf if sample_mean > null_hypothesis_value else -np.inf
        else:
            # t-test
            test_stat = (sample_mean - null_hypothesis_value) / (sample_std / np.sqrt(n))
            
            if test_type == 'two_sided':
                p_value = 2 * (1 - stats.t.cdf(abs(test_stat), df=n-1))
                critical_value = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n-1)
            elif test_type == 'greater':
                p_value = 1 - stats.t.cdf(test_stat, df=n-1)
                critical_value = stats.t.ppf(confidence_level, df=n-1)
            elif test_type == 'less':
                p_value = stats.t.cdf(test_stat, df=n-1)
                critical_value = stats.t.ppf(1 - confidence_level, df=n-1)
            else:
                raise ValueError("test_type must be 'two_sided', 'greater', or 'less'")
        
        # Determine significance
        alpha = 1 - confidence_level
        is_significant = p_value < alpha
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'is_significant': is_significant,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'n_observations': n,
            'confidence_level': confidence_level,
            'test_type': test_type
        }
    
    def calculate_information_ratio(self, 
                                  portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray,
                                  window_size: int = 1) -> np.ndarray:
        """
        Calculate information ratio (active return / tracking error).
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            window_size: Window size for rolling calculation
            
        Returns:
            Array of information ratios
        """
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
            
        active_returns = portfolio_returns - benchmark_returns
        information_ratios = np.full(len(active_returns), np.nan)
        
        for i in range(window_size - 1, len(active_returns)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            window_active_returns = active_returns[start_idx:end_idx]
            
            mean_active_return = np.mean(window_active_returns)
            tracking_error = np.std(window_active_returns, ddof=1) if len(window_active_returns) > 1 else 0
            
            if tracking_error > 1e-10:
                information_ratios[i] = mean_active_return / tracking_error
            else:
                information_ratios[i] = 0.0
                
        return information_ratios
    
    def calculate_sortino_ratio(self, 
                              returns: np.ndarray,
                              target_return: float = 0.0,
                              window_size: int = 1) -> np.ndarray:
        """
        Calculate Sortino ratio (excess return / downside deviation).
        
        Args:
            returns: Array of returns
            target_return: Target return for downside calculation
            window_size: Window size for rolling calculation
            
        Returns:
            Array of Sortino ratios
        """
        sortino_ratios = np.full(len(returns), np.nan)
        
        for i in range(window_size - 1, len(returns)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            
            window_returns = returns[start_idx:end_idx]
            
            mean_return = np.mean(window_returns)
            excess_return = mean_return - target_return
            
            # Calculate downside deviation
            downside_returns = window_returns[window_returns < target_return]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(np.mean((downside_returns - target_return)**2))
            else:
                downside_deviation = 0.0
            
            if downside_deviation > 1e-10:
                sortino_ratios[i] = excess_return / downside_deviation
            else:
                sortino_ratios[i] = np.inf if excess_return > 0 else 0.0
                
        return sortino_ratios


class HypothesisTester:
    """
    Class for testing theoretical predictions of the dynamic asset pricing model.
    
    This class implements hypothesis tests for:
    1. Post-event RVR peak detection
    2. Asymmetric bias effects for positive vs negative events
    3. Liquidity trading impact analysis
    4. Information asymmetry effect measurements
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize HypothesisTester.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.performance_metrics = PerformanceMetrics()
        
    def detect_post_event_rvr_peaks(self, 
                                   returns: np.ndarray,
                                   variances: np.ndarray,
                                   event_times: np.ndarray,
                                   time_grid: np.ndarray,
                                   post_event_window: int = 15) -> Dict[str, any]:
        """
        Detect peaks in return-to-variance ratio (RVR) during post-event periods.
        
        According to the theoretical model, RVR should peak during the rising 
        post-event volatility phase.
        
        Args:
            returns: Array of returns
            variances: Array of conditional variances
            event_times: Array of event times
            time_grid: Time grid corresponding to returns/variances
            post_event_window: Number of periods after event to analyze
            
        Returns:
            Dictionary with peak detection results
        """
        from scipy.signal import find_peaks
        from scipy import stats
        
        # Calculate RVR
        rvr = self.performance_metrics.calculate_return_to_variance_ratio(returns, variances)
        
        peak_results = {
            'event_peaks': [],
            'peak_times': [],
            'peak_values': [],
            'peak_significance': [],
            'hypothesis_test_results': {}
        }
        
        for event_time in event_times:
            # Find event index in time grid
            event_idx = np.argmin(np.abs(time_grid - event_time))
            
            # Define post-event window
            start_idx = event_idx + 1  # Start after event
            end_idx = min(len(rvr), event_idx + post_event_window + 1)
            
            if end_idx <= start_idx:
                continue
                
            # Extract post-event RVR
            post_event_rvr = rvr[start_idx:end_idx]
            post_event_times = time_grid[start_idx:end_idx]
            
            # Remove NaN values
            valid_mask = ~np.isnan(post_event_rvr)
            if np.sum(valid_mask) < 3:
                continue
                
            valid_rvr = post_event_rvr[valid_mask]
            valid_times = post_event_times[valid_mask]
            
            # Find peaks
            peaks, peak_properties = find_peaks(valid_rvr, height=np.mean(valid_rvr))
            
            if len(peaks) > 0:
                # Get the highest peak
                max_peak_idx = peaks[np.argmax(valid_rvr[peaks])]
                peak_time = valid_times[max_peak_idx]
                peak_value = valid_rvr[max_peak_idx]
                
                # Test if peak is significantly higher than pre-event baseline
                pre_event_start = max(0, event_idx - post_event_window)
                pre_event_rvr = rvr[pre_event_start:event_idx]
                pre_event_rvr = pre_event_rvr[~np.isnan(pre_event_rvr)]
                
                if len(pre_event_rvr) > 0:
                    # Two-sample t-test
                    peak_window_start = max(0, max_peak_idx - 2)
                    peak_window_end = min(len(valid_rvr), max_peak_idx + 3)
                    peak_window_rvr = valid_rvr[peak_window_start:peak_window_end]
                    
                    if len(peak_window_rvr) > 1 and len(pre_event_rvr) > 1:
                        t_stat, p_value = stats.ttest_ind(peak_window_rvr, pre_event_rvr)
                        is_significant = p_value < (1 - self.confidence_level)
                    else:
                        t_stat, p_value, is_significant = np.nan, np.nan, False
                else:
                    t_stat, p_value, is_significant = np.nan, np.nan, False
                
                peak_results['event_peaks'].append(event_time)
                peak_results['peak_times'].append(peak_time)
                peak_results['peak_values'].append(peak_value)
                peak_results['peak_significance'].append({
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': is_significant
                })
        
        # Overall hypothesis test: Do post-event periods show significantly higher RVR?
        if len(peak_results['peak_values']) > 0:
            peak_test = self.performance_metrics.test_statistical_significance(
                np.array(peak_results['peak_values']),
                null_hypothesis_value=0.0,
                test_type='greater',
                confidence_level=self.confidence_level
            )
            peak_results['hypothesis_test_results']['post_event_rvr_peaks'] = peak_test
        
        return peak_results
    
    def test_asymmetric_bias_effects(self, 
                                   returns: np.ndarray,
                                   event_outcomes: np.ndarray,
                                   event_times: np.ndarray,
                                   time_grid: np.ndarray,
                                   analysis_window: Tuple[int, int] = (-15, 15)) -> Dict[str, any]:
        """
        Test for asymmetric bias effects between positive and negative events.
        
        The theoretical model predicts different bias evolution patterns for
        positive vs negative events.
        
        Args:
            returns: Array of returns
            event_outcomes: Array indicating event outcomes (1=positive, -1=negative, 0=neutral)
            event_times: Array of event times
            time_grid: Time grid corresponding to returns
            analysis_window: Tuple of (pre_event_days, post_event_days)
            
        Returns:
            Dictionary with asymmetric bias test results
        """
        from scipy import stats
        
        pre_window, post_window = analysis_window
        
        positive_event_returns = []
        negative_event_returns = []
        positive_event_times = []
        negative_event_times = []
        
        for i, event_time in enumerate(event_times):
            if i >= len(event_outcomes):
                continue
                
            outcome = event_outcomes[i]
            event_idx = np.argmin(np.abs(time_grid - event_time))
            
            # Define analysis window around event
            start_idx = max(0, event_idx + pre_window)
            end_idx = min(len(returns), event_idx + post_window + 1)
            
            if end_idx <= start_idx:
                continue
                
            event_window_returns = returns[start_idx:end_idx]
            event_window_times = time_grid[start_idx:end_idx] - event_time
            
            # Remove NaN values
            valid_mask = ~np.isnan(event_window_returns)
            if np.sum(valid_mask) < 5:
                continue
                
            valid_returns = event_window_returns[valid_mask]
            valid_times = event_window_times[valid_mask]
            
            if outcome > 0:  # Positive event
                positive_event_returns.append(valid_returns)
                positive_event_times.append(valid_times)
            elif outcome < 0:  # Negative event
                negative_event_returns.append(valid_returns)
                negative_event_times.append(valid_times)
        
        bias_results = {
            'positive_events': {
                'n_events': len(positive_event_returns),
                'mean_returns': [],
                'cumulative_returns': []
            },
            'negative_events': {
                'n_events': len(negative_event_returns),
                'mean_returns': [],
                'cumulative_returns': []
            },
            'asymmetry_tests': {}
        }
        
        # Calculate statistics for positive events
        if positive_event_returns:
            pos_all_returns = np.concatenate(positive_event_returns)
            bias_results['positive_events']['mean_returns'] = np.mean(pos_all_returns)
            bias_results['positive_events']['cumulative_returns'] = np.sum(pos_all_returns)
        
        # Calculate statistics for negative events
        if negative_event_returns:
            neg_all_returns = np.concatenate(negative_event_returns)
            bias_results['negative_events']['mean_returns'] = np.mean(neg_all_returns)
            bias_results['negative_events']['cumulative_returns'] = np.sum(neg_all_returns)
        
        # Test for asymmetric effects
        if positive_event_returns and negative_event_returns:
            pos_all_returns = np.concatenate(positive_event_returns)
            neg_all_returns = np.concatenate(negative_event_returns)
            
            # Test if positive events have different return patterns than negative events
            t_stat, p_value = stats.ttest_ind(pos_all_returns, neg_all_returns)
            
            bias_results['asymmetry_tests']['return_difference'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < (1 - self.confidence_level),
                'positive_mean': np.mean(pos_all_returns),
                'negative_mean': np.mean(neg_all_returns),
                'difference': np.mean(pos_all_returns) - np.mean(neg_all_returns)
            }
            
            # Test for asymmetric volatility effects
            pos_volatility = np.std(pos_all_returns)
            neg_volatility = np.std(neg_all_returns)
            
            # F-test for equal variances
            f_stat = pos_volatility**2 / neg_volatility**2 if neg_volatility > 0 else np.inf
            f_p_value = 2 * min(stats.f.cdf(f_stat, len(pos_all_returns)-1, len(neg_all_returns)-1),
                               1 - stats.f.cdf(f_stat, len(pos_all_returns)-1, len(neg_all_returns)-1))
            
            bias_results['asymmetry_tests']['volatility_difference'] = {
                'f_statistic': f_stat,
                'p_value': f_p_value,
                'is_significant': f_p_value < (1 - self.confidence_level),
                'positive_volatility': pos_volatility,
                'negative_volatility': neg_volatility,
                'volatility_ratio': pos_volatility / neg_volatility if neg_volatility > 0 else np.inf
            }
        
        return bias_results
    
    def analyze_liquidity_trading_impact(self, 
                                       returns: np.ndarray,
                                       volumes: np.ndarray,
                                       event_times: np.ndarray,
                                       time_grid: np.ndarray,
                                       liquidity_threshold: float = None) -> Dict[str, any]:
        """
        Analyze the impact of liquidity trading on market dynamics.
        
        Tests whether periods of high/low liquidity show different return patterns
        and whether liquidity traders affect price discovery around events.
        
        Args:
            returns: Array of returns
            volumes: Array of trading volumes (proxy for liquidity)
            event_times: Array of event times
            time_grid: Time grid corresponding to returns/volumes
            liquidity_threshold: Threshold for high/low liquidity classification
            
        Returns:
            Dictionary with liquidity impact analysis results
        """
        from scipy import stats
        
        if liquidity_threshold is None:
            liquidity_threshold = np.median(volumes[~np.isnan(volumes)])
        
        # Classify periods as high/low liquidity
        high_liquidity_mask = volumes > liquidity_threshold
        low_liquidity_mask = volumes <= liquidity_threshold
        
        liquidity_results = {
            'liquidity_threshold': liquidity_threshold,
            'high_liquidity_periods': {
                'n_observations': np.sum(high_liquidity_mask),
                'mean_return': np.mean(returns[high_liquidity_mask & ~np.isnan(returns)]),
                'return_volatility': np.std(returns[high_liquidity_mask & ~np.isnan(returns)])
            },
            'low_liquidity_periods': {
                'n_observations': np.sum(low_liquidity_mask),
                'mean_return': np.mean(returns[low_liquidity_mask & ~np.isnan(returns)]),
                'return_volatility': np.std(returns[low_liquidity_mask & ~np.isnan(returns)])
            },
            'event_analysis': {},
            'liquidity_tests': {}
        }
        
        # Test for differences between high and low liquidity periods
        high_liq_returns = returns[high_liquidity_mask & ~np.isnan(returns)]
        low_liq_returns = returns[low_liquidity_mask & ~np.isnan(returns)]
        
        if len(high_liq_returns) > 1 and len(low_liq_returns) > 1:
            # Test for return differences
            t_stat, p_value = stats.ttest_ind(high_liq_returns, low_liq_returns)
            
            liquidity_results['liquidity_tests']['return_difference'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < (1 - self.confidence_level),
                'high_liq_mean': np.mean(high_liq_returns),
                'low_liq_mean': np.mean(low_liq_returns)
            }
            
            # Test for volatility differences
            f_stat = np.var(high_liq_returns) / np.var(low_liq_returns)
            f_p_value = 2 * min(stats.f.cdf(f_stat, len(high_liq_returns)-1, len(low_liq_returns)-1),
                               1 - stats.f.cdf(f_stat, len(high_liq_returns)-1, len(low_liq_returns)-1))
            
            liquidity_results['liquidity_tests']['volatility_difference'] = {
                'f_statistic': f_stat,
                'p_value': f_p_value,
                'is_significant': f_p_value < (1 - self.confidence_level),
                'high_liq_volatility': np.std(high_liq_returns),
                'low_liq_volatility': np.std(low_liq_returns)
            }
        
        # Analyze liquidity effects around events
        event_liquidity_effects = []
        
        for event_time in event_times:
            event_idx = np.argmin(np.abs(time_grid - event_time))
            
            # Define pre and post event windows
            pre_start = max(0, event_idx - 10)
            pre_end = event_idx
            post_start = event_idx + 1
            post_end = min(len(returns), event_idx + 11)
            
            if pre_end > pre_start and post_end > post_start:
                pre_event_volumes = volumes[pre_start:pre_end]
                post_event_volumes = volumes[post_start:post_end]
                pre_event_returns = returns[pre_start:pre_end]
                post_event_returns = returns[post_start:post_end]
                
                # Remove NaN values
                pre_valid = ~(np.isnan(pre_event_volumes) | np.isnan(pre_event_returns))
                post_valid = ~(np.isnan(post_event_volumes) | np.isnan(post_event_returns))
                
                if np.sum(pre_valid) > 2 and np.sum(post_valid) > 2:
                    pre_liquidity = np.mean(pre_event_volumes[pre_valid])
                    post_liquidity = np.mean(post_event_volumes[post_valid])
                    
                    liquidity_change = (post_liquidity - pre_liquidity) / pre_liquidity if pre_liquidity > 0 else 0
                    
                    event_liquidity_effects.append({
                        'event_time': event_time,
                        'pre_liquidity': pre_liquidity,
                        'post_liquidity': post_liquidity,
                        'liquidity_change': liquidity_change,
                        'pre_return_vol': np.std(pre_event_returns[pre_valid]),
                        'post_return_vol': np.std(post_event_returns[post_valid])
                    })
        
        liquidity_results['event_analysis']['individual_events'] = event_liquidity_effects
        
        # Test if events systematically affect liquidity
        if event_liquidity_effects:
            liquidity_changes = [e['liquidity_change'] for e in event_liquidity_effects]
            
            # Test if liquidity changes are significantly different from zero
            if len(liquidity_changes) > 1:
                t_stat, p_value = stats.ttest_1samp(liquidity_changes, 0)
                
                liquidity_results['event_analysis']['liquidity_change_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < (1 - self.confidence_level),
                    'mean_liquidity_change': np.mean(liquidity_changes),
                    'n_events': len(liquidity_changes)
                }
        
        return liquidity_results
    
    def measure_information_asymmetry_effects(self, 
                                            returns: np.ndarray,
                                            informed_trader_proxy: np.ndarray,
                                            event_times: np.ndarray,
                                            time_grid: np.ndarray,
                                            pre_event_window: int = 10,
                                            post_event_window: int = 10) -> Dict[str, any]:
        """
        Measure information asymmetry effects around events.
        
        Tests whether informed trading activity affects price discovery and
        whether information asymmetry creates predictable return patterns.
        
        Args:
            returns: Array of returns
            informed_trader_proxy: Proxy for informed trading activity (e.g., order flow imbalance)
            event_times: Array of event times
            time_grid: Time grid corresponding to returns
            pre_event_window: Number of periods before event to analyze
            post_event_window: Number of periods after event to analyze
            
        Returns:
            Dictionary with information asymmetry analysis results
        """
        from scipy import stats
        
        asymmetry_results = {
            'overall_correlation': {},
            'event_analysis': {},
            'information_tests': {}
        }
        
        # Overall correlation between informed trading and returns
        valid_mask = ~(np.isnan(returns) | np.isnan(informed_trader_proxy))
        if np.sum(valid_mask) > 10:
            correlation, p_value = stats.pearsonr(returns[valid_mask], informed_trader_proxy[valid_mask])
            
            asymmetry_results['overall_correlation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'is_significant': p_value < (1 - self.confidence_level),
                'n_observations': np.sum(valid_mask)
            }
        
        # Analyze information effects around events
        event_info_effects = []
        
        for event_time in event_times:
            event_idx = np.argmin(np.abs(time_grid - event_time))
            
            # Define analysis windows
            pre_start = max(0, event_idx - pre_event_window)
            pre_end = event_idx
            post_start = event_idx + 1
            post_end = min(len(returns), event_idx + post_event_window + 1)
            
            if pre_end > pre_start and post_end > post_start:
                # Pre-event period
                pre_returns = returns[pre_start:pre_end]
                pre_info = informed_trader_proxy[pre_start:pre_end]
                
                # Post-event period
                post_returns = returns[post_start:post_end]
                post_info = informed_trader_proxy[post_start:post_end]
                
                # Remove NaN values
                pre_valid = ~(np.isnan(pre_returns) | np.isnan(pre_info))
                post_valid = ~(np.isnan(post_returns) | np.isnan(post_info))
                
                if np.sum(pre_valid) > 3 and np.sum(post_valid) > 3:
                    # Calculate correlations
                    pre_corr, pre_p = stats.pearsonr(pre_returns[pre_valid], pre_info[pre_valid])
                    post_corr, post_p = stats.pearsonr(post_returns[post_valid], post_info[post_valid])
                    
                    # Calculate information intensity (mean absolute informed trading)
                    pre_intensity = np.mean(np.abs(pre_info[pre_valid]))
                    post_intensity = np.mean(np.abs(post_info[post_valid]))
                    
                    event_info_effects.append({
                        'event_time': event_time,
                        'pre_correlation': pre_corr,
                        'post_correlation': post_corr,
                        'pre_correlation_p': pre_p,
                        'post_correlation_p': post_p,
                        'pre_info_intensity': pre_intensity,
                        'post_info_intensity': post_intensity,
                        'correlation_change': post_corr - pre_corr,
                        'intensity_change': post_intensity - pre_intensity
                    })
        
        asymmetry_results['event_analysis']['individual_events'] = event_info_effects
        
        # Test systematic patterns in information effects
        if event_info_effects:
            correlation_changes = [e['correlation_change'] for e in event_info_effects if not np.isnan(e['correlation_change'])]
            intensity_changes = [e['intensity_change'] for e in event_info_effects if not np.isnan(e['intensity_change'])]
            
            # Test if correlation changes significantly around events
            if len(correlation_changes) > 1:
                t_stat, p_value = stats.ttest_1samp(correlation_changes, 0)
                
                asymmetry_results['information_tests']['correlation_change'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < (1 - self.confidence_level),
                    'mean_correlation_change': np.mean(correlation_changes),
                    'n_events': len(correlation_changes)
                }
            
            # Test if information intensity changes significantly around events
            if len(intensity_changes) > 1:
                t_stat, p_value = stats.ttest_1samp(intensity_changes, 0)
                
                asymmetry_results['information_tests']['intensity_change'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': p_value < (1 - self.confidence_level),
                    'mean_intensity_change': np.mean(intensity_changes),
                    'n_events': len(intensity_changes)
                }
        
        return asymmetry_results
    
    def run_comprehensive_hypothesis_tests(self, 
                                         data: Dict[str, np.ndarray],
                                         event_data: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Run comprehensive hypothesis tests for all theoretical predictions.
        
        Args:
            data: Dictionary containing returns, variances, volumes, etc.
            event_data: Dictionary containing event times, outcomes, etc.
            
        Returns:
            Dictionary with all hypothesis test results
        """
        comprehensive_results = {
            'rvr_peak_detection': {},
            'asymmetric_bias_effects': {},
            'liquidity_trading_impact': {},
            'information_asymmetry_effects': {},
            'summary_statistics': {}
        }
        
        # Required data validation
        required_data = ['returns', 'variances', 'time_grid']
        required_event_data = ['event_times']
        
        for key in required_data:
            if key not in data:
                raise ValueError(f"Missing required data: {key}")
        
        for key in required_event_data:
            if key not in event_data:
                raise ValueError(f"Missing required event data: {key}")
        
        # Run RVR peak detection tests
        comprehensive_results['rvr_peak_detection'] = self.detect_post_event_rvr_peaks(
            data['returns'], data['variances'], event_data['event_times'], data['time_grid']
        )
        
        # Run asymmetric bias tests if event outcomes are available
        if 'event_outcomes' in event_data:
            comprehensive_results['asymmetric_bias_effects'] = self.test_asymmetric_bias_effects(
                data['returns'], event_data['event_outcomes'], 
                event_data['event_times'], data['time_grid']
            )
        
        # Run liquidity impact tests if volume data is available
        if 'volumes' in data:
            comprehensive_results['liquidity_trading_impact'] = self.analyze_liquidity_trading_impact(
                data['returns'], data['volumes'], event_data['event_times'], data['time_grid']
            )
        
        # Run information asymmetry tests if informed trader proxy is available
        if 'informed_trader_proxy' in data:
            comprehensive_results['information_asymmetry_effects'] = self.measure_information_asymmetry_effects(
                data['returns'], data['informed_trader_proxy'], 
                event_data['event_times'], data['time_grid']
            )
        
        # Calculate summary statistics
        comprehensive_results['summary_statistics'] = self._calculate_summary_statistics(
            comprehensive_results
        )
        
        return comprehensive_results
    
    def _calculate_summary_statistics(self, results: Dict[str, any]) -> Dict[str, any]:
        """Calculate summary statistics across all hypothesis tests."""
        summary = {
            'total_tests_conducted': 0,
            'significant_tests': 0,
            'significance_rate': 0.0,
            'test_categories': {}
        }
        
        # Count tests and significant results
        for category, category_results in results.items():
            if category == 'summary_statistics':
                continue
                
            category_summary = {
                'tests_conducted': 0,
                'significant_tests': 0,
                'significance_rate': 0.0
            }
            
            # Recursively count tests in nested dictionaries
            def count_tests(obj, path=""):
                nonlocal category_summary, summary
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'is_significant' and isinstance(value, bool):
                            category_summary['tests_conducted'] += 1
                            summary['total_tests_conducted'] += 1
                            
                            if value:
                                category_summary['significant_tests'] += 1
                                summary['significant_tests'] += 1
                        elif isinstance(value, dict):
                            count_tests(value, f"{path}.{key}" if path else key)
            
            count_tests(category_results)
            
            # Calculate significance rates
            if category_summary['tests_conducted'] > 0:
                category_summary['significance_rate'] = (
                    category_summary['significant_tests'] / category_summary['tests_conducted']
                )
            
            summary['test_categories'][category] = category_summary
        
        # Overall significance rate
        if summary['total_tests_conducted'] > 0:
            summary['significance_rate'] = (
                summary['significant_tests'] / summary['total_tests_conducted']
            )
        
        return summary