# --- START OF FILE models.py ---
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
    # ... (No changes from previous correct version) ...
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
    # ... (No changes from previous correct version) ...
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
        
        if alpha + beta >= 0.99999:
            if context=="final": warnings.warn(f"GARCH alpha+beta ({alpha+beta:.4f}) too high. Adjusting for stationarity.")
            current_sum = alpha + beta
            target_sum = 0.9998 
            if current_sum > 1e-7:
                alpha_new = (alpha / current_sum) * target_sum
                beta_new = (beta / current_sum) * target_sum
                # Ensure individual params remain non-negative after scaling
                alpha = max(0, alpha_new)
                beta = max(0, beta_new)
                # If sum is still off due to flooring at 0, adjust one (e.g. beta)
                if alpha + beta >= 0.99999 :
                    beta = target_sum - alpha if target_sum > alpha else 0.0
            else: 
                alpha = 0.05 
                beta = 0.90
            valid = False
        return omega, alpha, beta, valid


    def _neg_log_likelihood(self, params, returns_centered):
        omega, alpha, beta = params
        if not (omega > 1e-9 and alpha >= 0 and beta >= 0 and alpha + beta < 0.99999):
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
            # warnings.warn(self.fit_message + " Using initial parameters.")
            self._use_initial_params_for_history(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7: 
            self.fit_message = "GARCH: Return series has very low variance."
            # warnings.warn(self.fit_message + " Using simplified variance.")
            self._handle_low_variance_series(returns_np, std_dev) 
            return self

        clip_threshold = 7 * std_dev 
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean
        if np.std(returns_centered) < 1e-7: 
            self.fit_message = "GARCH: Demeaned returns have very low variance."
            # warnings.warn(self.fit_message + " Using simplified variance.")
            self._handle_low_variance_series(returns_np, np.std(returns_centered))
            return self

        # Use checked initial parameters
        omega_i, alpha_i, beta_i, _ = self._check_parameters(self.omega_init, self.alpha_init, self.beta_init, context="initial_guess")
        initial_params = [omega_i, alpha_i, beta_i]
        
        bounds = [(1e-8, 0.1), (0, 0.998), (0, 0.998)] # omega, alpha, beta bounds (alpha/beta can be 0)

        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9, 'eps': 1e-8} # Added eps for L-BFGS-B
        
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
                    if params_valid and (alpha_fit + beta_fit < 0.9999): 
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
            # warnings.warn(f"GARCH optimization failed ({msg_detail}). Using robust initial parameters.")
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
        if stationarity_val >= 0.99999:
            if context=="final": warnings.warn(f"GJR sum condition ({stationarity_val:.4f}) too high. Adjusting for stationarity.")
            target_sum_overall = 0.9998
            if alpha + beta >= target_sum_overall : 
                gamma = max(0, gamma * 0.01) # Drastically reduce gamma if alpha+beta is the issue
                current_sum_alpha_beta = alpha + beta
                required_sum_alpha_beta_for_gjr = target_sum_overall - 0.5 * gamma
                if current_sum_alpha_beta > required_sum_alpha_beta_for_gjr and current_sum_alpha_beta > 1e-7:
                    alpha_new = (alpha / current_sum_alpha_beta) * required_sum_alpha_beta_for_gjr
                    beta_new = (beta / current_sum_alpha_beta) * required_sum_alpha_beta_for_gjr
                    alpha = max(0, alpha_new); beta = max(0, beta_new)
                    if alpha + beta + 0.5 * gamma >= 0.99999: # Final attempt to fix
                        beta = target_sum_overall - alpha - 0.5*gamma if (target_sum_overall - 0.5*gamma) > alpha else 0.0
                elif current_sum_alpha_beta <= 1e-7 : 
                     alpha = 0.03; beta = 0.80; gamma = 0.02 # Reset all to typical GJR stationary
            else: 
                gamma = max(0, (target_sum_overall - (alpha + beta)) * 2 * 0.99 ) 
            valid = False
        return omega, alpha, beta, gamma, valid

    def _neg_log_likelihood(self, params, returns_centered): 
        omega, alpha, beta, gamma = params
        if not (omega > 1e-9 and alpha >= 0 and beta >= 0 and gamma >= 0 and \
                alpha + beta + 0.5 * gamma < 0.99999):
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
            # warnings.warn(self.fit_message + " Using initial parameters.")
            self._use_initial_params_for_history_gjr(returns_np)
            return self

        std_dev = np.std(returns_np)
        if std_dev < 1e-7:
            self.fit_message = "GJR: Return series has very low variance."
            # warnings.warn(self.fit_message + " Using simplified variance.")
            self._handle_low_variance_series_gjr(returns_np, std_dev)
            return self
            
        clip_threshold = 7 * std_dev 
        returns_np = np.clip(returns_np, -clip_threshold, clip_threshold)
        self.mean = np.mean(returns_np)
        returns_centered = returns_np - self.mean
        if np.std(returns_centered) < 1e-7:
            self.fit_message = "GJR: Demeaned returns have very low variance."
            # warnings.warn(self.fit_message + " Using simplified variance.")
            self._handle_low_variance_series_gjr(returns_np, np.std(returns_centered))
            return self

        omega_i, alpha_i, beta_i, gamma_i, _ = self._check_gjr_parameters(self.omega_init, self.alpha_init, self.beta_init, self.gamma_init, context="initial_guess")
        initial_params = [omega_i, alpha_i, beta_i, gamma_i]
        
        bounds = [(1e-8, 0.1), (0, 0.998), (0, 0.998), (0, 0.998)] 

        optimizer_options = {'maxiter': max_iter, 'disp': False, 'ftol': 1e-9, 'eps': 1e-8}
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
                    if params_valid and (a + b + 0.5 * g < 0.9999):
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
            # warnings.warn(f"GJR-GARCH optimization failed ({msg_detail}). Using robust initial parameters.")
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
    def __init__(self, baseline_model: Union[GARCHModel, GJRGARCHModel],
                 k1: float = 1.5, k2: float = 2.0, 
                 delta_t1: float = 5.0, delta_t2: float = 3.0, delta_t3: float = 10.0, 
                 delta: int = 5):
        self.baseline_model = baseline_model
        self.k1 = k1
        self.k2 = k2
        self.delta_t1 = max(1e-3, delta_t1) # Ensure positive delta
        self.delta_t2 = max(1e-3, delta_t2)
        self.delta_t3 = max(1e-3, delta_t3)
        self.delta = delta
        
        if k1 <= 1: raise ValueError("k1 must be greater than 1")
        if k2 <= 1: raise ValueError("k2 must be greater than 1")
    
    def phi1(self, t: float, t_event: float) -> float: # t can be float
        return (self.k1 - 1) * np.exp(-((t - t_event)**2) / (2 * self.delta_t1**2))
    
    def phi2(self, t: float, t_event: float) -> float:
        time_diff = t - t_event
        # Ensure delta_t2 is positive to prevent division by zero or issues with exp
        if self.delta_t2 <= 1e-6 : return (self.k2 - 1) if time_diff > 0 else 0.0 
        # Prevent overflow if time_diff / self.delta_t2 is very large negative
        exp_arg = -time_diff / self.delta_t2
        if exp_arg < -700: # np.exp(-709) is approx 0
            return self.k2 -1 
        return (self.k2 - 1) * (1 - np.exp(exp_arg))
    
    def phi3(self, t: float, t_event: float) -> float:
        time_diff = t - (t_event + self.delta)
        if self.delta_t3 <= 1e-6: return 0.0 
        exp_arg = -time_diff / self.delta_t3
        if exp_arg < -700: return 0.0
        return (self.k2 - 1) * np.exp(exp_arg)
    
    def calculate_volatility(self, t: float, t_event: float, sigma_e0: float) -> float:
        if sigma_e0 < 1e-8: sigma_e0 = 1e-8 
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
            days_to_event_np = np.array(days_to_event, dtype=float) 
        else:
            days_to_event_np = days_to_event.astype(float)

        if baseline_conditional_vol_series is not None:
            if len(baseline_conditional_vol_series) != len(days_to_event_np):
                 raise ValueError(
                     f"Length of baseline_conditional_vol_series ({len(baseline_conditional_vol_series)}) "
                     f"must match days_to_event ({len(days_to_event_np)}). "
                     "Alignment error in calling function."
                 )
            sigma_e0_series = np.maximum(baseline_conditional_vol_series, 1e-8)
        else: 
            if not self.baseline_model.is_fitted:
                raise RuntimeError("Baseline model must be fitted or baseline_conditional_vol_series provided")
            
            bm = self.baseline_model
            if isinstance(bm, GJRGARCHModel):
                denominator = (1 - bm.alpha - bm.beta - 0.5 * bm.gamma)
            else: 
                denominator = (1 - bm.alpha - bm.beta)
            
            uncond_var = 1e-7 # Default
            if denominator > 1e-7: 
                uncond_var = bm.omega / denominator
            elif bm.variance_history is not None and len(bm.variance_history) > 0 and bm.variance_history[-1] > 1e-8 : 
                uncond_var = bm.variance_history[-1] 
            
            sigma_e0_val = np.sqrt(max(uncond_var, 1e-8)) 
            sigma_e0_series = np.full_like(days_to_event_np, sigma_e0_val, dtype=float)

        volatility_series = np.array([self.calculate_volatility(t_rel, 0.0, sigma_e0_series[i]) 
                                      for i, t_rel in enumerate(days_to_event_np)], dtype=float)
        
        return volatility_series
