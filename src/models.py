import numpy as np
import polars as pl # Use Polars for type hints if applicable, but core logic is NumPy/Sklearn
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings

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