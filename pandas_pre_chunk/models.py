import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings

class TimeSeriesRidge(Ridge):
    """
    Ridge regression with temporal smoothing penalty.

    The model minimizes:
    ||y - Xβ||² + α||β||² + λ₂||Dβ||²

    Where D is a differencing matrix penalizing changes between *consecutive* coefficients
    assuming features in X are ordered meaningfully (e.g., by time lag or window size).
    Note: The effectiveness depends heavily on the order of features in X.
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
        If self.feature_order is set, X is reordered before applying the penalty.
        """
        # Ensure X is a DataFrame if feature_order is specified
        if self.feature_order is not None and not isinstance(X, pd.DataFrame):
             raise TypeError("X must be a pandas DataFrame when feature_order is specified.")

        # Reorder X according to feature_order if provided
        if self.feature_order is not None:
             missing_features = set(self.feature_order) - set(X.columns)
             if missing_features:
                 raise ValueError(f"Features specified in feature_order are missing from X: {missing_features}")
             extra_features = set(X.columns) - set(self.feature_order)
             if extra_features:
                 print(f"Warning: X contains features not in feature_order: {extra_features}. They will be placed at the end.")
                 # Maintain all columns, but order according to feature_order first
                 ordered_cols = self.feature_order + list(extra_features)
                 X_ordered = X[ordered_cols].copy()
             else:
                 X_ordered = X[self.feature_order].copy()
             # print(f"Fitting TimeSeriesRidge with feature order: {X_ordered.columns.tolist()}")
        else:
             X_ordered = X.copy()
             # if isinstance(X_ordered, pd.DataFrame):
             #     print(f"Fitting TimeSeriesRidge with default feature order: {X_ordered.columns.tolist()}")


        # Convert inputs to numpy arrays and ensure they're float64
        X_np = np.asarray(X_ordered, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X_np.shape

        # Basic checks for NaN/inf
        if np.isnan(X_np).any() or np.isinf(X_np).any():
             raise ValueError("NaN or Inf values detected in feature matrix X before fitting.")
        if np.isnan(y_np).any() or np.isinf(y_np).any():
             raise ValueError("NaN or Inf values detected in target vector y before fitting.")


        # Get differencing matrix
        D = self._get_differencing_matrix(n_features)

        # If no differencing is possible (e.g., 1 feature), fall back to standard Ridge
        if D.shape[0] == 0 or self.lambda2 == 0:
            # print("Applying standard Ridge regression (lambda2=0 or n_features<=1).")
            # Fit using the original alpha
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))
            ridge_model.fit(X_np, y_np, sample_weight)
            self.coef_ = ridge_model.coef_
            self.intercept_ = ridge_model.intercept_
            # Store feature names if possible
            if isinstance(X, pd.DataFrame): self.feature_names_in_ = X.columns.tolist()
            else: self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]
            return self


        # --- Augmentation Method for Combined Penalty ---
        # Create augmented data for custom regularization with sqrt(lambda2) * D
        sqrt_lambda2_D = np.sqrt(self.lambda2) * D
        X_augmented = np.vstack([X_np, sqrt_lambda2_D])
        y_augmented = np.concatenate([y_np, np.zeros(D.shape[0])])

        # Fit standard Ridge on augmented data with the original alpha (self.alpha)
        ridge_solver = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.get_params(deep=False))

        # Adjust sample weights if provided
        if sample_weight is not None:
             augmented_weights = np.concatenate([sample_weight, np.ones(D.shape[0])])
             ridge_solver.fit(X_augmented, y_augmented, sample_weight=augmented_weights)
        else:
             ridge_solver.fit(X_augmented, y_augmented)

        # Store the fitted coefficients and intercept
        self.coef_ = ridge_solver.coef_
        self.intercept_ = ridge_solver.intercept_

        # If features were reordered, store coefficients in the original order
        if self.feature_order is not None:
             coef_dict = dict(zip(X_ordered.columns, self.coef_))
             original_order_coef = [coef_dict.get(col, 0) for col in X.columns] # Use original X columns
             self.coef_ = np.array(original_order_coef)
             # Store feature names in the order of the coefficients
             self.feature_names_in_ = X.columns.tolist()
        elif isinstance(X, pd.DataFrame):
             self.feature_names_in_ = X.columns.tolist()
        else:
             self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]

        return self

    def predict(self, X):
         # If features were reordered during fit, ensure prediction uses the same internal order
         if self.feature_order is not None:
              if not isinstance(X, pd.DataFrame):
                   raise TypeError("X must be a pandas DataFrame for prediction if feature_order was used in fit.")
              # Ensure prediction X has the columns expected based on fitting order
              if not hasattr(self, 'feature_names_in_'):
                   raise RuntimeError("Model not fitted or feature names not stored.")
              internal_feature_order = self.feature_names_in_ # Use the order saved during fit
              missing_cols = set(internal_feature_order) - set(X.columns)
              if missing_cols:
                  raise ValueError(f"Prediction data missing columns used during fit: {missing_cols}")
              X_ordered = X[internal_feature_order]
              return super().predict(X_ordered)
         else:
              # Standard prediction if no reordering occurred
              return super().predict(X)


class XGBoostDecileModel:
    """
    XGBoostDecile Ensemble Model that combines an XGBoost model with a decile-based model.
    The decile models are TimeSeriesRidge regressions fitted per decile.

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
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1 # Use all available cores
            }
        else:
            self.xgb_params = xgb_params

        # Initialize XGBoost model
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)

        # Initialize decile models (TimeSeriesRidge) and boundaries
        self.decile_models = [None] * n_deciles
        self.decile_boundaries = None
        self.feature_names_in_ = None # To store feature names from training

    def _create_decile_assignments(self, X):
        """Assign observations to deciles based on the momentum feature value using training data boundaries."""
        if self.decile_boundaries is None:
             raise RuntimeError("Decile boundaries have not been calculated. Call fit first.")
        if self.momentum_feature not in X.columns:
             raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in input data.")

        momentum_values = X[self.momentum_feature].values

        # Handle potential NaNs in momentum values before assigning deciles
        nan_mask = np.isnan(momentum_values)
        if np.any(nan_mask):
            # Decide how to handle NaNs, e.g., assign to a specific decile (like median) or raise error
            # Here, we'll assign them to decile 0 as a simple strategy
            print(f"Warning: {np.sum(nan_mask)} NaN values found in momentum feature '{self.momentum_feature}'. Assigning them to decile 0.")
            momentum_values = np.nan_to_num(momentum_values, nan=self.decile_boundaries[0]) # Replace NaN with value below first boundary


        # Assign each observation to a decile using pre-calculated boundaries
        decile_assignments = np.searchsorted(self.decile_boundaries, momentum_values, side='right') - 1
        decile_assignments = np.clip(decile_assignments, 0, self.n_deciles - 1)

        return decile_assignments

    def fit(self, X, y):
        """
        Fit both the XGBoost model and the decile-based TimeSeriesRidge models.
        """
        # print("Fitting XGBoostDecileModel...")
        if not isinstance(X, pd.DataFrame):
             X = pd.DataFrame(X) # Ensure X is DataFrame for column access
        self.feature_names_in_ = X.columns.tolist() # Store feature names

        # --- Fit XGBoost Component ---
        # print(f"Fitting XGBoost component (Weight: {self.weight})...")
        try:
            # Use eval_set for early stopping
            eval_set = [(X, y)]
            self.xgb_model.fit(X, y, eval_set=eval_set,
                               early_stopping_rounds=self.xgb_params.get('early_stopping_rounds', 10),
                               verbose=False)
            # print("XGBoost fitting complete (used early stopping).")
        except TypeError as e:
             if "use_label_encoder" in str(e): # Handle older XGBoost version param
                print("Adjusting for older XGBoost version (use_label_encoder=False)")
                self.xgb_model = xgb.XGBRegressor(**{**self.xgb_params, 'use_label_encoder': False})
                self.xgb_model.fit(X, y, eval_metric='rmse', eval_set=[(X, y)],
                                   early_stopping_rounds=self.xgb_params.get('early_stopping_rounds', 10),
                                   verbose=False)
                # print("XGBoost fitting complete (used early stopping).")
             else: raise e
        except Exception as e: # Fallback if early stopping causes issues
            print(f"Warning: Error during XGBoost fit with early stopping: {e}. Retrying without.")
            try:
                 # Remove early stopping param if present
                 fallback_params = self.xgb_params.copy()
                 fallback_params.pop('early_stopping_rounds', None)
                 self.xgb_model = xgb.XGBRegressor(**fallback_params)
                 self.xgb_model.fit(X, y)
                 # print("XGBoost fitting complete (without early stopping).")
            except Exception as e2: print(f"Error during XGBoost fallback fit: {e2}"); raise e2


        # --- Fit Decile Components ---
        if self.weight < 1.0: # Only fit decile models if they have non-zero weight
             # print(f"Fitting Decile TimeSeriesRidge components (Weight: {1 - self.weight})...")
             if self.momentum_feature not in X.columns:
                 raise ValueError(f"Momentum feature '{self.momentum_feature}' not found in training data X.")

             # Calculate decile boundaries based on the training data
             momentum_values = X[self.momentum_feature].values
             self.decile_boundaries = np.nanquantile(momentum_values, np.linspace(0, 1, self.n_deciles + 1)[1:-1])
             # print(f"Calculated {len(self.decile_boundaries)} decile boundaries based on '{self.momentum_feature}'.")

             # Assign training data points to deciles using these boundaries
             decile_assignments = self._create_decile_assignments(X)

             # Fit a separate TimeSeriesRidge model for each decile
             for d in range(self.n_deciles):
                 decile_mask = (decile_assignments == d)
                 X_decile = X[decile_mask]
                 y_decile = y[decile_mask]

                 # print(f"  Decile {d+1}/{self.n_deciles}: {len(X_decile)} samples.")
                 min_samples_required = max(5, X.shape[1] + 1) # Need slightly more samples than features
                 if len(X_decile) >= min_samples_required:
                     try:
                         # print(f"    Fitting TimeSeriesRidge for Decile {d+1}...")
                         decile_model = TimeSeriesRidge(
                             alpha=self.alpha,
                             lambda2=self.lambda_smooth,
                             feature_order=self.ts_ridge_feature_order # Pass feature order
                         )
                         # Ensure the decile data has the features in the expected order
                         if self.ts_ridge_feature_order:
                              X_decile_ordered = X_decile[self.ts_ridge_feature_order]
                         else:
                              X_decile_ordered = X_decile
                         decile_model.fit(X_decile_ordered, y_decile)
                         self.decile_models[d] = decile_model
                         # print(f"    Decile {d+1} model fitted.")
                     except Exception as e:
                         print(f"    Warning: Failed to fit model for Decile {d+1}. Reason: {e}")
                         self.decile_models[d] = None # Mark as failed
                 else:
                     # print(f"    Warning: Not enough samples ({len(X_decile)} < {min_samples_required}) for Decile {d+1}. Skipping.")
                     self.decile_models[d] = None
        # else: print("Skipping Decile model fitting as weight is 1.0 (XGBoost only).")

        # print("XGBoostDecileModel fitting complete.")
        return self

    def predict(self, X):
        """
        Generate predictions using the ensemble of XGBoost and decile-based models.
        """
        if self.feature_names_in_ is None:
             raise RuntimeError("Model has not been fitted yet. Call fit first.")
        if not isinstance(X, pd.DataFrame):
             X = pd.DataFrame(X, columns=self.feature_names_in_) # Ensure DataFrame with correct columns
        else:
             # Ensure prediction data has the same columns as training data
             missing_cols = set(self.feature_names_in_) - set(X.columns)
             extra_cols = set(X.columns) - set(self.feature_names_in_)
             if missing_cols:
                 raise ValueError(f"Missing columns in prediction data: {missing_cols}")
             if extra_cols:
                 # print(f"Warning: Extra columns found in prediction data: {extra_cols}. They will be ignored.")
                 X = X[self.feature_names_in_] # Reorder and select columns


        # --- Get XGBoost Predictions ---
        xgb_preds = self.xgb_model.predict(X)

        # --- Get Decile Predictions (if needed) ---
        if self.weight < 1.0:
             if self.decile_boundaries is None:
                 raise RuntimeError("Decile boundaries not set. Model needs fitting or boundaries are missing.")

             decile_preds = np.zeros_like(xgb_preds)
             decile_assignments = self._create_decile_assignments(X) # Assign test data to deciles

             for d in range(self.n_deciles):
                 decile_mask = (decile_assignments == d)
                 if np.any(decile_mask): # If any samples fall into this decile
                     X_decile_test = X[decile_mask]

                     # Use the fitted model for this decile
                     if self.decile_models[d] is not None:
                         try:
                             # Ensure prediction data for decile model has correct feature order if specified
                             if self.ts_ridge_feature_order:
                                  X_decile_test_ordered = X_decile_test[self.ts_ridge_feature_order]
                             else:
                                  X_decile_test_ordered = X_decile_test
                             decile_preds[decile_mask] = self.decile_models[d].predict(X_decile_test_ordered)
                         except Exception as e:
                             print(f"Warning: Error predicting with model for Decile {d+1}. Using XGBoost prediction as fallback. Error: {e}")
                             decile_preds[decile_mask] = xgb_preds[decile_mask] # Fallback
                     else:
                         # If no model was trained for this decile, use XGBoost prediction
                         decile_preds[decile_mask] = xgb_preds[decile_mask] # Fallback

             # Combine predictions with weight
             ensemble_preds = self.weight * xgb_preds + (1 - self.weight) * decile_preds
        else:
             # If weight is 1, only use XGBoost predictions
             ensemble_preds = xgb_preds

        return ensemble_preds