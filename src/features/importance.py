"""
Feature Importance and Selection Module

This module implements feature importance methods based on
"Advances in Financial Machine Learning" by Marcos López de Prado.

Key methods implemented:
- Mean Decrease Impurity (MDI)
- Mean Decrease Accuracy (MDA)
- Single Feature Importance (SFI)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FeatureImportanceResults:
    """Container for feature importance analysis results."""

    mdi_importance: Optional[pd.Series] = None
    mda_importance: Optional[pd.Series] = None
    sfi_importance: Optional[pd.Series] = None
    clustered_importance: Optional[pd.Series] = None
    feature_ranking: Optional[pd.DataFrame] = None
    importance_plots: Optional[Dict[str, Any]] = None


class FeatureImportance:
    """
    Implementation of feature importance methods from AFML.

    This class provides various methods for assessing feature importance
    in financial machine learning contexts, with emphasis on avoiding
    common pitfalls like data leakage and overfitting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize feature importance analyzer.

        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum depth of trees
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def calculate_all_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> FeatureImportanceResults:
        """
        Calculate all feature importance measures.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weights: Optional sample weights
            cv_folds: Number of cross-validation folds for MDA

        Returns:
            FeatureImportanceResults with all importance measures
        """
        results = FeatureImportanceResults()

        # Align data and remove NaN values
        X_clean, y_clean, weights_clean = self._prepare_data(X, y, sample_weights)

        if len(X_clean) < 10:
            warnings.warn("Insufficient data for feature importance analysis")
            return results

        # 1. Mean Decrease Impurity (MDI)
        print("Calculating MDI importance...")
        results.mdi_importance = self.calculate_mdi_importance(
            X_clean, y_clean, weights_clean
        )

        # 2. Mean Decrease Accuracy (MDA)
        print("Calculating MDA importance...")
        results.mda_importance = self.calculate_mda_importance(
            X_clean, y_clean, weights_clean, cv_folds
        )

        # 3. Single Feature Importance (SFI)
        print("Calculating SFI importance...")
        results.sfi_importance = self.calculate_sfi_importance(
            X_clean, y_clean, weights_clean, cv_folds
        )

        # 4. Create comprehensive ranking
        results.feature_ranking = self._create_feature_ranking(results)

        return results

    def _prepare_data(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Prepare and clean data for analysis."""
        # Align all data
        if sample_weights is not None:
            common_index = X.index.intersection(y.index).intersection(
                sample_weights.index
            )
            weights_clean = sample_weights.loc[common_index]
        else:
            common_index = X.index.intersection(y.index)
            weights_clean = None

        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Remove rows with any NaN values
        valid_mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())

        X_clean = X_aligned.loc[valid_mask]
        y_clean = y_aligned.loc[valid_mask]

        if weights_clean is not None:
            weights_clean = weights_clean.loc[valid_mask]

        return X_clean, y_clean, weights_clean

    def calculate_mdi_importance(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate Mean Decrease Impurity (MDI) feature importance.

        MDI measures the average decrease in node impurity weighted by the probability
        of reaching that node. This is the standard feature importance in sklearn.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weights: Optional sample weights

        Returns:
            Series with MDI importance scores
        """
        # Determine if classification or regression
        if self._is_classification(y):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

        # Fit model
        if sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)

        # Get feature importance
        importance = pd.Series(model.feature_importances_, index=X.columns, name="MDI")

        return importance.sort_values(ascending=False)

    def calculate_mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> pd.Series:
        """
        Calculate Mean Decrease Accuracy (MDA) feature importance.

        MDA measures the decrease in accuracy when a feature is randomly shuffled,
        thus breaking the relationship between the feature and the target.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weights: Optional sample weights
            cv_folds: Number of cross-validation folds

        Returns:
            Series with MDA importance scores
        """
        # Get baseline accuracy
        baseline_score = self._get_baseline_score(X, y, sample_weights, cv_folds)

        # Calculate importance for each feature
        importance_scores = {}

        for feature in X.columns:
            # Create copy of data with shuffled feature
            X_shuffled = X.copy()
            np.random.seed(self.random_state)
            X_shuffled[feature] = np.random.permutation(X_shuffled[feature].values)

            # Get score with shuffled feature
            shuffled_score = self._get_baseline_score(
                X_shuffled, y, sample_weights, cv_folds
            )

            # Importance is the decrease in accuracy
            importance_scores[feature] = baseline_score - shuffled_score

        importance = pd.Series(importance_scores, name="MDA")

        return importance.sort_values(ascending=False)

    def calculate_sfi_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> pd.Series:
        """
        Calculate Single Feature Importance (SFI).

        SFI measures the performance of a model using only one feature at a time.
        This helps identify features that are individually predictive.

        Args:
            X: Feature matrix
            y: Target variable
            sample_weights: Optional sample weights
            cv_folds: Number of cross-validation folds

        Returns:
            Series with SFI importance scores
        """
        importance_scores = {}

        for feature in X.columns:
            # Use only single feature
            X_single = X[[feature]]

            # Get score using only this feature
            score = self._get_baseline_score(X_single, y, sample_weights, cv_folds)
            importance_scores[feature] = score

        importance = pd.Series(importance_scores, name="SFI")

        return importance.sort_values(ascending=False)

    def _get_baseline_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        cv_folds: int = 5,
    ) -> float:
        """Get baseline score using cross-validation."""
        # Determine if classification or regression
        if self._is_classification(y):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            scoring = "accuracy"
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            scoring = "r2"

        # Perform cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        if sample_weights is not None:
            # Manual cross-validation with sample weights
            scores = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                weights_train = (
                    sample_weights.iloc[train_idx]
                    if sample_weights is not None
                    else None
                )

                model.fit(X_train, y_train, sample_weight=weights_train)

                if self._is_classification(y):
                    score = accuracy_score(y_test, model.predict(X_test))
                else:
                    score = model.score(X_test, y_test)

                scores.append(score)

            return np.mean(scores)
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return np.mean(scores)

    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if target is for classification or regression."""
        # Simple heuristic: if less than 10 unique values, treat as classification
        return y.nunique() <= 10 or y.dtype == "object" or y.dtype.name == "category"

    def _create_feature_ranking(
        self, results: FeatureImportanceResults
    ) -> pd.DataFrame:
        """Create comprehensive feature ranking combining all methods."""
        ranking_data = {}

        if results.mdi_importance is not None:
            ranking_data["MDI"] = results.mdi_importance
            ranking_data["MDI_rank"] = results.mdi_importance.rank(ascending=False)

        if results.mda_importance is not None:
            ranking_data["MDA"] = results.mda_importance
            ranking_data["MDA_rank"] = results.mda_importance.rank(ascending=False)

        if results.sfi_importance is not None:
            ranking_data["SFI"] = results.sfi_importance
            ranking_data["SFI_rank"] = results.sfi_importance.rank(ascending=False)

        if not ranking_data:
            return pd.DataFrame()

        ranking_df = pd.DataFrame(ranking_data)

        # Calculate average rank
        rank_columns = [col for col in ranking_df.columns if col.endswith("_rank")]
        if rank_columns:
            ranking_df["avg_rank"] = ranking_df[rank_columns].mean(axis=1)
            ranking_df = ranking_df.sort_values("avg_rank")

        return ranking_df

    def plot_feature_importance(
        self,
        results: FeatureImportanceResults,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Dict[str, Any]:
        """
        Create visualization plots for feature importance.

        Args:
            results: Feature importance results
            top_n: Number of top features to show
            figsize: Figure size for plots

        Returns:
            Dictionary with plot objects
        """
        plots = {}

        # Create subplots
        n_methods = sum(
            [
                results.mdi_importance is not None,
                results.mda_importance is not None,
                results.sfi_importance is not None,
            ]
        )

        if n_methods == 0:
            return plots

        fig, axes = plt.subplots(n_methods, 1, figsize=figsize)
        if n_methods == 1:
            axes = [axes]

        plot_idx = 0

        # MDI plot
        if results.mdi_importance is not None:
            top_mdi = results.mdi_importance.head(top_n)
            axes[plot_idx].barh(range(len(top_mdi)), top_mdi.values)
            axes[plot_idx].set_yticks(range(len(top_mdi)))
            axes[plot_idx].set_yticklabels(top_mdi.index)
            axes[plot_idx].set_xlabel("MDI Importance")
            axes[plot_idx].set_title("Mean Decrease Impurity (MDI)")
            axes[plot_idx].invert_yaxis()
            plot_idx += 1

        # MDA plot
        if results.mda_importance is not None:
            top_mda = results.mda_importance.head(top_n)
            axes[plot_idx].barh(range(len(top_mda)), top_mda.values)
            axes[plot_idx].set_yticks(range(len(top_mda)))
            axes[plot_idx].set_yticklabels(top_mda.index)
            axes[plot_idx].set_xlabel("MDA Importance")
            axes[plot_idx].set_title("Mean Decrease Accuracy (MDA)")
            axes[plot_idx].invert_yaxis()
            plot_idx += 1

        # SFI plot
        if results.sfi_importance is not None:
            top_sfi = results.sfi_importance.head(top_n)
            axes[plot_idx].barh(range(len(top_sfi)), top_sfi.values)
            axes[plot_idx].set_yticks(range(len(top_sfi)))
            axes[plot_idx].set_yticklabels(top_sfi.index)
            axes[plot_idx].set_xlabel("SFI Score")
            axes[plot_idx].set_title("Single Feature Importance (SFI)")
            axes[plot_idx].invert_yaxis()

        plt.tight_layout()
        plots["importance_comparison"] = fig

        # Correlation heatmap of importance methods
        if results.feature_ranking is not None and len(results.feature_ranking) > 1:
            importance_cols = [
                col
                for col in results.feature_ranking.columns
                if not col.endswith("_rank") and col != "avg_rank"
            ]

            if len(importance_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                corr_matrix = results.feature_ranking[importance_cols].corr()
                sns.heatmap(
                    corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax_corr
                )
                ax_corr.set_title("Correlation Between Importance Methods")
                plots["method_correlation"] = fig_corr

        return plots

    def select_features_by_importance(
        self,
        results: FeatureImportanceResults,
        method: str = "avg_rank",
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Select features based on importance ranking.

        Args:
            results: Feature importance results
            method: Selection method ('mdi', 'mda', 'sfi', 'avg_rank')
            n_features: Number of top features to select
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names
        """
        if results.feature_ranking is None or len(results.feature_ranking) == 0:
            return []

        if method == "avg_rank" and "avg_rank" in results.feature_ranking.columns:
            # Select by average rank (lower is better)
            ranking = results.feature_ranking.sort_values("avg_rank")

            if n_features is not None:
                return ranking.head(n_features).index.tolist()
            elif threshold is not None:
                return ranking[ranking["avg_rank"] <= threshold].index.tolist()
            else:
                return ranking.index.tolist()

        else:
            # Select by specific importance method
            importance_series = None

            if method == "mdi" and results.mdi_importance is not None:
                importance_series = results.mdi_importance
            elif method == "mda" and results.mda_importance is not None:
                importance_series = results.mda_importance
            elif method == "sfi" and results.sfi_importance is not None:
                importance_series = results.sfi_importance

            if importance_series is None:
                return []

            if n_features is not None:
                return importance_series.head(n_features).index.tolist()
            elif threshold is not None:
                return importance_series[importance_series >= threshold].index.tolist()
            else:
                return importance_series.index.tolist()

    def get_feature_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 10,
        sample_fraction: float = 0.8,
    ) -> pd.DataFrame:
        """
        Assess feature importance stability across different data samples.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of trials with different samples
            sample_fraction: Fraction of data to use in each trial

        Returns:
            DataFrame with stability metrics for each feature
        """
        importance_trials = []

        for trial in range(n_trials):
            # Sample data
            sample_size = int(len(X) * sample_fraction)
            np.random.seed(self.random_state + trial)
            sample_idx = np.random.choice(len(X), size=sample_size, replace=False)

            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]

            # Calculate importance for this sample
            mdi_importance = self.calculate_mdi_importance(X_sample, y_sample)
            importance_trials.append(mdi_importance)

        # Create stability metrics
        importance_df = pd.concat(importance_trials, axis=1)

        stability_metrics = pd.DataFrame(index=X.columns)
        stability_metrics["mean_importance"] = importance_df.mean(axis=1)
        stability_metrics["std_importance"] = importance_df.std(axis=1)
        stability_metrics["cv_importance"] = (
            stability_metrics["std_importance"] / stability_metrics["mean_importance"]
        )
        stability_metrics["stability_score"] = 1 / (
            1 + stability_metrics["cv_importance"]
        )

        return stability_metrics.sort_values("stability_score", ascending=False)
