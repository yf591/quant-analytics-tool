"""
Feature Quality Validation Module

This module implements comprehensive feature quality validation methods
including stationarity tests, multicollinearity detection, and automated
feature engineering quality metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class FeatureQualityResults:
    """Container for feature quality validation results."""

    stationarity_results: Optional[Dict[str, Any]] = None
    multicollinearity_results: Optional[Dict[str, Any]] = None
    distribution_results: Optional[Dict[str, Any]] = None
    completeness_results: Optional[Dict[str, Any]] = None
    stability_results: Optional[Dict[str, Any]] = None
    outlier_results: Optional[Dict[str, Any]] = None
    quality_score: Optional[pd.Series] = None
    recommendations: Optional[List[str]] = None


class FeatureQualityValidator:
    """
    Comprehensive feature quality validation system.

    This class implements various tests and metrics to assess the quality
    of engineered features for financial machine learning applications.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize feature quality validator.

        Args:
            significance_level: Statistical significance level for tests
        """
        self.significance_level = significance_level

    def validate_all_features(self, features: pd.DataFrame) -> FeatureQualityResults:
        """
        Perform comprehensive feature quality validation.

        Args:
            features: Feature matrix to validate

        Returns:
            FeatureQualityResults with all validation results
        """
        results = FeatureQualityResults()

        print("Validating feature quality...")

        # 1. Stationarity tests
        print("  - Testing stationarity...")
        results.stationarity_results = self.test_stationarity(features)

        # 2. Multicollinearity detection
        print("  - Checking multicollinearity...")
        results.multicollinearity_results = self.detect_multicollinearity(features)

        # 3. Distribution analysis
        print("  - Analyzing distributions...")
        results.distribution_results = self.analyze_distributions(features)

        # 4. Data completeness
        print("  - Checking data completeness...")
        results.completeness_results = self.check_completeness(features)

        # 5. Feature stability
        print("  - Assessing feature stability...")
        results.stability_results = self.assess_stability(features)

        # 6. Outlier detection
        print("  - Detecting outliers...")
        results.outlier_results = self.detect_outliers(features)

        # 7. Overall quality score
        print("  - Computing quality scores...")
        results.quality_score = self.compute_quality_score(results, features)

        # 8. Generate recommendations
        results.recommendations = self.generate_recommendations(results)

        return results

    def test_stationarity(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Test stationarity of features using multiple methods.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with stationarity test results
        """
        stationarity_results = {
            "adf_test": {},
            "kpss_test": {},
            "variance_ratio_test": {},
            "summary": {},
        }

        for column in features.columns:
            series = features[column].dropna()

            if len(series) < 20:
                stationarity_results["summary"][column] = "insufficient_data"
                continue

            # 1. Augmented Dickey-Fuller test
            adf_result = self._adf_test(series)
            stationarity_results["adf_test"][column] = adf_result

            # 2. KPSS test
            kpss_result = self._kpss_test(series)
            stationarity_results["kpss_test"][column] = kpss_result

            # 3. Variance ratio test
            vr_result = self._variance_ratio_test(series)
            stationarity_results["variance_ratio_test"][column] = vr_result

            # Summary decision
            stationary_votes = sum(
                [
                    adf_result["is_stationary"],
                    kpss_result["is_stationary"],
                    vr_result["is_stationary"],
                ]
            )

            if stationary_votes >= 2:
                stationarity_results["summary"][column] = "stationary"
            elif stationary_votes == 1:
                stationarity_results["summary"][column] = "questionable"
            else:
                stationarity_results["summary"][column] = "non_stationary"

        return stationarity_results

    def _adf_test(self, series: pd.Series) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test for stationarity."""
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(series, autolag="AIC")

            return {
                "test_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < self.significance_level,
            }
        except:
            # Fallback simple test
            return self._simple_stationarity_test(series)

    def _kpss_test(self, series: pd.Series) -> Dict[str, Any]:
        """KPSS test for stationarity."""
        try:
            from statsmodels.tsa.stattools import kpss

            result = kpss(series, regression="c")

            return {
                "test_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[3],
                "is_stationary": result[1]
                > self.significance_level,  # KPSS: null is stationary
            }
        except:
            # Fallback simple test
            return self._simple_stationarity_test(series)

    def _variance_ratio_test(self, series: pd.Series) -> Dict[str, Any]:
        """Variance ratio test for random walk hypothesis."""
        try:
            # Simple variance ratio test implementation
            returns = series.pct_change().dropna()

            if len(returns) < 10:
                return {"is_stationary": True, "p_value": 1.0}

            # Calculate variance ratio for different periods
            periods = [2, 4, 8, 16]
            variance_ratios = []

            for period in periods:
                if len(returns) >= period * 2:
                    # Variance of period-returns
                    period_returns = returns.rolling(period).sum().dropna()
                    var_period = period_returns.var()

                    # Variance of single-period returns
                    var_single = returns.var()

                    # Variance ratio
                    if var_single > 0:
                        vr = var_period / (period * var_single)
                        variance_ratios.append(abs(vr - 1.0))

            if variance_ratios:
                avg_deviation = np.mean(variance_ratios)
                # If average deviation from 1.0 is small, likely stationary
                is_stationary = avg_deviation < 0.2
                p_value = min(avg_deviation * 5, 1.0)  # Approximate p-value
            else:
                is_stationary = True
                p_value = 1.0

            return {
                "variance_ratios": variance_ratios,
                "avg_deviation": avg_deviation if variance_ratios else 0,
                "is_stationary": is_stationary,
                "p_value": p_value,
            }
        except:
            return {"is_stationary": True, "p_value": 1.0}

    def _simple_stationarity_test(self, series: pd.Series) -> Dict[str, Any]:
        """Simple stationarity test based on rolling statistics."""
        try:
            window = min(len(series) // 4, 50)
            if window < 5:
                return {"is_stationary": True, "p_value": 1.0}

            # Rolling mean and std
            rolling_mean = series.rolling(window).mean()
            rolling_std = series.rolling(window).std()

            # Test if rolling statistics are stable
            mean_stability = (
                rolling_mean.std() / rolling_mean.mean()
                if rolling_mean.mean() != 0
                else 0
            )
            std_stability = (
                rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 0
            )

            # Consider stationary if both are stable (low coefficient of variation)
            is_stationary = (mean_stability < 0.1) and (std_stability < 0.1)
            p_value = (mean_stability + std_stability) / 2

            return {
                "test_statistic": mean_stability + std_stability,
                "p_value": min(p_value, 1.0),
                "is_stationary": is_stationary,
            }
        except:
            return {"is_stationary": True, "p_value": 1.0}

    def detect_multicollinearity(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect multicollinearity using correlation matrix and VIF.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with multicollinearity detection results
        """
        results = {
            "correlation_matrix": None,
            "high_correlations": [],
            "vif_scores": {},
            "condition_number": None,
            "problematic_features": [],
        }

        # Clean data
        features_clean = features.select_dtypes(include=[np.number]).dropna()

        if len(features_clean.columns) < 2:
            return results

        # 1. Correlation matrix
        corr_matrix = features_clean.corr()
        results["correlation_matrix"] = corr_matrix

        # 2. Find high correlations
        threshold = 0.95
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        results["high_correlations"] = high_corr_pairs

        # 3. Variance Inflation Factor (VIF)
        results["vif_scores"] = self._calculate_vif(features_clean)

        # 4. Condition number
        try:
            # Standardize features first
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            features_std = scaler.fit_transform(features_clean.fillna(0))

            # Calculate condition number
            _, s, _ = np.linalg.svd(features_std)
            condition_number = s[0] / s[-1] if s[-1] != 0 else np.inf
            results["condition_number"] = condition_number
        except:
            results["condition_number"] = None

        # 5. Identify problematic features
        problematic = set()

        # From high correlations
        for pair in high_corr_pairs:
            problematic.add(pair["feature2"])  # Remove second feature in pair

        # From high VIF
        for feature, vif in results["vif_scores"].items():
            if vif > 10:
                problematic.add(feature)

        results["problematic_features"] = list(problematic)

        return results

    def _calculate_vif(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate Variance Inflation Factor for each feature."""
        vif_scores = {}

        try:
            from sklearn.linear_model import LinearRegression

            for i, feature in enumerate(features.columns):
                # Prepare data
                y = features[feature]
                X = features.drop(columns=[feature])

                # Remove any remaining NaN values
                combined = pd.concat([X, y], axis=1).dropna()
                if len(combined) < 10:
                    vif_scores[feature] = 1.0
                    continue

                X_clean = combined.iloc[:, :-1]
                y_clean = combined.iloc[:, -1]

                if X_clean.shape[1] == 0:
                    vif_scores[feature] = 1.0
                    continue

                # Fit regression
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                r_squared = model.score(X_clean, y_clean)

                # Calculate VIF
                if r_squared < 0.9999:  # Avoid division by very small number
                    vif = 1 / (1 - r_squared)
                else:
                    vif = 1000  # Very high VIF for perfect correlation

                vif_scores[feature] = vif

        except Exception as e:
            # Fallback: set all VIF to 1 (no multicollinearity)
            for feature in features.columns:
                vif_scores[feature] = 1.0

        return vif_scores

    def analyze_distributions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze feature distributions for normality and other properties.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with distribution analysis results
        """
        results = {
            "normality_tests": {},
            "skewness": {},
            "kurtosis": {},
            "outlier_percentages": {},
            "zero_percentages": {},
        }

        for column in features.columns:
            series = features[column].dropna()

            if len(series) < 10:
                continue

            # Normality tests
            results["normality_tests"][column] = self._test_normality(series)

            # Skewness and kurtosis
            results["skewness"][column] = series.skew()
            results["kurtosis"][column] = series.kurtosis()

            # Outlier percentage (using IQR method)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            results["outlier_percentages"][column] = outliers.mean()

            # Zero percentage
            results["zero_percentages"][column] = (series == 0).mean()

        return results

    def _test_normality(self, series: pd.Series) -> Dict[str, Any]:
        """Test normality using multiple methods."""
        normality_results = {}

        if len(series) < 8:
            return {"is_normal": False, "method": "insufficient_data"}

        try:
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(series)
            normality_results["jarque_bera"] = {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "is_normal": jb_pvalue > self.significance_level,
            }
        except:
            normality_results["jarque_bera"] = {"is_normal": False}

        try:
            # Shapiro-Wilk test (for smaller samples)
            if len(series) <= 5000:
                sw_stat, sw_pvalue = shapiro(series)
                normality_results["shapiro_wilk"] = {
                    "statistic": sw_stat,
                    "p_value": sw_pvalue,
                    "is_normal": sw_pvalue > self.significance_level,
                }
        except:
            pass

        # Overall normality decision
        normal_votes = []
        if "jarque_bera" in normality_results:
            normal_votes.append(normality_results["jarque_bera"]["is_normal"])
        if "shapiro_wilk" in normality_results:
            normal_votes.append(normality_results["shapiro_wilk"]["is_normal"])

        if normal_votes:
            normality_results["is_normal"] = sum(normal_votes) >= len(normal_votes) / 2
        else:
            normality_results["is_normal"] = False

        return normality_results

    def check_completeness(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness and missing value patterns.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with completeness analysis results
        """
        results = {
            "missing_percentages": {},
            "missing_patterns": {},
            "completeness_score": 0.0,
        }

        # Missing percentages per feature
        missing_pct = features.isnull().mean()
        results["missing_percentages"] = missing_pct.to_dict()

        # Overall completeness score
        results["completeness_score"] = 1 - missing_pct.mean()

        # Missing patterns
        missing_patterns = features.isnull().sum(axis=1).value_counts().sort_index()
        results["missing_patterns"] = missing_patterns.to_dict()

        return results

    def assess_stability(
        self, features: pd.DataFrame, window: int = 100
    ) -> Dict[str, Any]:
        """
        Assess feature stability over time.

        Args:
            features: Feature matrix
            window: Rolling window size for stability assessment

        Returns:
            Dictionary with stability analysis results
        """
        results = {
            "mean_stability": {},
            "variance_stability": {},
            "stability_scores": {},
        }

        for column in features.columns:
            series = features[column].dropna()

            if len(series) < window * 2:
                results["stability_scores"][column] = 1.0
                continue

            # Rolling statistics
            rolling_mean = series.rolling(window).mean()
            rolling_var = series.rolling(window).var()

            # Stability metrics
            mean_stability = (
                1 - (rolling_mean.std() / rolling_mean.mean())
                if rolling_mean.mean() != 0
                else 1.0
            )
            var_stability = (
                1 - (rolling_var.std() / rolling_var.mean())
                if rolling_var.mean() != 0
                else 1.0
            )

            # Overall stability score
            stability_score = (mean_stability + var_stability) / 2

            results["mean_stability"][column] = max(0, mean_stability)
            results["variance_stability"][column] = max(0, var_stability)
            results["stability_scores"][column] = max(0, stability_score)

        return results

    def detect_outliers(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with outlier detection results
        """
        results = {
            "iqr_outliers": {},
            "zscore_outliers": {},
            "isolation_forest_outliers": {},
            "outlier_summary": {},
        }

        for column in features.columns:
            series = features[column].dropna()

            if len(series) < 10:
                continue

            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = (series < lower_bound) | (series > upper_bound)
            results["iqr_outliers"][column] = iqr_outliers.mean()

            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            zscore_outliers = z_scores > 3
            results["zscore_outliers"][column] = zscore_outliers.mean()

            # Summary
            results["outlier_summary"][column] = {
                "iqr_percentage": iqr_outliers.mean(),
                "zscore_percentage": zscore_outliers.mean(),
                "recommended_action": (
                    "investigate"
                    if max(iqr_outliers.mean(), zscore_outliers.mean()) > 0.05
                    else "ok"
                ),
            }

        return results

    def compute_quality_score(
        self, results: FeatureQualityResults, features: pd.DataFrame
    ) -> pd.Series:
        """
        Compute overall quality score for each feature.

        Args:
            results: Feature quality validation results
            features: Original feature matrix

        Returns:
            Series with quality scores for each feature
        """
        quality_scores = pd.Series(index=features.columns, dtype=float)

        for feature in features.columns:
            score = 1.0

            # Completeness penalty
            if (
                results.completeness_results
                and feature in results.completeness_results["missing_percentages"]
            ):
                missing_pct = results.completeness_results["missing_percentages"][
                    feature
                ]
                score *= 1 - missing_pct

            # Stationarity bonus
            if (
                results.stationarity_results
                and feature in results.stationarity_results["summary"]
            ):
                stationarity = results.stationarity_results["summary"][feature]
                if stationarity == "stationary":
                    score *= 1.1
                elif stationarity == "non_stationary":
                    score *= 0.8

            # Multicollinearity penalty
            if (
                results.multicollinearity_results
                and feature in results.multicollinearity_results["vif_scores"]
            ):
                vif = results.multicollinearity_results["vif_scores"][feature]
                if vif > 10:
                    score *= 0.5
                elif vif > 5:
                    score *= 0.8

            # Stability bonus
            if (
                results.stability_results
                and feature in results.stability_results["stability_scores"]
            ):
                stability = results.stability_results["stability_scores"][feature]
                score *= 0.5 + 0.5 * stability

            # Outlier penalty
            if (
                results.outlier_results
                and feature in results.outlier_results["outlier_summary"]
            ):
                outlier_pct = results.outlier_results["outlier_summary"][feature][
                    "iqr_percentage"
                ]
                if outlier_pct > 0.1:
                    score *= 0.7
                elif outlier_pct > 0.05:
                    score *= 0.9

            quality_scores[feature] = max(0, min(1, score))

        return quality_scores.sort_values(ascending=False)

    def generate_recommendations(self, results: FeatureQualityResults) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Multicollinearity recommendations
        if (
            results.multicollinearity_results
            and results.multicollinearity_results["high_correlations"]
        ):
            n_high_corr = len(results.multicollinearity_results["high_correlations"])
            recommendations.append(
                f"Consider removing {n_high_corr} highly correlated feature pairs to reduce multicollinearity"
            )

        # Missing data recommendations
        if results.completeness_results:
            high_missing = {
                k: v
                for k, v in results.completeness_results["missing_percentages"].items()
                if v > 0.1
            }
            if high_missing:
                recommendations.append(
                    f"Features with >10% missing data: {list(high_missing.keys())}. Consider imputation or removal."
                )

        # Stationarity recommendations
        if results.stationarity_results:
            non_stationary = [
                k
                for k, v in results.stationarity_results["summary"].items()
                if v == "non_stationary"
            ]
            if non_stationary:
                recommendations.append(
                    f"Non-stationary features detected: {non_stationary}. Consider differencing or transformation."
                )

        # Outlier recommendations
        if results.outlier_results:
            high_outlier_features = []
            for feature, summary in results.outlier_results["outlier_summary"].items():
                if summary["iqr_percentage"] > 0.05:
                    high_outlier_features.append(feature)

            if high_outlier_features:
                recommendations.append(
                    f"Features with >5% outliers: {high_outlier_features}. Consider outlier treatment."
                )

        # Overall recommendations
        if not recommendations:
            recommendations.append(
                "Feature quality looks good overall. No major issues detected."
            )

        return recommendations

    def plot_quality_summary(
        self,
        results: FeatureQualityResults,
        features: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """Create comprehensive quality visualization."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. Quality scores
        if results.quality_score is not None:
            top_features = results.quality_score.head(20)
            axes[0, 0].barh(range(len(top_features)), top_features.values)
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features.index)
            axes[0, 0].set_xlabel("Quality Score")
            axes[0, 0].set_title("Top 20 Features by Quality Score")
            axes[0, 0].invert_yaxis()

        # 2. Missing data
        if results.completeness_results:
            missing_data = pd.Series(
                results.completeness_results["missing_percentages"]
            )
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            if len(missing_data) > 0:
                axes[0, 1].bar(range(len(missing_data)), missing_data.values)
                axes[0, 1].set_xticks(range(len(missing_data)))
                axes[0, 1].set_xticklabels(missing_data.index, rotation=45)
                axes[0, 1].set_ylabel("Missing Percentage")
                axes[0, 1].set_title("Features with Missing Data")

        # 3. Correlation heatmap (top correlated features)
        if (
            results.multicollinearity_results
            and results.multicollinearity_results["correlation_matrix"] is not None
        ):
            corr_matrix = results.multicollinearity_results["correlation_matrix"]
            # Show only features with high correlations
            high_corr_features = set()
            for pair in results.multicollinearity_results["high_correlations"]:
                high_corr_features.add(pair["feature1"])
                high_corr_features.add(pair["feature2"])

            if high_corr_features:
                subset_corr = corr_matrix.loc[
                    list(high_corr_features), list(high_corr_features)
                ]
                sns.heatmap(
                    subset_corr, annot=True, cmap="coolwarm", center=0, ax=axes[0, 2]
                )
                axes[0, 2].set_title("High Correlation Features")
            else:
                axes[0, 2].text(
                    0.5,
                    0.5,
                    "No high correlations detected",
                    ha="center",
                    va="center",
                    transform=axes[0, 2].transAxes,
                )
                axes[0, 2].set_title("Correlation Analysis")

        # 4. Stationarity summary
        if results.stationarity_results:
            stationarity_summary = pd.Series(results.stationarity_results["summary"])
            stationarity_counts = stationarity_summary.value_counts()
            axes[1, 0].pie(
                stationarity_counts.values,
                labels=stationarity_counts.index,
                autopct="%1.1f%%",
            )
            axes[1, 0].set_title("Stationarity Test Results")

        # 5. VIF scores
        if (
            results.multicollinearity_results
            and results.multicollinearity_results["vif_scores"]
        ):
            vif_scores = pd.Series(results.multicollinearity_results["vif_scores"])
            high_vif = vif_scores[vif_scores > 5].sort_values(ascending=False)
            if len(high_vif) > 0:
                axes[1, 1].bar(range(len(high_vif)), high_vif.values)
                axes[1, 1].set_xticks(range(len(high_vif)))
                axes[1, 1].set_xticklabels(high_vif.index, rotation=45)
                axes[1, 1].set_ylabel("VIF Score")
                axes[1, 1].set_title("Features with High VIF (>5)")
                axes[1, 1].axhline(y=10, color="r", linestyle="--", label="VIF=10")
                axes[1, 1].legend()
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No high VIF detected",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("VIF Analysis")

        # 6. Outlier summary
        if results.outlier_results:
            outlier_percentages = []
            feature_names = []
            for feature, summary in results.outlier_results["outlier_summary"].items():
                outlier_percentages.append(summary["iqr_percentage"])
                feature_names.append(feature)

            if outlier_percentages:
                outlier_series = pd.Series(outlier_percentages, index=feature_names)
                high_outliers = outlier_series[outlier_series > 0.01].sort_values(
                    ascending=False
                )
                if len(high_outliers) > 0:
                    axes[1, 2].bar(range(len(high_outliers)), high_outliers.values)
                    axes[1, 2].set_xticks(range(len(high_outliers)))
                    axes[1, 2].set_xticklabels(high_outliers.index, rotation=45)
                    axes[1, 2].set_ylabel("Outlier Percentage")
                    axes[1, 2].set_title("Features with Outliers (>1%)")
                else:
                    axes[1, 2].text(
                        0.5,
                        0.5,
                        "No significant outliers",
                        ha="center",
                        va="center",
                        transform=axes[1, 2].transAxes,
                    )
                    axes[1, 2].set_title("Outlier Analysis")

        plt.tight_layout()
        return fig
