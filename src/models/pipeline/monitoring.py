"""
Model Monitoring and Alerting

This module provides comprehensive monitoring capabilities for production ML models
including performance tracking, drift detection, and automated alerting.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import threading
import time
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from scipy import stats
import warnings


@dataclass
class Alert:
    """Model alert definition"""

    alert_id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    model_id: str
    deployment_id: str
    timestamp: datetime
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring"""

    # Performance monitoring
    performance_window_hours: int = 24
    performance_threshold: float = 0.05  # 5% degradation threshold

    # Data drift monitoring
    drift_detection_enabled: bool = True
    drift_threshold: float = 0.1  # Statistical significance threshold
    feature_drift_window: int = 1000  # Number of samples for drift detection

    # Prediction monitoring
    prediction_anomaly_threshold: float = 3.0  # Standard deviations
    confidence_threshold: float = 0.7  # Minimum confidence threshold

    # System health monitoring
    latency_threshold_ms: float = 1000.0  # Max prediction latency
    error_rate_threshold: float = 0.01  # Max error rate (1%)

    # Alerting
    alert_cooldown_minutes: int = 60  # Minutes between duplicate alerts
    critical_alert_immediate: bool = True

    # Monitoring frequency
    monitoring_interval_minutes: int = 5


class ModelMonitor:
    """
    Comprehensive model monitoring system for production ML models.

    Features:
    - Performance monitoring and degradation detection
    - Data drift detection using statistical tests
    - Prediction anomaly detection
    - System health monitoring (latency, errors)
    - Automated alerting with configurable thresholds
    - Historical trend analysis
    - Dashboard metrics for visualization
    """

    def __init__(
        self, monitoring_db_path: str = "monitoring.db", config: MonitoringConfig = None
    ):
        """
        Initialize model monitor.

        Args:
            monitoring_db_path: Path to monitoring database
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.monitoring_db_path = monitoring_db_path

        # Initialize database
        self._init_monitoring_db()

        # Monitoring state
        self.active_monitors = {}  # deployment_id -> monitor thread
        self.prediction_history = defaultdict(lambda: deque(maxlen=10000))
        self.feature_history = defaultdict(lambda: deque(maxlen=10000))
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.alert_cache = defaultdict(lambda: deque(maxlen=100))

        # Alert handlers
        self.alert_handlers = []

        # Thread safety
        self._lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        with sqlite3.connect(self.monitoring_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    deployment_id TEXT NOT NULL,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    features TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL,
                    latency_ms REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_deployment ON monitoring_metrics(deployment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON monitoring_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_deployment ON alerts(deployment_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_deployment ON prediction_logs(deployment_id)"
            )

    def start_monitoring(self, deployment_id: str, model_id: str):
        """
        Start monitoring for a deployment.

        Args:
            deployment_id: Deployment identifier
            model_id: Model identifier
        """
        if deployment_id in self.active_monitors:
            self.logger.warning(
                f"Monitoring already active for deployment {deployment_id}"
            )
            return

        # Create monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(deployment_id, model_id), daemon=True
        )

        self.active_monitors[deployment_id] = {
            "thread": monitor_thread,
            "model_id": model_id,
            "started_at": datetime.now(),
            "status": "starting",
        }

        monitor_thread.start()

        self.logger.info(f"Started monitoring for deployment {deployment_id}")

    def stop_monitoring(self, deployment_id: str):
        """
        Stop monitoring for a deployment.

        Args:
            deployment_id: Deployment identifier
        """
        if deployment_id not in self.active_monitors:
            self.logger.warning(f"No active monitoring for deployment {deployment_id}")
            return

        monitor_info = self.active_monitors[deployment_id]
        monitor_info["status"] = "stopping"

        # Wait for thread to finish (with timeout)
        if monitor_info["thread"].is_alive():
            monitor_info["thread"].join(timeout=10)

        del self.active_monitors[deployment_id]

        self.logger.info(f"Stopped monitoring for deployment {deployment_id}")

    def _monitoring_loop(self, deployment_id: str, model_id: str):
        """Main monitoring loop for a deployment"""
        self.active_monitors[deployment_id]["status"] = "active"

        self.logger.info(f"Monitoring loop started for deployment {deployment_id}")

        try:
            while (
                deployment_id in self.active_monitors
                and self.active_monitors[deployment_id]["status"] == "active"
            ):

                try:
                    # Run monitoring checks
                    self._run_monitoring_checks(deployment_id, model_id)

                    # Sleep until next check
                    time.sleep(self.config.monitoring_interval_minutes * 60)

                except Exception as e:
                    self.logger.error(
                        f"Error in monitoring loop for {deployment_id}: {str(e)}"
                    )
                    time.sleep(60)  # Wait 1 minute before retrying

        except Exception as e:
            self.logger.error(
                f"Fatal error in monitoring loop for {deployment_id}: {str(e)}"
            )

        finally:
            if deployment_id in self.active_monitors:
                self.active_monitors[deployment_id]["status"] = "stopped"

    def _run_monitoring_checks(self, deployment_id: str, model_id: str):
        """Run all monitoring checks for a deployment"""
        try:
            # Performance monitoring
            self._check_model_performance(deployment_id, model_id)

            # Data drift detection
            if self.config.drift_detection_enabled:
                self._check_data_drift(deployment_id, model_id)

            # Prediction anomaly detection
            self._check_prediction_anomalies(deployment_id, model_id)

            # System health monitoring
            self._check_system_health(deployment_id, model_id)

            # Update monitoring metrics
            self._record_monitoring_metrics(deployment_id, model_id)

        except Exception as e:
            self.logger.error(
                f"Error running monitoring checks for {deployment_id}: {str(e)}"
            )

    def _check_model_performance(self, deployment_id: str, model_id: str):
        """Check model performance degradation"""
        try:
            # Get recent performance metrics
            recent_metrics = self._get_recent_performance_metrics(
                deployment_id, model_id
            )

            if not recent_metrics:
                return

            # Get baseline performance
            baseline_metrics = self._get_baseline_performance(deployment_id, model_id)

            if not baseline_metrics:
                # Store current metrics as baseline if none exists
                self._store_baseline_performance(
                    deployment_id, model_id, recent_metrics
                )
                return

            # Calculate performance degradation
            degradation = self._calculate_performance_degradation(
                recent_metrics, baseline_metrics
            )

            # Check if degradation exceeds threshold
            if degradation > self.config.performance_threshold:
                self._create_alert(
                    alert_type="performance_degradation",
                    severity="high",
                    message=f"Model performance degraded by {degradation:.2%}",
                    model_id=model_id,
                    deployment_id=deployment_id,
                    metadata={
                        "degradation": degradation,
                        "threshold": self.config.performance_threshold,
                        "recent_metrics": recent_metrics,
                        "baseline_metrics": baseline_metrics,
                    },
                )

        except Exception as e:
            self.logger.error(f"Error checking model performance: {str(e)}")

    def _check_data_drift(self, deployment_id: str, model_id: str):
        """Check for data drift using statistical tests"""
        try:
            # Get recent feature data
            recent_features = self._get_recent_features(deployment_id, model_id)

            if len(recent_features) < self.config.feature_drift_window:
                return  # Not enough data for drift detection

            # Get baseline feature distribution
            baseline_features = self._get_baseline_features(deployment_id, model_id)

            if not baseline_features:
                # Store current features as baseline
                self._store_baseline_features(deployment_id, model_id, recent_features)
                return

            # Perform drift detection for each feature
            drift_detected = False
            drift_features = []

            recent_df = pd.DataFrame(recent_features)
            baseline_df = pd.DataFrame(baseline_features)

            for feature in recent_df.columns:
                if feature in baseline_df.columns:
                    # Perform Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(
                        baseline_df[feature].dropna(), recent_df[feature].dropna()
                    )

                    if p_value < self.config.drift_threshold:
                        drift_detected = True
                        drift_features.append(
                            {
                                "feature": feature,
                                "ks_statistic": statistic,
                                "p_value": p_value,
                            }
                        )

            if drift_detected:
                self._create_alert(
                    alert_type="data_drift",
                    severity="medium",
                    message=f"Data drift detected in {len(drift_features)} features",
                    model_id=model_id,
                    deployment_id=deployment_id,
                    metadata={
                        "drift_features": drift_features,
                        "threshold": self.config.drift_threshold,
                    },
                )

        except Exception as e:
            self.logger.error(f"Error checking data drift: {str(e)}")

    def _check_prediction_anomalies(self, deployment_id: str, model_id: str):
        """Check for prediction anomalies"""
        try:
            # Get recent predictions
            recent_predictions = self._get_recent_predictions(deployment_id, model_id)

            if len(recent_predictions) < 100:  # Need sufficient data
                return

            predictions = [p["prediction"] for p in recent_predictions]
            confidences = [
                p["confidence"]
                for p in recent_predictions
                if p["confidence"] is not None
            ]

            # Check for prediction outliers
            prediction_mean = np.mean(predictions)
            prediction_std = np.std(predictions)

            outliers = []
            for i, pred in enumerate(predictions):
                z_score = abs(pred - prediction_mean) / prediction_std
                if z_score > self.config.prediction_anomaly_threshold:
                    outliers.append(
                        {
                            "index": i,
                            "prediction": pred,
                            "z_score": z_score,
                            "timestamp": recent_predictions[i]["timestamp"],
                        }
                    )

            if outliers:
                self._create_alert(
                    alert_type="prediction_anomaly",
                    severity="medium",
                    message=f"Detected {len(outliers)} prediction outliers",
                    model_id=model_id,
                    deployment_id=deployment_id,
                    metadata={
                        "outliers": outliers[:10],  # Limit to first 10
                        "total_outliers": len(outliers),
                        "threshold": self.config.prediction_anomaly_threshold,
                    },
                )

            # Check confidence levels
            if confidences:
                low_confidence_count = sum(
                    1 for c in confidences if c < self.config.confidence_threshold
                )
                low_confidence_rate = low_confidence_count / len(confidences)

                if low_confidence_rate > 0.1:  # More than 10% low confidence
                    self._create_alert(
                        alert_type="low_confidence",
                        severity="low",
                        message=f"High rate of low-confidence predictions: {low_confidence_rate:.2%}",
                        model_id=model_id,
                        deployment_id=deployment_id,
                        metadata={
                            "low_confidence_rate": low_confidence_rate,
                            "threshold": self.config.confidence_threshold,
                        },
                    )

        except Exception as e:
            self.logger.error(f"Error checking prediction anomalies: {str(e)}")

    def _check_system_health(self, deployment_id: str, model_id: str):
        """Check system health metrics"""
        try:
            # Get recent prediction logs
            recent_logs = self._get_recent_prediction_logs(deployment_id, model_id)

            if not recent_logs:
                return

            # Check latency
            latencies = [
                log["latency_ms"]
                for log in recent_logs
                if log["latency_ms"] is not None
            ]

            if latencies:
                avg_latency = np.mean(latencies)
                max_latency = np.max(latencies)

                if avg_latency > self.config.latency_threshold_ms:
                    self._create_alert(
                        alert_type="high_latency",
                        severity="medium",
                        message=f"High average latency: {avg_latency:.2f}ms",
                        model_id=model_id,
                        deployment_id=deployment_id,
                        metadata={
                            "avg_latency": avg_latency,
                            "max_latency": max_latency,
                            "threshold": self.config.latency_threshold_ms,
                        },
                    )

            # Check error rate (would need error logging)
            # This is simplified - in production, you'd track actual errors
            total_predictions = len(recent_logs)
            error_count = 0  # Placeholder - implement actual error tracking

            if total_predictions > 0:
                error_rate = error_count / total_predictions

                if error_rate > self.config.error_rate_threshold:
                    self._create_alert(
                        alert_type="high_error_rate",
                        severity="high",
                        message=f"High error rate: {error_rate:.2%}",
                        model_id=model_id,
                        deployment_id=deployment_id,
                        metadata={
                            "error_rate": error_rate,
                            "error_count": error_count,
                            "total_predictions": total_predictions,
                            "threshold": self.config.error_rate_threshold,
                        },
                    )

        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        model_id: str,
        deployment_id: str,
        metadata: Dict[str, Any],
    ):
        """Create and process an alert"""
        # Check alert cooldown
        alert_key = f"{deployment_id}_{alert_type}"

        with self._lock:
            recent_alerts = self.alert_cache[alert_key]

            if recent_alerts:
                last_alert_time = recent_alerts[-1]
                time_since_last = datetime.now() - last_alert_time

                if time_since_last.total_seconds() < (
                    self.config.alert_cooldown_minutes * 60
                ):
                    return  # Skip duplicate alert

            # Add to alert cache
            recent_alerts.append(datetime.now())

        # Create alert
        alert_id = f"{alert_type}_{deployment_id}_{int(datetime.now().timestamp())}"

        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            model_id=model_id,
            deployment_id=deployment_id,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        # Store alert in database
        self._store_alert(alert)

        # Process alert through handlers
        self._process_alert(alert)

        self.logger.warning(
            f"Alert created: {alert_type} for {deployment_id} - {message}"
        )

    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, alert_type, severity, message, model_id, 
                        deployment_id, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.alert_type,
                        alert.severity,
                        alert.message,
                        alert.model_id,
                        alert.deployment_id,
                        json.dumps(alert.metadata, default=str),
                        alert.timestamp,
                    ),
                )

        except Exception as e:
            self.logger.error(f"Error storing alert: {str(e)}")

    def _process_alert(self, alert: Alert):
        """Process alert through registered handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)

    def log_prediction(
        self,
        deployment_id: str,
        model_id: str,
        features: Dict[str, Any],
        prediction: float,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        """
        Log a prediction for monitoring.

        Args:
            deployment_id: Deployment identifier
            model_id: Model identifier
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence
            latency_ms: Prediction latency in milliseconds
        """
        try:
            # Store in memory for real-time monitoring
            with self._lock:
                self.prediction_history[deployment_id].append(
                    {
                        "features": features,
                        "prediction": prediction,
                        "confidence": confidence,
                        "latency_ms": latency_ms,
                        "timestamp": datetime.now(),
                    }
                )

                self.feature_history[deployment_id].append(features)

            # Store in database
            with sqlite3.connect(self.monitoring_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO prediction_logs (
                        deployment_id, model_id, features, prediction, 
                        confidence, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        deployment_id,
                        model_id,
                        json.dumps(features, default=str),
                        prediction,
                        confidence,
                        latency_ms,
                    ),
                )

        except Exception as e:
            self.logger.error(f"Error logging prediction: {str(e)}")

    def _get_recent_predictions(
        self, deployment_id: str, model_id: str, hours: int = 24
    ) -> List[Dict]:
        """Get recent predictions from database"""
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT features, prediction, confidence, latency_ms, timestamp
                    FROM prediction_logs
                    WHERE deployment_id = ? AND model_id = ?
                    AND timestamp > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                    LIMIT 10000
                """.format(
                        hours
                    ),
                    (deployment_id, model_id),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "features": json.loads(row[0]),
                            "prediction": row[1],
                            "confidence": row[2],
                            "latency_ms": row[3],
                            "timestamp": row[4],
                        }
                    )

                return results

        except Exception as e:
            self.logger.error(f"Error getting recent predictions: {str(e)}")
            return []

    def _get_recent_features(self, deployment_id: str, model_id: str) -> List[Dict]:
        """Get recent features for drift detection"""
        with self._lock:
            recent_features = list(self.feature_history[deployment_id])
            return recent_features[-self.config.feature_drift_window :]

    def _get_recent_prediction_logs(
        self, deployment_id: str, model_id: str
    ) -> List[Dict]:
        """Get recent prediction logs for system health"""
        return self._get_recent_predictions(deployment_id, model_id, hours=1)

    def _record_monitoring_metrics(self, deployment_id: str, model_id: str):
        """Record monitoring metrics to database"""
        try:
            # Calculate current metrics
            recent_predictions = self._get_recent_predictions(
                deployment_id, model_id, hours=1
            )

            if recent_predictions:
                predictions = [p["prediction"] for p in recent_predictions]
                confidences = [
                    p["confidence"]
                    for p in recent_predictions
                    if p["confidence"] is not None
                ]
                latencies = [
                    p["latency_ms"]
                    for p in recent_predictions
                    if p["latency_ms"] is not None
                ]

                metrics = {
                    "prediction_count": len(predictions),
                    "avg_prediction": np.mean(predictions),
                    "prediction_std": np.std(predictions),
                    "avg_confidence": np.mean(confidences) if confidences else None,
                    "avg_latency": np.mean(latencies) if latencies else None,
                    "max_latency": np.max(latencies) if latencies else None,
                }

                # Store metrics
                with sqlite3.connect(self.monitoring_db_path) as conn:
                    for metric_name, metric_value in metrics.items():
                        if metric_value is not None:
                            conn.execute(
                                """
                                INSERT INTO monitoring_metrics (
                                    deployment_id, model_id, metric_type, 
                                    metric_name, metric_value
                                ) VALUES (?, ?, ?, ?, ?)
                            """,
                                (
                                    deployment_id,
                                    model_id,
                                    "system",
                                    metric_name,
                                    metric_value,
                                ),
                            )

        except Exception as e:
            self.logger.error(f"Error recording monitoring metrics: {str(e)}")

    def get_monitoring_dashboard(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get monitoring dashboard data for a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Dashboard data with metrics and visualizations
        """
        try:
            dashboard_data = {
                "deployment_id": deployment_id,
                "last_updated": datetime.now().isoformat(),
                "monitoring_status": (
                    "active" if deployment_id in self.active_monitors else "inactive"
                ),
            }

            # Get recent alerts
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT alert_type, severity, message, created_at, resolved
                    FROM alerts
                    WHERE deployment_id = ?
                    AND created_at > datetime('now', '-24 hours')
                    ORDER BY created_at DESC
                    LIMIT 50
                """,
                    (deployment_id,),
                )

                dashboard_data["recent_alerts"] = [
                    {
                        "alert_type": row[0],
                        "severity": row[1],
                        "message": row[2],
                        "created_at": row[3],
                        "resolved": bool(row[4]),
                    }
                    for row in cursor.fetchall()
                ]

            # Get performance metrics
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT metric_name, metric_value, timestamp
                    FROM monitoring_metrics
                    WHERE deployment_id = ?
                    AND timestamp > datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                """,
                    (deployment_id,),
                )

                metrics_data = defaultdict(list)
                for row in cursor.fetchall():
                    metrics_data[row[0]].append({"value": row[1], "timestamp": row[2]})

                dashboard_data["metrics"] = dict(metrics_data)

            # Get prediction statistics
            recent_predictions = self._get_recent_predictions(
                deployment_id, "", hours=24
            )
            if recent_predictions:
                predictions = [p["prediction"] for p in recent_predictions]
                confidences = [
                    p["confidence"]
                    for p in recent_predictions
                    if p["confidence"] is not None
                ]

                dashboard_data["prediction_stats"] = {
                    "total_predictions": len(predictions),
                    "avg_prediction": float(np.mean(predictions)),
                    "prediction_std": float(np.std(predictions)),
                    "min_prediction": float(np.min(predictions)),
                    "max_prediction": float(np.max(predictions)),
                    "avg_confidence": (
                        float(np.mean(confidences)) if confidences else None
                    ),
                }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error getting monitoring dashboard: {str(e)}")
            return {"error": str(e)}

    def get_alerts(
        self,
        deployment_id: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filters.

        Args:
            deployment_id: Filter by deployment
            severity: Filter by severity
            resolved: Filter by resolution status
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []

            if deployment_id:
                query += " AND deployment_id = ?"
                params.append(deployment_id)

            if severity:
                query += " AND severity = ?"
                params.append(severity)

            if resolved is not None:
                query += " AND resolved = ?"
                params.append(resolved)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.execute(query, params)

                alerts = []
                for row in cursor.fetchall():
                    alerts.append(
                        {
                            "alert_id": row[0],
                            "alert_type": row[1],
                            "severity": row[2],
                            "message": row[3],
                            "model_id": row[4],
                            "deployment_id": row[5],
                            "metadata": json.loads(row[6]) if row[6] else {},
                            "resolved": bool(row[7]),
                            "created_at": row[8],
                            "resolved_at": row[9],
                        }
                    )

                return alerts

        except Exception as e:
            self.logger.error(f"Error getting alerts: {str(e)}")
            return []

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.monitoring_db_path) as conn:
                cursor = conn.execute(
                    """
                    UPDATE alerts 
                    SET resolved = TRUE, resolved_at = CURRENT_TIMESTAMP
                    WHERE alert_id = ?
                """,
                    (alert_id,),
                )

                if cursor.rowcount > 0:
                    self.logger.info(f"Resolved alert {alert_id}")
                    return True
                else:
                    self.logger.warning(f"Alert {alert_id} not found")
                    return False

        except Exception as e:
            self.logger.error(f"Error resolving alert: {str(e)}")
            return False

    # Placeholder methods for baseline storage/retrieval
    # In production, these would store/retrieve from a dedicated baseline storage

    def _get_recent_performance_metrics(
        self, deployment_id: str, model_id: str
    ) -> Dict:
        """Get recent performance metrics - placeholder implementation"""
        return {"accuracy": 0.85, "precision": 0.87, "recall": 0.83}

    def _get_baseline_performance(self, deployment_id: str, model_id: str) -> Dict:
        """Get baseline performance metrics - placeholder implementation"""
        return {"accuracy": 0.88, "precision": 0.89, "recall": 0.86}

    def _store_baseline_performance(
        self, deployment_id: str, model_id: str, metrics: Dict
    ):
        """Store baseline performance metrics - placeholder implementation"""
        pass

    def _get_baseline_features(self, deployment_id: str, model_id: str) -> List[Dict]:
        """Get baseline feature distribution - placeholder implementation"""
        return []

    def _store_baseline_features(
        self, deployment_id: str, model_id: str, features: List[Dict]
    ):
        """Store baseline feature distribution - placeholder implementation"""
        pass

    def _calculate_performance_degradation(self, recent: Dict, baseline: Dict) -> float:
        """Calculate performance degradation percentage"""
        if "accuracy" in recent and "accuracy" in baseline:
            return (baseline["accuracy"] - recent["accuracy"]) / baseline["accuracy"]
        return 0.0
