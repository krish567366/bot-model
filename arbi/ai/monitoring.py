"""
Model Monitoring & Data Drift Detection

Production-grade monitoring system for ML models including:
- Data drift detection (KS/JS divergence)
- Model performance monitoring
- Model calibration and uncertainty quantification
- Alert system for model degradation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import json
import pickle
import warnings

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Data drift alert information"""
    timestamp: datetime
    feature_name: str
    drift_type: str  # 'feature_drift' or 'label_drift'
    metric_name: str  # 'ks_statistic', 'js_divergence'
    metric_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    message: str


@dataclass
class PerformanceAlert:
    """Model performance degradation alert"""
    timestamp: datetime
    model_id: str
    metric_name: str  # 'auc', 'pnl', 'sharpe'
    current_value: float
    baseline_value: float
    threshold: float
    window_size: int
    severity: str
    message: str


@dataclass
class ModelCalibration:
    """Model calibration metrics"""
    model_id: str
    timestamp: datetime
    brier_score: float
    calibration_slope: float
    calibration_intercept: float
    expected_calibration_error: float
    reliability_diagram_path: Optional[str] = None


class DataDriftDetector:
    """Detect data drift in features and labels"""
    
    def __init__(self, 
                 ks_threshold: float = 0.05,
                 js_threshold: float = 0.1,
                 min_samples: int = 100):
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        self.min_samples = min_samples
        self.reference_distributions = {}
        self.drift_history = []
    
    def set_reference_distribution(self, feature_name: str, reference_data: np.ndarray):
        """Set reference distribution for a feature"""
        self.reference_distributions[feature_name] = {
            'data': reference_data.copy(),
            'mean': np.mean(reference_data),
            'std': np.std(reference_data),
            'timestamp': datetime.utcnow()
        }
        logger.info(f"Set reference distribution for {feature_name}: {len(reference_data)} samples")
    
    def detect_feature_drift(self, feature_name: str, current_data: np.ndarray) -> Optional[DriftAlert]:
        """Detect drift in a single feature"""
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for {feature_name}")
            return None
        
        if len(current_data) < self.min_samples:
            return None
        
        reference_data = self.reference_distributions[feature_name]['data']
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_data, current_data)
        
        # Jensen-Shannon divergence (for binned distributions)
        try:
            # Create histograms with same bins
            bins = np.linspace(
                min(np.min(reference_data), np.min(current_data)),
                max(np.max(reference_data), np.max(current_data)),
                50
            )
            ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
            cur_hist, _ = np.histogram(current_data, bins=bins, density=True)
            
            # Normalize to probabilities
            ref_hist = ref_hist / np.sum(ref_hist)
            cur_hist = cur_hist / np.sum(cur_hist)
            
            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Renormalize
            ref_hist = ref_hist / np.sum(ref_hist)
            cur_hist = cur_hist / np.sum(cur_hist)
            
            js_divergence = jensenshannon(ref_hist, cur_hist)
            
        except Exception as e:
            logger.error(f"Error computing JS divergence: {e}")
            js_divergence = 0.0
        
        # Check for drift
        drift_detected = False
        severity = "info"
        alerts = []
        
        if ks_pvalue < self.ks_threshold:
            drift_detected = True
            severity = "critical" if ks_pvalue < 0.01 else "warning"
            alerts.append(DriftAlert(
                timestamp=datetime.utcnow(),
                feature_name=feature_name,
                drift_type="feature_drift",
                metric_name="ks_statistic",
                metric_value=ks_statistic,
                threshold=self.ks_threshold,
                severity=severity,
                message=f"KS test p-value {ks_pvalue:.4f} < {self.ks_threshold}"
            ))
        
        if js_divergence > self.js_threshold:
            drift_detected = True
            severity = "critical" if js_divergence > 0.2 else "warning"
            alerts.append(DriftAlert(
                timestamp=datetime.utcnow(),
                feature_name=feature_name,
                drift_type="feature_drift",
                metric_name="js_divergence",
                metric_value=js_divergence,
                threshold=self.js_threshold,
                severity=severity,
                message=f"JS divergence {js_divergence:.4f} > {self.js_threshold}"
            ))
        
        # Store drift history
        drift_record = {
            'timestamp': datetime.utcnow(),
            'feature_name': feature_name,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'js_divergence': js_divergence,
            'drift_detected': drift_detected,
            'severity': severity
        }
        self.drift_history.append(drift_record)
        
        return alerts[0] if alerts else None
    
    def detect_batch_drift(self, feature_data: Dict[str, np.ndarray]) -> List[DriftAlert]:
        """Detect drift across multiple features"""
        all_alerts = []
        
        for feature_name, data in feature_data.items():
            alert = self.detect_feature_drift(feature_name, data)
            if alert:
                all_alerts.append(alert)
        
        return all_alerts
    
    def get_drift_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get drift detection summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        recent_history = [
            record for record in self.drift_history
            if record['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {'status': 'no_data', 'features_monitored': 0}
        
        features_with_drift = set()
        critical_alerts = 0
        warning_alerts = 0
        
        for record in recent_history:
            if record['drift_detected']:
                features_with_drift.add(record['feature_name'])
                if record['severity'] == 'critical':
                    critical_alerts += 1
                else:
                    warning_alerts += 1
        
        return {
            'status': 'drift_detected' if features_with_drift else 'stable',
            'features_monitored': len(set(r['feature_name'] for r in recent_history)),
            'features_with_drift': len(features_with_drift),
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'drift_features': list(features_with_drift)
        }


class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, 
                 auc_threshold: float = 0.05,
                 pnl_threshold: float = 0.1,
                 window_size: int = 100,
                 min_predictions: int = 50):
        self.auc_threshold = auc_threshold
        self.pnl_threshold = pnl_threshold
        self.window_size = window_size
        self.min_predictions = min_predictions
        
        self.performance_history = {}
        self.baseline_performance = {}
        self.alerts = []
    
    def set_baseline_performance(self, model_id: str, metrics: Dict[str, float]):
        """Set baseline performance metrics for a model"""
        self.baseline_performance[model_id] = {
            'metrics': metrics.copy(),
            'timestamp': datetime.utcnow()
        }
        logger.info(f"Set baseline performance for {model_id}: {metrics}")
    
    def record_prediction(self, model_id: str, prediction: float, actual: float, 
                         timestamp: Optional[datetime] = None, 
                         metadata: Optional[Dict] = None):
        """Record a single prediction for performance monitoring"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        record = {
            'timestamp': timestamp or datetime.utcnow(),
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }
        
        self.performance_history[model_id].append(record)
        
        # Keep only recent predictions
        if len(self.performance_history[model_id]) > self.window_size * 2:
            self.performance_history[model_id] = self.performance_history[model_id][-self.window_size:]
    
    def compute_rolling_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Compute rolling performance metrics"""
        if model_id not in self.performance_history:
            return None
        
        recent_records = self.performance_history[model_id][-self.window_size:]
        
        if len(recent_records) < self.min_predictions:
            return None
        
        predictions = np.array([r['prediction'] for r in recent_records])
        actuals = np.array([r['actual'] for r in recent_records])
        
        metrics = {}
        
        try:
            # Binary classification metrics (if predictions are probabilities)
            if np.all((predictions >= 0) & (predictions <= 1)) and np.all(np.isin(actuals, [0, 1])):
                metrics['auc'] = roc_auc_score(actuals, predictions)
                metrics['brier_score'] = brier_score_loss(actuals, predictions)
                metrics['log_loss'] = log_loss(actuals, predictions)
            
            # Regression metrics
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            
            metrics['mse'] = mse
            metrics['mae'] = mae
            metrics['rmse'] = np.sqrt(mse)
            
            # Correlation
            if np.std(predictions) > 0 and np.std(actuals) > 0:
                metrics['correlation'] = np.corrcoef(predictions, actuals)[0, 1]
            
            # PnL-like metrics (assuming actuals are returns)
            if len(actuals) > 1:
                # Directional accuracy
                pred_direction = np.sign(predictions)
                actual_direction = np.sign(actuals)
                metrics['directional_accuracy'] = np.mean(pred_direction == actual_direction)
                
                # Simulated PnL
                pnl = predictions * actuals  # Simplified PnL calculation
                metrics['mean_pnl'] = np.mean(pnl)
                metrics['pnl_std'] = np.std(pnl)
                metrics['sharpe_ratio'] = np.mean(pnl) / np.std(pnl) if np.std(pnl) > 0 else 0
        
        except Exception as e:
            logger.error(f"Error computing metrics for {model_id}: {e}")
            return None
        
        return metrics
    
    def check_performance_degradation(self, model_id: str) -> List[PerformanceAlert]:
        """Check for performance degradation"""
        if model_id not in self.baseline_performance:
            return []
        
        current_metrics = self.compute_rolling_metrics(model_id)
        if not current_metrics:
            return []
        
        baseline_metrics = self.baseline_performance[model_id]['metrics']
        alerts = []
        
        for metric_name in ['auc', 'sharpe_ratio', 'directional_accuracy']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                # Check for significant degradation
                if baseline_value > 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                    threshold = self.auc_threshold if metric_name == 'auc' else self.pnl_threshold
                    
                    if relative_change < -threshold:
                        severity = "critical" if relative_change < -threshold * 2 else "warning"
                        
                        alert = PerformanceAlert(
                            timestamp=datetime.utcnow(),
                            model_id=model_id,
                            metric_name=metric_name,
                            current_value=current_value,
                            baseline_value=baseline_value,
                            threshold=threshold,
                            window_size=self.window_size,
                            severity=severity,
                            message=f"{metric_name} degraded by {relative_change:.2%} "
                                   f"(current: {current_value:.4f}, baseline: {baseline_value:.4f})"
                        )
                        alerts.append(alert)
        
        return alerts
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get current model status"""
        current_metrics = self.compute_rolling_metrics(model_id)
        alerts = self.check_performance_degradation(model_id)
        
        status = "healthy"
        if any(alert.severity == "critical" for alert in alerts):
            status = "degraded"
        elif any(alert.severity == "warning" for alert in alerts):
            status = "warning"
        
        return {
            'model_id': model_id,
            'status': status,
            'current_metrics': current_metrics,
            'recent_alerts': [
                {
                    'severity': alert.severity,
                    'metric': alert.metric_name,
                    'message': alert.message
                } for alert in alerts
            ],
            'predictions_count': len(self.performance_history.get(model_id, [])),
            'last_prediction': self.performance_history.get(model_id, [{}])[-1].get('timestamp')
        }


class ModelCalibrationMonitor:
    """Monitor model calibration and uncertainty"""
    
    def __init__(self):
        self.calibration_history = []
    
    def compute_calibration_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_prob: np.ndarray,
                                  model_id: str,
                                  n_bins: int = 10) -> ModelCalibration:
        """Compute calibration metrics"""
        
        # Brier score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Fit line to calibration curve for slope/intercept
        try:
            slope, intercept, _, _, _ = stats.linregress(mean_predicted_value, fraction_of_positives)
        except:
            slope, intercept = 1.0, 0.0
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        calibration = ModelCalibration(
            model_id=model_id,
            timestamp=datetime.utcnow(),
            brier_score=brier_score,
            calibration_slope=slope,
            calibration_intercept=intercept,
            expected_calibration_error=ece
        )
        
        self.calibration_history.append(calibration)
        return calibration
    
    def create_reliability_diagram(self, 
                                 y_true: np.ndarray, 
                                 y_prob: np.ndarray,
                                 model_id: str,
                                 save_path: str = None) -> str:
        """Create reliability diagram"""
        try:
            import matplotlib.pyplot as plt
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Model {model_id}")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Reliability Diagram - {model_id}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if not save_path:
                save_path = f"calibration_plots/reliability_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating reliability diagram: {e}")
            return None
    
    def get_uncertainty_intervals(self, 
                                predictions: np.ndarray,
                                model_type: str = "ensemble",
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals"""
        
        if model_type == "ensemble" and predictions.ndim > 1:
            # For ensemble predictions, use quantiles
            alpha = 1 - confidence_level
            lower = np.quantile(predictions, alpha/2, axis=1)
            upper = np.quantile(predictions, 1 - alpha/2, axis=1)
            return lower, upper
        
        else:
            # Simple heuristic based on prediction confidence
            # This would be replaced with proper uncertainty quantification
            std_est = np.std(predictions) if len(predictions) > 1 else 0.1
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            
            lower = predictions - z_score * std_est
            upper = predictions + z_score * std_est
            return lower, upper


class AlertManager:
    """Manage and dispatch alerts"""
    
    def __init__(self, alert_config: Optional[Dict] = None):
        self.settings = get_settings()
        self.alert_config = alert_config or {}
        self.alert_history = []
    
    def send_drift_alert(self, alert: DriftAlert):
        """Send data drift alert"""
        message = f"🚨 Data Drift Alert: {alert.message}"
        logger.warning(message)
        
        # Store alert
        self.alert_history.append({
            'type': 'drift',
            'timestamp': alert.timestamp,
            'severity': alert.severity,
            'message': alert.message,
            'details': alert.__dict__
        })
        
        # TODO: Implement actual alerting (Slack/Discord/Email)
        self._dispatch_alert(message, alert.severity)
    
    def send_performance_alert(self, alert: PerformanceAlert):
        """Send performance degradation alert"""
        message = f"📉 Performance Alert: {alert.message}"
        logger.warning(message)
        
        self.alert_history.append({
            'type': 'performance',
            'timestamp': alert.timestamp,
            'severity': alert.severity,
            'message': alert.message,
            'details': alert.__dict__
        })
        
        self._dispatch_alert(message, alert.severity)
    
    def _dispatch_alert(self, message: str, severity: str):
        """Dispatch alert to configured channels"""
        # Print to console
        print(f"[{severity.upper()}] {message}")
        
        # TODO: Add integrations
        # if self.alert_config.get('slack_webhook'):
        #     self._send_slack(message, severity)
        # if self.alert_config.get('discord_webhook'):
        #     self._send_discord(message, severity)
        # if self.alert_config.get('email'):
        #     self._send_email(message, severity)


class MonitoringOrchestrator:
    """Main orchestrator for all monitoring components"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        self.drift_detector = DataDriftDetector(
            ks_threshold=self.config.get('ks_threshold', 0.05),
            js_threshold=self.config.get('js_threshold', 0.1)
        )
        
        self.performance_monitor = ModelPerformanceMonitor(
            auc_threshold=self.config.get('auc_threshold', 0.05),
            pnl_threshold=self.config.get('pnl_threshold', 0.1)
        )
        
        self.calibration_monitor = ModelCalibrationMonitor()
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        
        self.monitoring_active = True
    
    def initialize_monitoring(self, reference_data: Dict[str, np.ndarray], 
                            baseline_performance: Dict[str, Dict[str, float]]):
        """Initialize monitoring with reference data"""
        # Set reference distributions
        for feature_name, data in reference_data.items():
            self.drift_detector.set_reference_distribution(feature_name, data)
        
        # Set baseline performance
        for model_id, metrics in baseline_performance.items():
            self.performance_monitor.set_baseline_performance(model_id, metrics)
        
        logger.info("Monitoring system initialized")
    
    def monitor_batch(self, 
                     feature_data: Dict[str, np.ndarray],
                     model_predictions: Dict[str, Dict[str, np.ndarray]]):
        """Monitor a batch of predictions"""
        if not self.monitoring_active:
            return
        
        # Check for data drift
        drift_alerts = self.drift_detector.detect_batch_drift(feature_data)
        for alert in drift_alerts:
            self.alert_manager.send_drift_alert(alert)
        
        # Check model performance
        for model_id, preds in model_predictions.items():
            if 'predictions' in preds and 'actuals' in preds:
                for pred, actual in zip(preds['predictions'], preds['actuals']):
                    self.performance_monitor.record_prediction(model_id, pred, actual)
                
                # Check for degradation
                perf_alerts = self.performance_monitor.check_performance_degradation(model_id)
                for alert in perf_alerts:
                    self.alert_manager.send_performance_alert(alert)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        drift_summary = self.drift_detector.get_drift_summary()
        
        model_statuses = {}
        for model_id in self.performance_monitor.performance_history.keys():
            model_statuses[model_id] = self.performance_monitor.get_model_status(model_id)
        
        return {
            'timestamp': datetime.utcnow(),
            'status': 'healthy' if drift_summary['status'] == 'stable' and 
                     all(status['status'] == 'healthy' for status in model_statuses.values()) 
                     else 'warning',
            'drift_summary': drift_summary,
            'model_statuses': model_statuses,
            'recent_alerts': self.alert_manager.alert_history[-10:],  # Last 10 alerts
            'monitoring_active': self.monitoring_active
        }
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown of monitoring and trading"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        self.monitoring_active = False
        
        # Send critical alert
        message = f"🚨 EMERGENCY SHUTDOWN: {reason}"
        self.alert_manager._dispatch_alert(message, "critical")
        
        # TODO: Trigger trading system shutdown
        # self.trading_system.emergency_stop()
        
        return True


# Singleton instance
_monitoring_orchestrator = None

def get_monitoring_orchestrator(config: Optional[Dict] = None) -> MonitoringOrchestrator:
    """Get singleton monitoring orchestrator"""
    global _monitoring_orchestrator
    if _monitoring_orchestrator is None:
        _monitoring_orchestrator = MonitoringOrchestrator(config)
    return _monitoring_orchestrator
