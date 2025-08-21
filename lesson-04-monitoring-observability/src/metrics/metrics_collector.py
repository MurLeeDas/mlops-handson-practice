import time
import threading
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    name: str
    metric_type: str  # counter, gauge, histogram
    description: str
    labels: list
    buckets: Optional[list] = None

class MLOpsMetricsCollector:
    """Production-grade metrics collection for MLOps systems"""
    
    def __init__(self):
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Initialize core metrics
        self.setup_core_metrics()
        
        # Initialize business metrics  
        self.setup_business_metrics()
        
        # Initialize model performance metrics
        self.setup_model_metrics()
        
        # Initialize data quality metrics
        self.setup_data_metrics()
        
        # Metrics storage for calculations
        self.prediction_history = []
        self.performance_history = {}
        
        # Start background metrics calculation
        self.start_background_calculations()
    
    def setup_core_metrics(self):
        """Setup core infrastructure metrics"""
        
        # API Performance Metrics
        self.api_request_count = Counter(
            'mlops_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'mlops_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.api_concurrent_requests = Gauge(
            'mlops_api_concurrent_requests',
            'Number of concurrent API requests',
            registry=self.registry
        )
        
        # Model Serving Metrics
        self.model_predictions_total = Counter(
            'mlops_model_predictions_total',
            'Total model predictions made',
            ['model_version', 'model_algorithm', 'prediction_type'],
            registry=self.registry
        )
        
        self.model_prediction_latency = Histogram(
            'mlops_model_prediction_latency_seconds',
            'Model prediction latency in seconds',
            ['model_version', 'model_algorithm'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_prediction_confidence = Histogram(
            'mlops_model_prediction_confidence',
            'Model prediction confidence scores',
            ['model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_state = Gauge(
            'mlops_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['circuit_breaker_name', 'model_version'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'mlops_circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['circuit_breaker_name', 'failure_reason'],
            registry=self.registry
        )
    
    def setup_business_metrics(self):
        """Setup business KPI metrics"""
        
        # Churn Metrics
        self.churn_predictions_high_risk = Gauge(
            'mlops_churn_predictions_high_risk',
            'Number of customers predicted as high churn risk',
            ['time_window', 'customer_segment'],
            registry=self.registry
        )
        
        self.churn_prevention_actions = Counter(
            'mlops_churn_prevention_actions_total',
            'Total churn prevention actions taken',
            ['action_type', 'customer_segment'],
            registry=self.registry
        )
        
        self.revenue_impact = Gauge(
            'mlops_revenue_impact_dollars',
            'Revenue impact from ML predictions',
            ['impact_type', 'time_window'],
            registry=self.registry
        )
        
        # Campaign Effectiveness
        self.campaign_effectiveness = Gauge(
            'mlops_campaign_effectiveness_rate',
            'Campaign effectiveness rate',
            ['campaign_type', 'model_version'],
            registry=self.registry
        )
        
        self.false_positive_rate = Gauge(
            'mlops_false_positive_rate',
            'False positive rate for predictions',
            ['model_version', 'threshold'],
            registry=self.registry
        )
    
    def setup_model_metrics(self):
        """Setup model performance metrics"""
        
        # Model Accuracy Metrics
        self.model_accuracy = Gauge(
            'mlops_model_accuracy',
            'Model accuracy over time window',
            ['model_version', 'time_window'],
            registry=self.registry
        )
        
        self.model_precision = Gauge(
            'mlops_model_precision',
            'Model precision',
            ['model_version', 'class_label'],
            registry=self.registry
        )
        
        self.model_recall = Gauge(
            'mlops_model_recall',
            'Model recall',
            ['model_version', 'class_label'],
            registry=self.registry
        )
        
        self.model_f1_score = Gauge(
            'mlops_model_f1_score',
            'Model F1 score',
            ['model_version'],
            registry=self.registry
        )
        
        # Model Drift Metrics
        self.data_drift_score = Gauge(
            'mlops_data_drift_score',
            'Data drift score for features',
            ['feature_name', 'drift_type'],
            registry=self.registry
        )
        
        self.concept_drift_score = Gauge(
            'mlops_concept_drift_score',
            'Concept drift score for models',
            ['model_version'],
            registry=self.registry
        )
        
        self.feature_importance_change = Gauge(
            'mlops_feature_importance_change',
            'Change in feature importance over time',
            ['feature_name', 'model_version'],
            registry=self.registry
        )
    
    def setup_data_metrics(self):
        """Setup data quality metrics"""
        
        # Data Quality Metrics
        self.data_quality_score = Gauge(
            'mlops_data_quality_score',
            'Overall data quality score',
            ['data_source', 'quality_dimension'],
            registry=self.registry
        )
        
        self.missing_values_percentage = Gauge(
            'mlops_missing_values_percentage',
            'Percentage of missing values',
            ['feature_name', 'data_source'],
            registry=self.registry
        )
        
        self.data_volume = Gauge(
            'mlops_data_volume_records',
            'Volume of data processed',
            ['data_source', 'time_window'],
            registry=self.registry
        )
        
        self.data_freshness = Gauge(
            'mlops_data_freshness_hours',
            'Data freshness in hours',
            ['data_source'],
            registry=self.registry
        )
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        self.api_request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.api_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, model_version: str, algorithm: str, prediction_type: str, 
                         latency: float, confidence: float):
        """Record model prediction metrics"""
        
        # Increment prediction counter
        self.model_predictions_total.labels(
            model_version=model_version,
            model_algorithm=algorithm,
            prediction_type=prediction_type
        ).inc()
        
        # Record latency
        self.model_prediction_latency.labels(
            model_version=model_version,
            model_algorithm=algorithm
        ).observe(latency)
        
        # Record confidence
        self.model_prediction_confidence.labels(model_version=model_version).observe(confidence)
        
        # Store for performance calculations
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'model_version': model_version,
            'algorithm': algorithm,
            'confidence': confidence,
            'latency': latency
        })
        
        # Keep only recent history (memory management)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.prediction_history = [
            p for p in self.prediction_history if p['timestamp'] > cutoff_time
        ]
    
    def record_circuit_breaker_event(self, circuit_name: str, model_version: str, 
                                   state: str, failure_reason: str = None):
        """Record circuit breaker metrics"""
        
        state_mapping = {'closed': 0, 'open': 1, 'half_open': 2}
        state_value = state_mapping.get(state.lower(), 0)
        
        self.circuit_breaker_state.labels(
            circuit_breaker_name=circuit_name,
            model_version=model_version
        ).set(state_value)
        
        if failure_reason:
            self.circuit_breaker_failures.labels(
                circuit_breaker_name=circuit_name,
                failure_reason=failure_reason
            ).inc()
    
    def record_business_metric(self, metric_type: str, value: float, labels: Dict[str, str]):
        """Record business KPI metrics"""
        
        if metric_type == 'churn_high_risk':
            self.churn_predictions_high_risk.labels(**labels).set(value)
        elif metric_type == 'revenue_impact':
            self.revenue_impact.labels(**labels).set(value)
        elif metric_type == 'campaign_effectiveness':
            self.campaign_effectiveness.labels(**labels).set(value)
        elif metric_type == 'false_positive_rate':
            self.false_positive_rate.labels(**labels).set(value)
    
    def record_model_performance(self, model_version: str, accuracy: float, precision: float, 
                               recall: float, f1_score: float, time_window: str = '1h'):
        """Record model performance metrics"""
        
        self.model_accuracy.labels(model_version=model_version, time_window=time_window).set(accuracy)
        self.model_precision.labels(model_version=model_version, class_label='churn').set(precision)
        self.model_recall.labels(model_version=model_version, class_label='churn').set(recall)
        self.model_f1_score.labels(model_version=model_version).set(f1_score)
        
        # Store for trend analysis
        if model_version not in self.performance_history:
            self.performance_history[model_version] = []
        
        self.performance_history[model_version].append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
    
    def record_drift_metrics(self, feature_name: str, drift_score: float, drift_type: str):
        """Record data/concept drift metrics"""
        self.data_drift_score.labels(feature_name=feature_name, drift_type=drift_type).set(drift_score)
    
    def start_background_calculations(self):
        """Start background thread for metric calculations"""
        def calculate_metrics():
            while True:
                try:
                    self.calculate_derived_metrics()
                    time.sleep(60)  # Calculate every minute
                except Exception as e:
                    logger.error(f"Error calculating metrics: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=calculate_metrics, daemon=True)
        thread.start()
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from stored data"""
        
        if not self.prediction_history:
            return
        
        # Calculate recent performance metrics
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if recent_predictions:
            # Calculate average confidence by model
            model_confidences = {}
            for pred in recent_predictions:
                model = pred['model_version']
                if model not in model_confidences:
                    model_confidences[model] = []
                model_confidences[model].append(pred['confidence'])
            
            # Update confidence metrics
            for model, confidences in model_confidences.items():
                avg_confidence = sum(confidences) / len(confidences)
                # This would typically update a gauge metric
                logger.info(f"Model {model} average confidence: {avg_confidence:.3f}")
    
    def get_metrics(self):
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def get_metrics_summary(self):
        """Get human-readable metrics summary"""
        summary = {
            'total_predictions': len(self.prediction_history),
            'recent_predictions_1h': len([
                p for p in self.prediction_history 
                if p['timestamp'] > datetime.now() - timedelta(hours=1)
            ]),
            'model_versions_active': len(set(p['model_version'] for p in self.prediction_history)),
            'avg_prediction_latency': np.mean([p['latency'] for p in self.prediction_history]) if self.prediction_history else 0,
            'avg_confidence': np.mean([p['confidence'] for p in self.prediction_history]) if self.prediction_history else 0
        }
        return summary

# Global metrics collector instance
metrics_collector = MLOpsMetricsCollector()