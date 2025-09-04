import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Any
import warnings
from datetime import datetime, timedelta
import joblib
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    feature_name: str
    drift_score: float
    drift_detected: bool
    drift_type: str
    p_value: float
    threshold: float
    test_statistic: float
    interpretation: str

class DataDriftDetector:
    """Advanced data drift detection for MLOps monitoring"""
    
    def __init__(self, sensitivity_threshold: float = 0.05):
        self.sensitivity_threshold = sensitivity_threshold
        self.reference_distributions = {}
        self.drift_history = []
        
    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection"""
        self.reference_data = reference_data.copy()
        self._calculate_reference_distributions()
        logger.info(f"Reference data set with {len(reference_data)} samples")
    
    def _calculate_reference_distributions(self):
        """Calculate reference distributions for all features"""
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                # Continuous features
                self.reference_distributions[column] = {
                    'type': 'continuous',
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'values': self.reference_data[column].values,
                    'quantiles': np.percentile(self.reference_data[column], [25, 50, 75])
                }
            else:
                # Categorical features
                value_counts = self.reference_data[column].value_counts(normalize=True)
                self.reference_distributions[column] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'categories': set(self.reference_data[column].unique())
                }
    
    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftResult]:
        """Detect drift between reference and current data"""
        drift_results = []
        
        for column in self.reference_data.columns:
            if column in current_data.columns:
                drift_result = self._detect_feature_drift(column, current_data[column])
                drift_results.append(drift_result)
                
                # Record drift metrics
                from ..metrics.metrics_collector import metrics_collector
                metrics_collector.record_drift_metrics(
                    feature_name=column,
                    drift_score=drift_result.drift_score,
                    drift_type=drift_result.drift_type
                )
        
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_results': drift_results,
            'total_features': len(drift_results),
            'drifted_features': sum(1 for r in drift_results if r.drift_detected)
        })
        
        return drift_results
    
    def _detect_feature_drift(self, feature_name: str, current_values: pd.Series) -> DriftResult:
        """Detect drift for a specific feature"""
        
        reference_info = self.reference_distributions[feature_name]
        
        if reference_info['type'] == 'continuous':
            return self._detect_continuous_drift(feature_name, current_values, reference_info)
        else:
            return self._detect_categorical_drift(feature_name, current_values, reference_info)
    
    def _detect_continuous_drift(self, feature_name: str, current_values: pd.Series, 
                               reference_info: Dict) -> DriftResult:
        """Detect drift in continuous features using Kolmogorov-Smirnov test"""
        
        reference_values = reference_info['values']
        
        # Remove NaN values
        current_clean = current_values.dropna()
        
        if len(current_clean) == 0:
            return DriftResult(
                feature_name=feature_name,
                drift_score=1.0,
                drift_detected=True,
                drift_type='missing_data',
                p_value=0.0,
                threshold=self.sensitivity_threshold,
                test_statistic=1.0,
                interpretation="All values are missing in current data"
            )
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(reference_values, current_clean)
        
        # Calculate additional drift metrics
        mean_shift = abs(current_clean.mean() - reference_info['mean']) / reference_info['std']
        std_ratio = current_clean.std() / reference_info['std'] if reference_info['std'] > 0 else 1.0
        
        # Combined drift score
        drift_score = max(ks_statistic, mean_shift / 3, abs(1 - std_ratio))
        
        # Determine drift type
        if mean_shift > 2:
            drift_type = 'mean_shift'
        elif abs(1 - std_ratio) > 0.5:
            drift_type = 'variance_change'
        else:
            drift_type = 'distribution_change'
        
        # Interpretation
        interpretation = self._interpret_continuous_drift(
            ks_statistic, p_value, mean_shift, std_ratio
        )
        
        return DriftResult(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=p_value < self.sensitivity_threshold,
            drift_type=drift_type,
            p_value=p_value,
            threshold=self.sensitivity_threshold,
            test_statistic=ks_statistic,
            interpretation=interpretation
        )
    
    def _detect_categorical_drift(self, feature_name: str, current_values: pd.Series, 
                                reference_info: Dict) -> DriftResult:
        """Detect drift in categorical features using Chi-square test"""
        
        reference_dist = reference_info['distribution']
        reference_categories = reference_info['categories']
        
        # Get current distribution
        current_dist = current_values.value_counts(normalize=True).to_dict()
        current_categories = set(current_values.unique())
        
        # Check for new categories
        new_categories = current_categories - reference_categories
        missing_categories = reference_categories - current_categories
        
        # Create contingency table
        all_categories = reference_categories.union(current_categories)
        reference_counts = [reference_dist.get(cat, 0) * len(self.reference_data) for cat in all_categories]
        current_counts = [current_dist.get(cat, 0) * len(current_values) for cat in all_categories]
        
        # Chi-square test
        try:
            chi2_statistic, p_value = stats.chisquare(current_counts, reference_counts)
        except ValueError:
            # Handle case where all values are zero
            chi2_statistic, p_value = 0, 1
        
        # Calculate drift score
        drift_score = min(chi2_statistic / len(all_categories), 1.0)
        
        # Determine drift type
        if new_categories:
            drift_type = 'new_categories'
        elif missing_categories:
            drift_type = 'missing_categories'
        else:
            drift_type = 'distribution_change'
        
        # Interpretation
        interpretation = self._interpret_categorical_drift(
            chi2_statistic, p_value, new_categories, missing_categories
        )
        
        return DriftResult(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=p_value < self.sensitivity_threshold,
            drift_type=drift_type,
            p_value=p_value,
            threshold=self.sensitivity_threshold,
            test_statistic=chi2_statistic,
            interpretation=interpretation
        )
    
    def _interpret_continuous_drift(self, ks_statistic: float, p_value: float, 
                                  mean_shift: float, std_ratio: float) -> str:
        """Generate human-readable interpretation of continuous drift"""
        
        interpretations = []
        
        if p_value < 0.001:
            interpretations.append("Strong statistical evidence of distribution change")
        elif p_value < 0.05:
            interpretations.append("Moderate evidence of distribution change")
        
        if mean_shift > 3:
            interpretations.append("Large shift in average values")
        elif mean_shift > 1:
            interpretations.append("Moderate shift in average values")
        
        if std_ratio > 2 or std_ratio < 0.5:
            interpretations.append("Significant change in data variability")
        
        if not interpretations:
            interpretations.append("No significant drift detected")
        
        return "; ".join(interpretations)
    
    def _interpret_categorical_drift(self, chi2_statistic: float, p_value: float,
                                   new_categories: set, missing_categories: set) -> str:
        """Generate human-readable interpretation of categorical drift"""
        
        interpretations = []
        
        if new_categories:
            interpretations.append(f"New categories detected: {list(new_categories)}")
        
        if missing_categories:
            interpretations.append(f"Missing categories: {list(missing_categories)}")
        
        if p_value < 0.001:
            interpretations.append("Strong evidence of category distribution change")
        elif p_value < 0.05:
            interpretations.append("Moderate evidence of category distribution change")
        
        if not interpretations:
            interpretations.append("No significant categorical drift detected")
        
        return "; ".join(interpretations)
    
    def get_drift_summary(self, time_window_hours: int = 24) -> Dict:
        """Get summary of drift detection over time window"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_drift = [d for d in self.drift_history if d['timestamp'] > cutoff_time]
        
        if not recent_drift:
            return {"message": "No drift detection history in the specified time window"}
        
        # Calculate aggregate statistics
        total_detections = len(recent_drift)
        total_features_checked = sum(d['total_features'] for d in recent_drift)
        total_drifted_features = sum(d['drifted_features'] for d in recent_drift)
        
        drift_rate = total_drifted_features / total_features_checked if total_features_checked > 0 else 0
        
        # Feature-level statistics
        feature_drift_counts = {}
        for drift_event in recent_drift:
            for result in drift_event['drift_results']:
                feature_name = result.feature_name
                if feature_name not in feature_drift_counts:
                    feature_drift_counts[feature_name] = 0
                if result.drift_detected:
                    feature_drift_counts[feature_name] += 1
        
        # Most problematic features
        problematic_features = sorted(
            feature_drift_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'time_window_hours': time_window_hours,
            'total_drift_checks': total_detections,
            'total_features_monitored': len(feature_drift_counts),
            'overall_drift_rate': drift_rate,
            'most_problematic_features': problematic_features,
            'latest_drift_check': recent_drift[-1]['timestamp'].isoformat() if recent_drift else None
        }

class ConceptDriftDetector:
    """Detect concept drift by monitoring model performance"""
    
    def __init__(self, performance_threshold: float = 0.05):
        self.performance_threshold = performance_threshold
        self.performance_history = []
        self.baseline_performance = None
        
    def set_baseline_performance(self, accuracy: float, precision: float, 
                               recall: float, f1_score: float):
        """Set baseline model performance metrics"""
        self.baseline_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'timestamp': datetime.now()
        }
        logger.info(f"Baseline performance set: Accuracy={accuracy:.3f}")
    
    def check_performance_drift(self, current_accuracy: float, current_precision: float,
                              current_recall: float, current_f1: float) -> Dict:
        """Check if current performance indicates concept drift"""
        
        if not self.baseline_performance:
            logger.warning("No baseline performance set for concept drift detection")
            return {"drift_detected": False, "reason": "No baseline set"}
        
        # Calculate performance degradation
        accuracy_drop = self.baseline_performance['accuracy'] - current_accuracy
        precision_drop = self.baseline_performance['precision'] - current_precision
        recall_drop = self.baseline_performance['recall'] - current_recall
        f1_drop = self.baseline_performance['f1_score'] - current_f1
        
        # Check if any metric dropped significantly
        drift_detected = (
            accuracy_drop > self.performance_threshold or
            precision_drop > self.performance_threshold or
            recall_drop > self.performance_threshold or
            f1_drop > self.performance_threshold
        )
        
        # Record performance
        performance_record = {
            'timestamp': datetime.now(),
            'accuracy': current_accuracy,
            'precision': current_precision,
            'recall': current_recall,
            'f1_score': current_f1,
            'accuracy_drop': accuracy_drop,
            'drift_detected': drift_detected
        }
        
        self.performance_history.append(performance_record)
        
        # Record concept drift metric
        from ..metrics.metrics_collector import metrics_collector
        drift_score = max(accuracy_drop, precision_drop, recall_drop, f1_drop)
        metrics_collector.record_drift_metrics(
            feature_name='model_performance',
            drift_score=drift_score,
            drift_type='concept_drift'
        )
        
        return {
            'drift_detected': drift_detected,
            'accuracy_drop': accuracy_drop,
            'precision_drop': precision_drop,
            'recall_drop': recall_drop,
            'f1_drop': f1_drop,
            'max_degradation': max(accuracy_drop, precision_drop, recall_drop, f1_drop),
            'recommendation': self._get_drift_recommendation(drift_detected, accuracy_drop)
        }
    
    def _get_drift_recommendation(self, drift_detected: bool, accuracy_drop: float) -> str:
        """Get recommendation based on drift detection"""
        
        if not drift_detected:
            return "No action needed - model performance is stable"
        
        if accuracy_drop > 0.1:
            return "URGENT: Immediate model retraining required - significant performance drop"
        elif accuracy_drop > 0.05:
            return "Schedule model retraining within 24 hours - performance degradation detected"
        else:
            return "Monitor closely - minor performance degradation detected"

# Global drift detectors
data_drift_detector = DataDriftDetector()
concept_drift_detector = ConceptDriftDetector()