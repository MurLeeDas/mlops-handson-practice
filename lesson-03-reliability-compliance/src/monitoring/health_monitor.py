import time
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = None

class ModelHealthMonitor:
    """Comprehensive health monitoring for ML models"""
    
    def __init__(self):
        self.health_history: List[HealthCheck] = []
        self.alert_thresholds = {
            'response_time_warning': 1.0,      # seconds
            'response_time_critical': 5.0,     # seconds
            'accuracy_warning': 0.75,          # below this triggers warning
            'accuracy_critical': 0.65,         # below this triggers critical
            'error_rate_warning': 0.05,        # 5% error rate
            'error_rate_critical': 0.15        # 15% error rate
        }
        self.logger = logging.getLogger("model_health_monitor")
    
    async def check_model_health(self, model_name: str, model_predict_func, test_data: Dict) -> HealthCheck:
        """Perform comprehensive health check on a model"""
        start_time = time.time()
        
        try:
            # Test model prediction
            result = model_predict_func(test_data)
            response_time = time.time() - start_time
            
            # Determine health status based on response time
            if response_time > self.alert_thresholds['response_time_critical']:
                status = HealthStatus.CRITICAL
                message = f"Critical: Response time {response_time:.2f}s exceeds {self.alert_thresholds['response_time_critical']}s"
            elif response_time > self.alert_thresholds['response_time_warning']:
                status = HealthStatus.DEGRADED
                message = f"Warning: Response time {response_time:.2f}s exceeds {self.alert_thresholds['response_time_warning']}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Healthy: Response time {response_time:.2f}s"
            
            health_check = HealthCheck(
                name=model_name,
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'prediction_successful': True,
                    'prediction_result': result,
                    'model_available': True
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            health_check = HealthCheck(
                name=model_name,
                status=HealthStatus.CRITICAL,
                message=f"Critical: Model failed - {str(e)}",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    'prediction_successful': False,
                    'error': str(e),
                    'model_available': False
                }
            )
        
        # Store health check result
        self.health_history.append(health_check)
        
        # Keep only recent history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-500:]
        
        return health_check
    
    def get_health_summary(self, model_name: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for models"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent health checks
        recent_checks = [
            check for check in self.health_history
            if check.timestamp >= cutoff_time and (model_name is None or check.name == model_name)
        ]
        
        if not recent_checks:
            return {"message": "No recent health data"}
        
        # Calculate metrics
        total_checks = len(recent_checks)
        healthy_checks = len([c for c in recent_checks if c.status == HealthStatus.HEALTHY])
        degraded_checks = len([c for c in recent_checks if c.status == HealthStatus.DEGRADED])
        unhealthy_checks = len([c for c in recent_checks if c.status == HealthStatus.UNHEALTHY])
        critical_checks = len([c for c in recent_checks if c.status == HealthStatus.CRITICAL])
        
        avg_response_time = sum(c.response_time for c in recent_checks) / total_checks
        max_response_time = max(c.response_time for c in recent_checks)
        
        return {
            'model_name': model_name or 'all_models',
            'period_hours': hours,
            'total_checks': total_checks,
            'health_distribution': {
                'healthy': healthy_checks,
                'degraded': degraded_checks,
                'unhealthy': unhealthy_checks,
                'critical': critical_checks
            },
            'health_percentage': (healthy_checks / total_checks) * 100,
            'performance_metrics': {
                'avg_response_time': round(avg_response_time, 3),
                'max_response_time': round(max_response_time, 3),
                'sla_compliance': (healthy_checks + degraded_checks) / total_checks * 100
            },
            'latest_status': recent_checks[-1].status.value,
            'alerts_triggered': critical_checks + unhealthy_checks
        }
    
    def should_trigger_alert(self, health_check: HealthCheck) -> bool:
        """Determine if health check should trigger an alert"""
        return health_check.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]
    
    def get_alert_message(self, health_check: HealthCheck) -> str:
        """Generate alert message for health check"""
        return (
            f"🚨 MODEL HEALTH ALERT 🚨\n"
            f"Model: {health_check.name}\n"
            f"Status: {health_check.status.value.upper()}\n"
            f"Message: {health_check.message}\n"
            f"Response Time: {health_check.response_time:.2f}s\n"
            f"Timestamp: {health_check.timestamp}\n"
            f"Action Required: Investigate model performance"
        )

# Global health monitor
health_monitor = ModelHealthMonitor()