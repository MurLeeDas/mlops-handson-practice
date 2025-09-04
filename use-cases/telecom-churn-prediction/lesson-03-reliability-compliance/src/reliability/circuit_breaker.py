import time
import logging
from enum import Enum
from typing import Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    timeout_duration: int = 60          # Seconds to wait before half-open
    success_threshold: int = 3          # Successes needed to close
    request_timeout: float = 5.0        # Max request time

class CircuitBreakerError(Exception):
    pass

class CircuitBreaker:
    """Circuit breaker pattern for ML model reliability"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout period has passed
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_duration
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.error(f"Circuit breaker {self.name} OPENED - too many failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.error(f"Circuit breaker {self.name} back to OPEN - test failed")
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self.can_execute():
            raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Add timeout to function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Check if execution took too long
            if execution_time > self.config.request_timeout:
                self.record_failure()
                raise CircuitBreakerError(f"Request timeout: {execution_time:.2f}s > {self.config.request_timeout}s")
            
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure()
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "can_execute": self.can_execute()
        }

# Circuit breaker manager for multiple models
class CircuitBreakerManager:
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker for a model"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict]:
        """Get state of all circuit breakers"""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}

# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()