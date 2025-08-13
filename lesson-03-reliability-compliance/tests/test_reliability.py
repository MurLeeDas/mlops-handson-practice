import pytest
import asyncio
import time
from src.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from src.monitoring.health_monitor import ModelHealthMonitor

class TestCircuitBreaker:
    def test_normal_operation(self):
        """Test circuit breaker in normal operation"""
        breaker = CircuitBreaker("test_model")
        
        # Should allow execution
        assert breaker.can_execute() == True
        
        # Successful execution
        result = breaker.execute(lambda x: x * 2, 5)
        assert result == 10
        assert breaker.failure_count == 0
    
    def test_failure_threshold(self):
        """Test circuit breaker opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=2, timeout_duration=1)
        breaker = CircuitBreaker("test_model", config)
        
        # Cause failures
        for _ in range(2):
            try:
                breaker.execute(lambda: 1/0)  # Division by zero
            except:
                pass
        
        # Should be open now
        assert breaker.can_execute() == False
        
        # Should raise circuit breaker error
        with pytest.raises(CircuitBreakerError):
            breaker.execute(lambda: "success")
    
    def test_timeout_recovery(self):
        """Test circuit breaker recovery after timeout"""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_duration=0.1)
        breaker = CircuitBreaker("test_model", config)
        
        # Cause failure
        try:
            breaker.execute(lambda: 1/0)
        except:
            pass
        
        # Should be open
        assert breaker.can_execute() == False
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should allow execution again
        assert breaker.can_execute() == True

if __name__ == "__main__":
    # Run basic tests
    test_cb = TestCircuitBreaker()
    test_cb.test_normal_operation()
    test_cb.test_failure_threshold()
    test_cb.test_timeout_recovery()
    print("✅ All circuit breaker tests passed!")