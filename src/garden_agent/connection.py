"""Advanced database connection management and session handling."""

import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, Callable
from functools import wraps
from threading import Lock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, TimeoutError
from sqlalchemy.pool import QueuePool
from sqlalchemy import text

from .database import db_manager


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Advanced connection management with pooling and retry logic."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize connection manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._stats = {
            'total_connections': 0,
            'failed_connections': 0,
            'retry_attempts': 0,
            'active_sessions': 0
        }
        self._stats_lock = Lock()
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection statistics."""
        with self._stats_lock:
            return self._stats.copy()
    
    def _increment_stat(self, stat_name: str, amount: int = 1):
        """Thread-safe statistics increment."""
        with self._stats_lock:
            self._stats[stat_name] = self._stats.get(stat_name, 0) + amount
    
    @contextmanager
    def get_session(
        self, 
        autocommit: bool = False,
        read_only: bool = False
    ) -> Generator[Session, None, None]:
        """Get database session with error handling and retry logic.
        
        Args:
            autocommit: Enable autocommit mode
            read_only: Session is read-only (optimization hint)
        """
        session = None
        attempt = 0
        
        while attempt <= self.max_retries:
            try:
                self._increment_stat('total_connections')
                self._increment_stat('active_sessions')
                
                session = db_manager.SessionLocal()
                
                # Configure session (skip pragma for now - causing issues)
                
                yield session
                
                # Commit if not autocommit and no error occurred
                if not autocommit and session.is_active:
                    session.commit()
                
                break  # Success, exit retry loop
                
            except (DisconnectionError, TimeoutError) as e:
                logger.warning(f"Database connection issue (attempt {attempt + 1}): {e}")
                
                if session:
                    try:
                        session.rollback()
                    except:
                        pass
                    session.close()
                    session = None
                
                attempt += 1
                self._increment_stat('retry_attempts')
                
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
                else:
                    self._increment_stat('failed_connections')
                    raise
            
            except Exception as e:
                logger.error(f"Database session error: {e}")
                
                if session:
                    try:
                        session.rollback()
                    except:
                        pass
                
                self._increment_stat('failed_connections')
                raise
            
            finally:
                self._increment_stat('active_sessions', -1)
                if session:
                    session.close()
    
    def test_connection(self) -> bool:
        """Test database connection health."""
        try:
            with self.get_session(read_only=True) as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class SessionManager:
    """Session lifecycle management with automatic cleanup."""
    
    def __init__(self, connection_manager: Optional[ConnectionManager] = None):
        """Initialize session manager."""
        self.connection_manager = connection_manager or ConnectionManager()
        self._active_sessions = {}
    
    def create_scoped_session(self, session_id: str) -> Session:
        """Create a scoped session that can be reused."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        session = db_manager.SessionLocal()
        self._active_sessions[session_id] = session
        return session
    
    def close_scoped_session(self, session_id: str):
        """Close and remove a scoped session."""
        if session_id in self._active_sessions:
            session = self._active_sessions.pop(session_id)
            try:
                session.close()
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")
    
    def close_all_sessions(self):
        """Close all active scoped sessions."""
        for session_id in list(self._active_sessions.keys()):
            self.close_scoped_session(session_id)
    
    @contextmanager
    def transaction(self, session: Optional[Session] = None) -> Generator[Session, None, None]:
        """Manage database transaction with automatic rollback on error."""
        if session:
            # Use provided session
            try:
                yield session
                if session.is_active:
                    session.commit()
            except Exception:
                session.rollback()
                raise
        else:
            # Create new session for transaction
            with self.connection_manager.get_session() as session:
                try:
                    yield session
                except Exception:
                    session.rollback()
                    raise


def with_db_session(
    autocommit: bool = False,
    read_only: bool = False,
    retry_on_failure: bool = True
):
    """Decorator to automatically provide database session to function.
    
    Args:
        autocommit: Enable autocommit mode
        read_only: Session is read-only
        retry_on_failure: Retry on connection failures
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            connection_manager = ConnectionManager() if retry_on_failure else None
            
            if connection_manager:
                with connection_manager.get_session(autocommit=autocommit, read_only=read_only) as session:
                    return func(session, *args, **kwargs)
            else:
                # Simple session without retry logic
                session = db_manager.SessionLocal()
                try:
                    result = func(session, *args, **kwargs)
                    if not autocommit and session.is_active:
                        session.commit()
                    return result
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()
        
        return wrapper
    return decorator


def with_transaction(func: Callable):
    """Decorator to wrap function in database transaction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_manager = SessionManager()
        with session_manager.transaction() as session:
            return func(session, *args, **kwargs)
    return wrapper


class PerformanceMonitor:
    """Monitor database performance and connection health."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.query_times = []
        self.slow_query_threshold = 1.0  # seconds
        self.slow_queries = []
    
    def log_query_time(self, query: str, execution_time: float):
        """Log query execution time."""
        self.query_times.append(execution_time)
        
        if execution_time > self.slow_query_threshold:
            self.slow_queries.append({
                'query': query[:200],  # Truncate long queries
                'time': execution_time,
                'timestamp': time.time()
            })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.query_times:
            return {'message': 'No query data available'}
        
        avg_time = sum(self.query_times) / len(self.query_times)
        max_time = max(self.query_times)
        
        return {
            'total_queries': len(self.query_times),
            'average_query_time': round(avg_time, 3),
            'max_query_time': round(max_time, 3),
            'slow_query_count': len(self.slow_queries),
            'recent_slow_queries': self.slow_queries[-5:] if self.slow_queries else []
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.query_times.clear()
        self.slow_queries.clear()


class DatabaseHealthChecker:
    """Monitor database health and connectivity."""
    
    def __init__(self, connection_manager: ConnectionManager):
        """Initialize health checker."""
        self.connection_manager = connection_manager
        self.last_check_time = None
        self.last_check_result = None
    
    def check_health(self, force_check: bool = False) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        now = time.time()
        
        # Use cached result if recent (within 60 seconds)
        if not force_check and self.last_check_time and (now - self.last_check_time) < 60:
            return self.last_check_result
        
        health_status = {
            'timestamp': now,
            'overall_healthy': True,
            'checks': {}
        }
        
        # Test basic connectivity
        try:
            connectivity_ok = self.connection_manager.test_connection()
            health_status['checks']['connectivity'] = {
                'status': 'ok' if connectivity_ok else 'failed',
                'healthy': connectivity_ok
            }
            if not connectivity_ok:
                health_status['overall_healthy'] = False
        except Exception as e:
            health_status['checks']['connectivity'] = {
                'status': 'error',
                'error': str(e),
                'healthy': False
            }
            health_status['overall_healthy'] = False
        
        # Check connection statistics
        stats = self.connection_manager.get_stats()
        health_status['checks']['connection_stats'] = {
            'status': 'ok',
            'healthy': True,
            'data': stats
        }
        
        # Check for high failure rate
        if stats['total_connections'] > 0:
            failure_rate = stats['failed_connections'] / stats['total_connections']
            if failure_rate > 0.1:  # More than 10% failures
                health_status['checks']['failure_rate'] = {
                    'status': 'warning',
                    'healthy': False,
                    'failure_rate': round(failure_rate, 3),
                    'message': 'High connection failure rate detected'
                }
                health_status['overall_healthy'] = False
        
        # Test query performance
        try:
            start_time = time.time()
            with self.connection_manager.get_session(read_only=True) as session:
                session.execute(text("SELECT COUNT(*) FROM plants"))
            query_time = time.time() - start_time
            
            health_status['checks']['query_performance'] = {
                'status': 'ok',
                'healthy': query_time < 1.0,
                'query_time': round(query_time, 3)
            }
            
            if query_time >= 1.0:
                health_status['overall_healthy'] = False
        
        except Exception as e:
            health_status['checks']['query_performance'] = {
                'status': 'error',
                'error': str(e),
                'healthy': False
            }
            health_status['overall_healthy'] = False
        
        self.last_check_time = now
        self.last_check_result = health_status
        return health_status


# Global instances
connection_manager = ConnectionManager()
session_manager = SessionManager(connection_manager)
performance_monitor = PerformanceMonitor()
health_checker = DatabaseHealthChecker(connection_manager)


# Convenience functions
def get_session(**kwargs) -> Generator[Session, None, None]:
    """Get database session with connection management."""
    return connection_manager.get_session(**kwargs)


def get_health_status(force_check: bool = False) -> Dict[str, Any]:
    """Get database health status."""
    return health_checker.check_health(force_check=force_check)


def get_performance_stats() -> Dict[str, Any]:
    """Get performance statistics."""
    return performance_monitor.get_performance_stats()