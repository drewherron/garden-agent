#!/usr/bin/env python3
"""Test setup and migration utilities."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from garden_agent.setup import setup_manager
from garden_agent.migrations import migration_manager
from garden_agent.database import db_manager


def test_setup_validation():
    """Test setup validation."""
    print("🧪 Testing setup validation...")
    
    status = setup_manager.validate_setup()
    
    # Check that status has required keys
    required_keys = ["directories", "database", "migrations", "config", "dependencies", "overall_status"]
    for key in required_keys:
        assert key in status, f"Missing key in status: {key}"
    
    print("✅ Setup validation works")


def test_backup_restore():
    """Test backup and restore functionality."""
    print("🧪 Testing backup and restore...")
    
    # Create backup
    backup_path = setup_manager.backup_database("test_backup.db")
    assert backup_path, "Backup creation failed"
    assert os.path.exists(backup_path), f"Backup file not found: {backup_path}"
    
    print("✅ Backup creation works")
    
    # Note: We won't test restore as it would overwrite our current database


def test_migration_system():
    """Test migration system."""
    print("🧪 Testing migration system...")
    
    # Check migration status
    status = migration_manager.migration_status()
    assert "current_version" in status
    assert "applied_migrations" in status
    assert "pending_migrations" in status
    
    print(f"✅ Migration system works (current: {status['current_version']})")


def test_database_operations():
    """Test database operations."""
    print("🧪 Testing database operations...")
    
    # Health check
    assert db_manager.health_check(), "Database health check failed"
    
    # Test session creation
    session = db_manager.SessionLocal()
    try:
        # Simple query
        from sqlalchemy import text
        result = session.execute(text("SELECT 1")).fetchone()
        assert result[0] == 1, "Simple query failed"
    finally:
        session.close()
    
    print("✅ Database operations work")


def test_schema_comparison():
    """Test schema comparison."""
    print("🧪 Testing schema comparison...")
    
    from garden_agent.migrations import compare_schemas
    
    diff = compare_schemas(db_manager.engine)
    assert "missing_tables" in diff
    assert "extra_tables" in diff
    assert "column_differences" in diff
    assert "schema_matches" in diff
    
    print("✅ Schema comparison works")


def test_directory_creation():
    """Test directory creation."""
    print("🧪 Testing directory creation...")
    
    # This should not fail even if directories exist
    setup_manager.ensure_directories()
    
    # Check some key directories exist
    from pathlib import Path
    data_dir = Path("data")
    assert data_dir.exists(), "Data directory not created"
    assert (data_dir / "migrations").exists(), "Migrations directory not created"
    
    print("✅ Directory creation works")


def main():
    """Run all tests."""
    print("🌱 Garden Agent - Setup Utilities Test\n")
    
    tests = [
        test_setup_validation,
        test_backup_restore,
        test_migration_system,
        test_database_operations,
        test_schema_comparison,
        test_directory_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All setup utility tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())