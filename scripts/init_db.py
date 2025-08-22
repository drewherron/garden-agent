#!/usr/bin/env python3
"""Initialize the Garden Agent database."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from garden_agent.database import init_database, add_sample_data, db_manager


def main():
    """Initialize database with optional sample data."""
    print("🌱 Garden Agent - Database Initialization\n")
    
    # Parse command line arguments
    drop_existing = "--drop" in sys.argv or "--reset" in sys.argv
    add_samples = "--samples" in sys.argv or "--reset" in sys.argv
    
    if drop_existing:
        print("⚠️  WARNING: This will drop all existing data!")
        if "--force" not in sys.argv:
            confirm = input("Continue? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return 1
    
    try:
        # Initialize database
        init_database(drop_existing=drop_existing)
        
        # Add sample data if requested
        if add_samples:
            add_sample_data()
        
        # Health check
        if db_manager.health_check():
            print("\n✅ Database initialization successful!")
            print(f"📁 Database location: {db_manager.engine.url}")
            return 0
        else:
            print("\n❌ Database health check failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        return 1


def show_help():
    """Show usage information."""
    print("""
Usage: python scripts/init_db.py [options]

Options:
  --drop      Drop existing tables before creating new ones
  --samples   Add sample data for testing
  --reset     Equivalent to --drop --samples
  --force     Skip confirmation prompts
  
Examples:
  python scripts/init_db.py                    # Create tables only
  python scripts/init_db.py --samples          # Create tables + sample data
  python scripts/init_db.py --reset --force    # Full reset with sample data
    """)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    sys.exit(main())