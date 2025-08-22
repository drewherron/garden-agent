"""Database connection and initialization utilities."""

import os
import hashlib
from typing import Optional, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base
from .config import default_config


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        """Initialize database manager.
        
        Args:
            database_url: Database URL (defaults to config)
            echo: Enable SQLAlchemy echo mode
        """
        if database_url is None:
            db_path = default_config.database.path
            # Ensure directory exists
            db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "."
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{db_path}"
        
        # Configure engine for SQLite
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        
        self.engine = create_engine(
            database_url,
            echo=echo or default_config.database.echo,
            connect_args=connect_args,
            poolclass=StaticPool if database_url.startswith("sqlite") else None
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        
    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
            
    def health_check(self) -> bool:
        """Check if database is accessible."""
        try:
            session = self.SessionLocal()
            try:
                session.execute(text("SELECT 1"))
                return True
            finally:
                session.close()
        except Exception:
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection for database sessions."""
    yield from db_manager.get_session()


def init_database(drop_existing: bool = False) -> None:
    """Initialize the database with tables and optional sample data.
    
    Args:
        drop_existing: Whether to drop existing tables first
    """
    if drop_existing:
        print("Dropping existing tables...")
        db_manager.drop_tables()
    
    print("Creating database tables...")
    db_manager.create_tables()
    
    print("Database initialization complete!")


def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def add_sample_data() -> None:
    """Add sample data for testing and development."""
    from .models import Plant, Season, Planting, KnowledgeBase
    from datetime import datetime
    
    session = db_manager.SessionLocal()
    try:
        # Check if data already exists
        if session.query(Plant).count() > 0:
            print("Sample data already exists, skipping...")
            return
        
        print("Adding sample data...")
        
        # Sample plants
        tomato = Plant(
            name="Roma Tomato",
            sun_requirement="Full sun",
            soil_ph_min=6.0,
            soil_ph_max=7.0,
            spacing_cm=45.0,
            germination_days=7,
            maturity_days=75,
            start_weeks_before_lf=6.0
        )
        
        basil = Plant(
            name="Sweet Basil",
            sun_requirement="Full sun",
            soil_ph_min=6.0,
            soil_ph_max=7.5,
            spacing_cm=20.0,
            germination_days=5,
            maturity_days=60,
            start_weeks_before_lf=8.0
        )
        
        lettuce = Plant(
            name="Butterhead Lettuce",
            sun_requirement="Partial sun",
            soil_ph_min=6.0,
            soil_ph_max=7.0,
            spacing_cm=15.0,
            germination_days=4,
            maturity_days=50,
            start_weeks_before_lf=4.0
        )
        
        session.add_all([tomato, basil, lettuce])
        session.commit()
        
        # Sample season data
        season_2024 = Season(
            year=2024,
            expected_last_frost=datetime(2024, 4, 15),
            actual_last_frost=datetime(2024, 4, 12)
        )
        
        session.add(season_2024)
        session.commit()
        
        # Sample plantings
        tomato_planting = Planting(
            plant_id=tomato.id,
            year=2024,
            season="Spring",
            batch_number=1,
            seed_start_date=datetime(2024, 3, 1),
            transplant_date=datetime(2024, 4, 20),
            indoor=True,
            fertilizer="Organic compost",
            notes="Started indoors under grow light"
        )
        
        session.add(tomato_planting)
        session.commit()
        
        # Sample knowledge base entry
        companion_planting = KnowledgeBase(
            title="Companion Planting Guide",
            content="""
            Companion planting is the practice of growing different plants together 
            for mutual benefit. Here are some classic combinations:
            
            - Tomatoes + Basil: Basil may improve tomato flavor and repel pests
            - Lettuce + Tomatoes: Lettuce grows well in tomato shade
            - Three Sisters: Corn, beans, and squash grow together symbiotically
            
            Benefits include pest control, improved soil health, and space efficiency.
            """,
            category="gardening techniques",
            tags=["companion planting", "pest control", "garden planning"]
        )
        
        session.add(companion_planting)
        session.commit()
        
        print(f"Added sample data: {session.query(Plant).count()} plants, "
              f"{session.query(Planting).count()} plantings, "
              f"{session.query(KnowledgeBase).count()} knowledge entries")
    
    finally:
        session.close()


if __name__ == "__main__":
    # Initialize database when run directly
    import sys
    
    drop_existing = "--drop" in sys.argv or "--reset" in sys.argv
    add_samples = "--samples" in sys.argv or "--reset" in sys.argv
    
    init_database(drop_existing=drop_existing)
    
    if add_samples:
        add_sample_data()
    
    # Health check
    if db_manager.health_check():
        print("✅ Database health check passed!")
    else:
        print("❌ Database health check failed!")
        sys.exit(1)