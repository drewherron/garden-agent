"""Database migration utilities for Garden Agent."""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy import text, inspect, MetaData
from sqlalchemy.engine import Engine

from .database import db_manager
from .models import Base


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    created_at: datetime


class MigrationManager:
    """Manages database migrations and schema versioning."""
    
    def __init__(self, migrations_dir: str = "data/migrations"):
        """Initialize migration manager.
        
        Args:
            migrations_dir: Directory to store migration files
        """
        self.migrations_dir = migrations_dir
        self.engine = db_manager.engine
        
        # Ensure migrations directory exists
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Create migrations table if it doesn't exist
        self._create_migrations_table()
    
    def _create_migrations_table(self) -> None:
        """Create the migrations tracking table."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    
    def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT version FROM schema_migrations 
                ORDER BY applied_at DESC 
                LIMIT 1
            """)).fetchone()
            return result[0] if result else None
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT version FROM schema_migrations 
                ORDER BY applied_at ASC
            """)).fetchall()
            return [row[0] for row in result]
    
    def load_migration_files(self) -> List[Migration]:
        """Load migration files from disk."""
        migrations = []
        
        if not os.path.exists(self.migrations_dir):
            return migrations
        
        for filename in sorted(os.listdir(self.migrations_dir)):
            if filename.endswith('.json'):
                filepath = os.path.join(self.migrations_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    migration = Migration(
                        version=data['version'],
                        name=data['name'],
                        description=data['description'],
                        up_sql=data['up_sql'],
                        down_sql=data['down_sql'],
                        created_at=datetime.fromisoformat(data['created_at'])
                    )
                    migrations.append(migration)
        
        return migrations
    
    def save_migration(self, migration: Migration) -> None:
        """Save migration to disk."""
        filename = f"{migration.version}_{migration.name.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.migrations_dir, filename)
        
        data = {
            'version': migration.version,
            'name': migration.name,
            'description': migration.description,
            'up_sql': migration.up_sql,
            'down_sql': migration.down_sql,
            'created_at': migration.created_at.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_migration(
        self, 
        name: str, 
        description: str = "",
        up_sql: str = "",
        down_sql: str = ""
    ) -> Migration:
        """Create a new migration file."""
        # Generate version timestamp
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        migration = Migration(
            version=version,
            name=name,
            description=description,
            up_sql=up_sql,
            down_sql=down_sql,
            created_at=datetime.now()
        )
        
        self.save_migration(migration)
        print(f"Created migration: {version}_{name}")
        
        return migration
    
    def apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        print(f"Applying migration {migration.version}: {migration.name}")
        
        with self.engine.connect() as conn:
            # Execute migration SQL
            if migration.up_sql.strip():
                for statement in migration.up_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        conn.execute(text(statement))
            
            # Record migration as applied
            conn.execute(text("""
                INSERT INTO schema_migrations (version, name, description)
                VALUES (:version, :name, :description)
            """), {
                "version": migration.version, 
                "name": migration.name, 
                "description": migration.description
            })
            
            conn.commit()
        
        print(f"✅ Applied migration {migration.version}")
    
    def rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration."""
        print(f"Rolling back migration {migration.version}: {migration.name}")
        
        with self.engine.connect() as conn:
            # Execute rollback SQL
            if migration.down_sql.strip():
                for statement in migration.down_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        conn.execute(text(statement))
            
            # Remove migration record
            conn.execute(text("""
                DELETE FROM schema_migrations WHERE version = :version
            """), {"version": migration.version})
            
            conn.commit()
        
        print(f"✅ Rolled back migration {migration.version}")
    
    def migrate_up(self, target_version: Optional[str] = None) -> None:
        """Apply migrations up to target version (or latest)."""
        migrations = self.load_migration_files()
        applied = set(self.get_applied_migrations())
        
        # Filter to unapplied migrations
        pending = [m for m in migrations if m.version not in applied]
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            print("No migrations to apply")
            return
        
        for migration in sorted(pending, key=lambda m: m.version):
            self.apply_migration(migration)
        
        print(f"✅ Applied {len(pending)} migrations")
    
    def migrate_down(self, target_version: str) -> None:
        """Rollback migrations to target version."""
        migrations = self.load_migration_files()
        applied = self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version in reversed(applied):
            if version > target_version:
                migration = next((m for m in migrations if m.version == version), None)
                if migration:
                    to_rollback.append(migration)
        
        if not to_rollback:
            print(f"Already at version {target_version}")
            return
        
        for migration in to_rollback:
            self.rollback_migration(migration)
        
        print(f"✅ Rolled back {len(to_rollback)} migrations")
    
    def migration_status(self) -> Dict[str, Any]:
        """Get migration status information."""
        migrations = self.load_migration_files()
        applied = set(self.get_applied_migrations())
        current_version = self.get_current_version()
        
        pending = [m for m in migrations if m.version not in applied]
        
        return {
            'current_version': current_version,
            'total_migrations': len(migrations),
            'applied_migrations': len(applied),
            'pending_migrations': len(pending),
            'pending_list': [{'version': m.version, 'name': m.name} for m in pending]
        }


def compare_schemas(engine: Engine) -> Dict[str, Any]:
    """Compare current database schema with SQLAlchemy models."""
    inspector = inspect(engine)
    
    # Get current database tables
    db_tables = set(inspector.get_table_names())
    
    # Get model tables
    model_tables = set(Base.metadata.tables.keys())
    
    # Compare
    missing_tables = model_tables - db_tables
    extra_tables = db_tables - model_tables
    
    # Check columns for existing tables
    column_differences = {}
    for table_name in model_tables.intersection(db_tables):
        db_columns = {col['name']: col for col in inspector.get_columns(table_name)}
        model_table = Base.metadata.tables[table_name]
        model_columns = {col.name: col for col in model_table.columns}
        
        missing_cols = set(model_columns.keys()) - set(db_columns.keys())
        extra_cols = set(db_columns.keys()) - set(model_columns.keys())
        
        if missing_cols or extra_cols:
            column_differences[table_name] = {
                'missing_columns': list(missing_cols),
                'extra_columns': list(extra_cols)
            }
    
    return {
        'missing_tables': list(missing_tables),
        'extra_tables': list(extra_tables),
        'column_differences': column_differences,
        'schema_matches': len(missing_tables) == 0 and len(extra_tables) == 0 and len(column_differences) == 0
    }


def generate_schema_migration() -> Optional[Migration]:
    """Generate migration to sync database with current models."""
    differences = compare_schemas(db_manager.engine)
    
    if differences['schema_matches']:
        print("Schema is already up to date")
        return None
    
    # Generate SQL for differences
    up_sql_parts = []
    down_sql_parts = []
    
    # Handle missing tables
    for table_name in differences['missing_tables']:
        table = Base.metadata.tables[table_name]
        create_sql = str(table.compile(db_manager.engine)).strip()
        up_sql_parts.append(create_sql)
        down_sql_parts.append(f"DROP TABLE {table_name}")
    
    # Handle column differences (simplified)
    for table_name, diffs in differences['column_differences'].items():
        for col_name in diffs['missing_columns']:
            table = Base.metadata.tables[table_name]
            column = table.columns[col_name]
            col_type = str(column.type.compile(db_manager.engine))
            
            nullable = "NULL" if column.nullable else "NOT NULL"
            default = f"DEFAULT {column.default.arg}" if column.default else ""
            
            up_sql_parts.append(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} {nullable} {default}".strip())
            down_sql_parts.append(f"ALTER TABLE {table_name} DROP COLUMN {col_name}")
    
    if not up_sql_parts:
        print("No actionable schema differences found")
        return None
    
    # Create migration
    manager = MigrationManager()
    migration = manager.create_migration(
        name="sync_schema_with_models",
        description="Sync database schema with current SQLAlchemy models",
        up_sql=";\n".join(up_sql_parts),
        down_sql=";\n".join(reversed(down_sql_parts))
    )
    
    return migration


# Global migration manager instance
migration_manager = MigrationManager()