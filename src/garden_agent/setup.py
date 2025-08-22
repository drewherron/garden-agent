"""Advanced setup utilities for Garden Agent."""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .database import db_manager, init_database, add_sample_data
from .migrations import migration_manager, compare_schemas, generate_schema_migration
from .config import default_config


class SetupManager:
    """Manages Garden Agent setup and configuration."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        
    def ensure_directories(self) -> None:
        """Create required directories."""
        directories = [
            self.data_dir,
            self.data_dir / "migrations",
            self.data_dir / "chromadb", 
            self.data_dir / "uploads",
            self.data_dir / "exports",
            self.data_dir / "backups",
            self.data_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print("✅ Created required directories")
    
    def create_config_file(self, config_path: Optional[str] = None) -> None:
        """Create configuration file with current settings."""
        if config_path is None:
            config_path = self.project_root / "garden_config.json"
        
        config_data = {
            "database": {
                "path": default_config.database.path,
                "echo": default_config.database.echo
            },
            "llm": {
                "model": default_config.llm.model,
                "timeout": default_config.llm.timeout,
                "temperature": default_config.llm.temperature,
                "max_retries": default_config.llm.max_retries
            },
            "vector": {
                "persist_directory": default_config.vector.persist_directory,
                "collection_name": default_config.vector.collection_name,
                "embedding_model": default_config.vector.embedding_model
            },
            "setup": {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "features_enabled": {
                    "chat_interface": True,
                    "vector_search": True,
                    "rag_system": True,
                    "file_uploads": True,
                    "analytics": True
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Created configuration file: {config_path}")
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create database backup."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"garden_backup_{timestamp}.db"
        
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        source_db = Path(default_config.database.path)
        backup_path = backup_dir / backup_name
        
        if source_db.exists():
            shutil.copy2(source_db, backup_path)
            print(f"✅ Database backed up to: {backup_path}")
            return str(backup_path)
        else:
            print("⚠️ No database file found to backup")
            return ""
    
    def restore_database(self, backup_path: str) -> None:
        """Restore database from backup."""
        backup_file = Path(backup_path)
        target_db = Path(default_config.database.path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Create backup of current database if it exists
        if target_db.exists():
            current_backup = self.backup_database("pre_restore_backup.db")
            print(f"📦 Current database backed up to: {current_backup}")
        
        # Restore from backup
        shutil.copy2(backup_file, target_db)
        print(f"✅ Database restored from: {backup_path}")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate current setup and return status."""
        status = {
            "directories": {},
            "database": {},
            "migrations": {},
            "config": {},
            "dependencies": {},
            "overall_status": "unknown"
        }
        
        # Check directories
        required_dirs = ["data", "data/migrations", "data/chromadb"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            status["directories"][dir_name] = dir_path.exists()
        
        # Check database
        try:
            status["database"]["accessible"] = db_manager.health_check()
            status["database"]["file_exists"] = Path(default_config.database.path).exists()
            
            # Check schema
            schema_diff = compare_schemas(db_manager.engine)
            status["database"]["schema_up_to_date"] = schema_diff["schema_matches"]
            status["database"]["schema_differences"] = schema_diff
            
        except Exception as e:
            status["database"]["error"] = str(e)
            status["database"]["accessible"] = False
        
        # Check migrations
        try:
            migration_status = migration_manager.migration_status()
            status["migrations"] = migration_status
        except Exception as e:
            status["migrations"]["error"] = str(e)
        
        # Check configuration
        config_file = self.project_root / "garden_config.json"
        status["config"]["file_exists"] = config_file.exists()
        
        # Check dependencies
        status["dependencies"]["ollama_available"] = shutil.which("ollama") is not None
        
        try:
            import sqlalchemy
            status["dependencies"]["sqlalchemy"] = sqlalchemy.__version__
        except ImportError:
            status["dependencies"]["sqlalchemy"] = "missing"
        
        # Determine overall status
        all_dirs_exist = all(status["directories"].values())
        db_healthy = status["database"].get("accessible", False)
        schema_ok = status["database"].get("schema_up_to_date", False)
        
        if all_dirs_exist and db_healthy and schema_ok:
            status["overall_status"] = "healthy"
        elif all_dirs_exist and db_healthy:
            status["overall_status"] = "needs_migration"
        elif all_dirs_exist:
            status["overall_status"] = "needs_database_init"
        else:
            status["overall_status"] = "needs_setup"
        
        return status
    
    def run_full_setup(
        self, 
        include_sample_data: bool = True,
        create_config: bool = True,
        run_migrations: bool = True
    ) -> None:
        """Run complete setup process."""
        print("🌱 Garden Agent - Full Setup")
        print("=" * 40)
        
        # Step 1: Create directories
        print("\n📁 Creating directories...")
        self.ensure_directories()
        
        # Step 2: Create configuration
        if create_config:
            print("\n⚙️ Creating configuration...")
            self.create_config_file()
        
        # Step 3: Initialize database
        print("\n🗄️ Initializing database...")
        init_database(drop_existing=False)
        
        # Step 4: Run migrations
        if run_migrations:
            print("\n🔄 Running migrations...")
            migration_manager.migrate_up()
        
        # Step 5: Add sample data
        if include_sample_data:
            print("\n📦 Adding sample data...")
            add_sample_data()
        
        # Step 6: Validate setup
        print("\n✅ Validating setup...")
        status = self.validate_setup()
        
        if status["overall_status"] == "healthy":
            print("🎉 Setup completed successfully!")
        else:
            print(f"⚠️ Setup completed with status: {status['overall_status']}")
            print("Run 'python scripts/setup.py --validate' for details")
    
    def repair_setup(self) -> None:
        """Attempt to repair common setup issues."""
        print("🔧 Garden Agent - Setup Repair")
        print("=" * 30)
        
        status = self.validate_setup()
        
        # Fix missing directories
        if not all(status["directories"].values()):
            print("📁 Fixing missing directories...")
            self.ensure_directories()
        
        # Fix database issues
        if not status["database"].get("accessible", False):
            print("🗄️ Fixing database issues...")
            try:
                init_database(drop_existing=False)
            except Exception as e:
                print(f"❌ Database repair failed: {e}")
        
        # Fix schema issues
        if not status["database"].get("schema_up_to_date", True):
            print("🔄 Fixing schema issues...")
            try:
                migration = generate_schema_migration()
                if migration:
                    migration_manager.apply_migration(migration)
            except Exception as e:
                print(f"❌ Schema repair failed: {e}")
        
        # Re-validate
        new_status = self.validate_setup()
        if new_status["overall_status"] == "healthy":
            print("✅ Setup repair completed successfully!")
        else:
            print(f"⚠️ Some issues remain: {new_status['overall_status']}")
    
    def cleanup_data(self, confirm: bool = False) -> None:
        """Clean up data files and reset to fresh state."""
        if not confirm:
            print("⚠️ This will delete all data! Use --confirm flag to proceed.")
            return
        
        print("🧹 Cleaning up data files...")
        
        # Remove database
        db_path = Path(default_config.database.path)
        if db_path.exists():
            db_path.unlink()
            print("✅ Removed database file")
        
        # Remove vector database
        vector_dir = Path(default_config.vector.persist_directory)
        if vector_dir.exists():
            shutil.rmtree(vector_dir)
            print("✅ Removed vector database")
        
        # Clear migrations tracking
        try:
            with db_manager.engine.connect() as conn:
                conn.execute("DROP TABLE IF EXISTS schema_migrations")
                conn.commit()
        except:
            pass  # Table might not exist
        
        print("✅ Data cleanup completed")


# Global setup manager instance
setup_manager = SetupManager()