#!/usr/bin/env python3
"""Advanced setup script for Garden Agent."""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from garden_agent.setup import setup_manager
from garden_agent.migrations import migration_manager, generate_schema_migration
from garden_agent.database import db_manager


def cmd_init(args):
    """Initialize the Garden Agent setup."""
    print("🌱 Initializing Garden Agent...")
    
    setup_manager.run_full_setup(
        include_sample_data=args.samples,
        create_config=args.config,
        run_migrations=not args.no_migrations
    )


def cmd_validate(args):
    """Validate current setup."""
    print("🔍 Validating Garden Agent setup...\n")
    
    status = setup_manager.validate_setup()
    
    # Print directories status
    print("📁 Directories:")
    for dir_name, exists in status["directories"].items():
        icon = "✅" if exists else "❌"
        print(f"  {icon} {dir_name}")
    
    # Print database status
    print("\n🗄️ Database:")
    db_status = status["database"]
    
    accessible = db_status.get("accessible", False)
    print(f"  {'✅' if accessible else '❌'} Database accessible")
    
    file_exists = db_status.get("file_exists", False)
    print(f"  {'✅' if file_exists else '❌'} Database file exists")
    
    schema_ok = db_status.get("schema_up_to_date", False)
    print(f"  {'✅' if schema_ok else '❌'} Schema up to date")
    
    if not schema_ok and "schema_differences" in db_status:
        diffs = db_status["schema_differences"]
        if diffs.get("missing_tables"):
            print(f"    Missing tables: {', '.join(diffs['missing_tables'])}")
        if diffs.get("extra_tables"):
            print(f"    Extra tables: {', '.join(diffs['extra_tables'])}")
        if diffs.get("column_differences"):
            print(f"    Column differences in {len(diffs['column_differences'])} tables")
    
    # Print migrations status
    print("\n🔄 Migrations:")
    migrations = status["migrations"]
    if "error" in migrations:
        print(f"  ❌ Error: {migrations['error']}")
    else:
        current = migrations.get("current_version", "none")
        applied = migrations.get("applied_migrations", 0)
        pending = migrations.get("pending_migrations", 0)
        
        print(f"  📌 Current version: {current}")
        print(f"  ✅ Applied: {applied}")
        print(f"  ⏳ Pending: {pending}")
        
        if pending > 0:
            print("  Pending migrations:")
            for migration in migrations.get("pending_list", []):
                print(f"    - {migration['version']}: {migration['name']}")
    
    # Print dependencies status
    print("\n📦 Dependencies:")
    deps = status["dependencies"]
    
    ollama = deps.get("ollama_available", False)
    print(f"  {'✅' if ollama else '❌'} Ollama")
    
    sqlalchemy = deps.get("sqlalchemy", "missing")
    if sqlalchemy != "missing":
        print(f"  ✅ SQLAlchemy {sqlalchemy}")
    else:
        print("  ❌ SQLAlchemy missing")
    
    # Overall status
    overall = status["overall_status"]
    status_icons = {
        "healthy": "🟢",
        "needs_migration": "🟡", 
        "needs_database_init": "🟡",
        "needs_setup": "🔴"
    }
    
    icon = status_icons.get(overall, "❓")
    print(f"\n{icon} Overall Status: {overall}")
    
    if args.json:
        print(f"\n📄 Full status (JSON):")
        print(json.dumps(status, indent=2, default=str))


def cmd_migrate(args):
    """Handle database migrations."""
    if args.status:
        status = migration_manager.migration_status()
        print("🔄 Migration Status:")
        print(f"  Current version: {status.get('current_version', 'none')}")
        print(f"  Applied: {status.get('applied_migrations', 0)}")
        print(f"  Pending: {status.get('pending_migrations', 0)}")
        
        if status.get('pending_migrations', 0) > 0:
            print("\n  Pending migrations:")
            for migration in status.get('pending_list', []):
                print(f"    {migration['version']}: {migration['name']}")
        
        return
    
    if args.generate:
        print("🔧 Generating schema migration...")
        migration = generate_schema_migration()
        if migration:
            print(f"✅ Created migration: {migration.version}")
        return
    
    if args.up:
        target = args.target if args.target else None
        print(f"⬆️ Migrating up" + (f" to {target}" if target else ""))
        migration_manager.migrate_up(target)
        return
    
    if args.down and args.target:
        print(f"⬇️ Migrating down to {args.target}")
        migration_manager.migrate_down(args.target)
        return
    
    print("❓ No migration action specified. Use --help for options.")


def cmd_backup(args):
    """Handle database backups."""
    if args.create:
        backup_path = setup_manager.backup_database(args.name)
        if backup_path:
            print(f"✅ Backup created: {backup_path}")
        return
    
    if args.restore and args.file:
        try:
            setup_manager.restore_database(args.file)
            print("✅ Database restored successfully")
        except Exception as e:
            print(f"❌ Restore failed: {e}")
        return
    
    if args.list:
        backup_dir = Path("data/backups")
        if backup_dir.exists():
            backups = list(backup_dir.glob("*.db"))
            if backups:
                print("📦 Available backups:")
                for backup in sorted(backups):
                    size = backup.stat().st_size
                    size_mb = size / (1024 * 1024)
                    print(f"  {backup.name} ({size_mb:.1f} MB)")
            else:
                print("No backups found")
        else:
            print("Backup directory doesn't exist")
        return
    
    print("❓ No backup action specified. Use --help for options.")


def cmd_repair(args):
    """Repair setup issues."""
    setup_manager.repair_setup()


def cmd_clean(args):
    """Clean up data files."""
    setup_manager.cleanup_data(confirm=args.confirm)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Garden Agent Setup and Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize Garden Agent')
    init_parser.add_argument('--no-samples', dest='samples', action='store_false', 
                           help='Skip adding sample data')
    init_parser.add_argument('--no-config', dest='config', action='store_false',
                           help='Skip creating config file')
    init_parser.add_argument('--no-migrations', action='store_true',
                           help='Skip running migrations')
    init_parser.set_defaults(func=cmd_init)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate setup')
    validate_parser.add_argument('--json', action='store_true',
                               help='Output full status as JSON')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Manage migrations')
    migrate_parser.add_argument('--status', action='store_true',
                              help='Show migration status')
    migrate_parser.add_argument('--generate', action='store_true', 
                              help='Generate migration for schema changes')
    migrate_parser.add_argument('--up', action='store_true',
                              help='Apply pending migrations')
    migrate_parser.add_argument('--down', action='store_true',
                              help='Rollback to target version')
    migrate_parser.add_argument('--target', type=str,
                              help='Target migration version')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Manage backups')
    backup_parser.add_argument('--create', action='store_true',
                             help='Create database backup')
    backup_parser.add_argument('--restore', action='store_true',
                             help='Restore from backup')
    backup_parser.add_argument('--list', action='store_true',
                             help='List available backups')
    backup_parser.add_argument('--name', type=str,
                             help='Backup name')
    backup_parser.add_argument('--file', type=str,
                             help='Backup file to restore from')
    backup_parser.set_defaults(func=cmd_backup)
    
    # Repair command
    repair_parser = subparsers.add_parser('repair', help='Repair setup issues')
    repair_parser.set_defaults(func=cmd_repair)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up data files')
    clean_parser.add_argument('--confirm', action='store_true',
                            help='Confirm data deletion')
    clean_parser.set_defaults(func=cmd_clean)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())