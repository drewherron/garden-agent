#!/usr/bin/env python3
"""Test database schema and operations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from garden_agent.database import db_manager
from garden_agent.models import Plant, Planting, KnowledgeBase, Conversation, Message
from datetime import datetime


def test_database_schema():
    """Test that all tables were created correctly."""
    print("🧪 Testing database schema...")
    
    session = db_manager.SessionLocal()
    try:
        # Test Plants table
        plants = session.query(Plant).all()
        print(f"✅ Plants table: {len(plants)} records")
        
        # Test Plantings table
        plantings = session.query(Planting).all() 
        print(f"✅ Plantings table: {len(plantings)} records")
        
        # Test KnowledgeBase table
        knowledge = session.query(KnowledgeBase).all()
        print(f"✅ KnowledgeBase table: {len(knowledge)} records")
        
        # Test Conversations table (should be empty)
        conversations = session.query(Conversation).all()
        print(f"✅ Conversations table: {len(conversations)} records")
        
        # Test Messages table (should be empty)
        messages = session.query(Message).all()
        print(f"✅ Messages table: {len(messages)} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False
    finally:
        session.close()


def test_basic_operations():
    """Test basic CRUD operations."""
    print("\n🧪 Testing basic operations...")
    
    session = db_manager.SessionLocal()
    try:
        # Test creating a conversation
        conversation = Conversation(title="Test Chat")
        session.add(conversation)
        session.commit()
        
        # Test creating messages
        msg1 = Message(
            conversation_id=conversation.id,
            role="user",
            content="What should I plant in spring?"
        )
        msg2 = Message(
            conversation_id=conversation.id,
            role="assistant", 
            content="Consider planting tomatoes, lettuce, and herbs in spring!"
        )
        
        session.add_all([msg1, msg2])
        session.commit()
        
        # Test querying
        conv_with_messages = session.query(Conversation).filter_by(id=conversation.id).first()
        if conv_with_messages:
            print(f"✅ Created conversation with {len(conv_with_messages.messages)} messages")
        
        # Test plant lookups
        tomato = session.query(Plant).filter(Plant.name.like("%Tomato%")).first()
        if tomato:
            print(f"✅ Found plant: {tomato.name} (ID: {tomato.id})")
            
            # Test related plantings
            tomato_plantings = session.query(Planting).filter_by(plant_id=tomato.id).all()
            print(f"✅ Tomato has {len(tomato_plantings)} planting records")
        
        return True
        
    except Exception as e:
        print(f"❌ Operations test failed: {e}")
        return False
    finally:
        session.close()


def test_indexes_and_performance():
    """Test that indexes are working."""
    print("\n🧪 Testing indexes and queries...")
    
    session = db_manager.SessionLocal()
    try:
        # Test indexed queries
        recent_plantings = session.query(Planting).filter(Planting.year == 2024).all()
        print(f"✅ Found {len(recent_plantings)} plantings for 2024")
        
        spring_plantings = session.query(Planting).filter(Planting.season == "Spring").all()
        print(f"✅ Found {len(spring_plantings)} spring plantings")
        
        # Test knowledge base category search
        gardening_docs = session.query(KnowledgeBase).filter(
            KnowledgeBase.category == "gardening techniques"
        ).all()
        print(f"✅ Found {len(gardening_docs)} gardening technique documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Index test failed: {e}")
        return False
    finally:
        session.close()


def main():
    """Run all database tests."""
    print("🌱 Garden Agent - Database Tests\n")
    
    # Check if database exists
    if not db_manager.health_check():
        print("❌ Database health check failed. Run init_db.py first.")
        return 1
    
    # Run tests
    tests = [
        test_database_schema,
        test_basic_operations,
        test_indexes_and_performance
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All database tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())