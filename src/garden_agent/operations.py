"""Core database operations for Garden Agent entities."""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.exc import IntegrityError

from .models import (
    Plant, Planting, Season, Conversation, Message, 
    KnowledgeBase, KnowledgeChunk, Import, 
    PlantEmbedding, ChunkEmbedding
)
from .database import get_db, calculate_content_hash


class PlantOperations:
    """CRUD operations for Plant entities."""
    
    @staticmethod
    def create(session: Session, **kwargs) -> Plant:
        """Create a new plant."""
        plant = Plant(**kwargs)
        session.add(plant)
        session.commit()
        session.refresh(plant)
        return plant
    
    @staticmethod
    def get_by_id(session: Session, plant_id: int) -> Optional[Plant]:
        """Get plant by ID."""
        return session.query(Plant).filter(Plant.id == plant_id).first()
    
    @staticmethod
    def get_by_name(session: Session, name: str, exact: bool = True) -> Optional[Plant]:
        """Get plant by name (exact or fuzzy match)."""
        if exact:
            return session.query(Plant).filter(Plant.name == name).first()
        else:
            return session.query(Plant).filter(Plant.name.ilike(f"%{name}%")).first()
    
    @staticmethod
    def search(
        session: Session, 
        query: Optional[str] = None,
        sun_requirement: Optional[str] = None,
        ph_range: Optional[tuple] = None,
        limit: int = 50
    ) -> List[Plant]:
        """Search plants with filters."""
        q = session.query(Plant)
        
        if query:
            q = q.filter(Plant.name.ilike(f"%{query}%"))
        
        if sun_requirement:
            q = q.filter(Plant.sun_requirement == sun_requirement)
        
        if ph_range:
            min_ph, max_ph = ph_range
            q = q.filter(
                and_(
                    Plant.soil_ph_min >= min_ph,
                    Plant.soil_ph_max <= max_ph
                )
            )
        
        return q.order_by(Plant.name).limit(limit).all()
    
    @staticmethod
    def update(session: Session, plant_id: int, **kwargs) -> Optional[Plant]:
        """Update plant by ID."""
        plant = session.query(Plant).filter(Plant.id == plant_id).first()
        if plant:
            for key, value in kwargs.items():
                if hasattr(plant, key):
                    setattr(plant, key, value)
            plant.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(plant)
        return plant
    
    @staticmethod
    def delete(session: Session, plant_id: int) -> bool:
        """Delete plant by ID."""
        plant = session.query(Plant).filter(Plant.id == plant_id).first()
        if plant:
            session.delete(plant)
            session.commit()
            return True
        return False
    
    @staticmethod
    def get_all(session: Session, limit: int = 100) -> List[Plant]:
        """Get all plants."""
        return session.query(Plant).order_by(Plant.name).limit(limit).all()
    
    @staticmethod
    def get_with_plantings(session: Session, plant_id: int) -> Optional[Plant]:
        """Get plant with its planting history."""
        return session.query(Plant).filter(Plant.id == plant_id).first()


class PlantingOperations:
    """CRUD operations for Planting entities."""
    
    @staticmethod
    def create(session: Session, **kwargs) -> Planting:
        """Create a new planting."""
        planting = Planting(**kwargs)
        session.add(planting)
        session.commit()
        session.refresh(planting)
        return planting
    
    @staticmethod
    def get_by_id(session: Session, planting_id: int) -> Optional[Planting]:
        """Get planting by ID."""
        return session.query(Planting).filter(Planting.id == planting_id).first()
    
    @staticmethod
    def get_by_plant(
        session: Session, 
        plant_id: int,
        year: Optional[int] = None,
        season: Optional[str] = None,
        limit: int = 20
    ) -> List[Planting]:
        """Get plantings for a specific plant."""
        q = session.query(Planting).filter(Planting.plant_id == plant_id)
        
        if year:
            q = q.filter(Planting.year == year)
        
        if season:
            q = q.filter(Planting.season == season)
        
        return q.order_by(desc(Planting.created_at)).limit(limit).all()
    
    @staticmethod
    def get_current_plantings(
        session: Session,
        year: Optional[int] = None,
        active_only: bool = True
    ) -> List[Planting]:
        """Get current/active plantings."""
        current_year = year or datetime.now().year
        q = session.query(Planting).filter(Planting.year == current_year)
        
        if active_only:
            # Consider plantings without harvest date as active
            q = q.filter(
                or_(
                    Planting.transplant_date.is_not(None),
                    Planting.seed_start_date.is_not(None)
                )
            )
        
        return q.order_by(desc(Planting.created_at)).all()
    
    @staticmethod
    def search_by_dates(
        session: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Planting]:
        """Search plantings by date range."""
        q = session.query(Planting)
        
        if start_date:
            q = q.filter(
                or_(
                    Planting.seed_start_date >= start_date,
                    Planting.transplant_date >= start_date
                )
            )
        
        if end_date:
            q = q.filter(
                or_(
                    Planting.seed_start_date <= end_date,
                    Planting.transplant_date <= end_date
                )
            )
        
        return q.order_by(desc(Planting.created_at)).limit(limit).all()
    
    @staticmethod
    def update(session: Session, planting_id: int, **kwargs) -> Optional[Planting]:
        """Update planting by ID."""
        planting = session.query(Planting).filter(Planting.id == planting_id).first()
        if planting:
            for key, value in kwargs.items():
                if hasattr(planting, key):
                    setattr(planting, key, value)
            planting.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(planting)
        return planting
    
    @staticmethod
    def delete(session: Session, planting_id: int) -> bool:
        """Delete planting by ID."""
        planting = session.query(Planting).filter(Planting.id == planting_id).first()
        if planting:
            session.delete(planting)
            session.commit()
            return True
        return False
    
    @staticmethod
    def append_notes(session: Session, planting_id: int, note: str) -> Optional[Planting]:
        """Append a timestamped note to planting."""
        planting = session.query(Planting).filter(Planting.id == planting_id).first()
        if planting:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            new_note = f"{timestamp}: {note}"
            
            if planting.notes:
                planting.notes = f"{planting.notes}\n{new_note}"
            else:
                planting.notes = new_note
            
            planting.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(planting)
        return planting


class ConversationOperations:
    """CRUD operations for Conversation and Message entities."""
    
    @staticmethod
    def create_conversation(session: Session, title: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return conversation
    
    @staticmethod
    def get_conversation(session: Session, conversation_id: int) -> Optional[Conversation]:
        """Get conversation by ID with messages."""
        return session.query(Conversation).filter(Conversation.id == conversation_id).first()
    
    @staticmethod
    def get_recent_conversations(session: Session, limit: int = 20) -> List[Conversation]:
        """Get recent conversations."""
        return session.query(Conversation).order_by(
            desc(Conversation.updated_at)
        ).limit(limit).all()
    
    @staticmethod
    def add_message(
        session: Session,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add message to conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_metadata=metadata
        )
        session.add(message)
        
        # Update conversation timestamp
        conversation = session.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if conversation:
            conversation.updated_at = datetime.utcnow()
        
        session.commit()
        session.refresh(message)
        return message
    
    @staticmethod
    def get_conversation_history(
        session: Session,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation message history."""
        q = session.query(Message).filter(Message.conversation_id == conversation_id)
        q = q.order_by(Message.created_at)
        
        if limit:
            q = q.limit(limit)
        
        return q.all()
    
    @staticmethod
    def delete_conversation(session: Session, conversation_id: int) -> bool:
        """Delete conversation and all messages."""
        conversation = session.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if conversation:
            session.delete(conversation)  # Messages deleted by cascade
            session.commit()
            return True
        return False
    
    @staticmethod
    def search_messages(
        session: Session,
        query: str,
        conversation_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Message]:
        """Search messages by content."""
        q = session.query(Message).filter(Message.content.ilike(f"%{query}%"))
        
        if conversation_id:
            q = q.filter(Message.conversation_id == conversation_id)
        
        return q.order_by(desc(Message.created_at)).limit(limit).all()


class KnowledgeOperations:
    """CRUD operations for Knowledge Base entities."""
    
    @staticmethod
    def create_document(
        session: Session,
        title: str,
        content: str,
        source: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeBase:
        """Create knowledge base document."""
        doc = KnowledgeBase(
            title=title,
            content=content,
            source=source,
            category=category,
            tags=tags or [],
            doc_metadata=metadata
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc
    
    @staticmethod
    def get_document(session: Session, doc_id: int) -> Optional[KnowledgeBase]:
        """Get knowledge document by ID."""
        return session.query(KnowledgeBase).filter(KnowledgeBase.id == doc_id).first()
    
    @staticmethod
    def search_documents(
        session: Session,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[KnowledgeBase]:
        """Search knowledge base documents."""
        q = session.query(KnowledgeBase)
        
        if query:
            q = q.filter(
                or_(
                    KnowledgeBase.title.ilike(f"%{query}%"),
                    KnowledgeBase.content.ilike(f"%{query}%")
                )
            )
        
        if category:
            q = q.filter(KnowledgeBase.category == category)
        
        # Note: Tags search would need JSON operations for SQLite
        # For now, skip tag filtering
        
        return q.order_by(desc(KnowledgeBase.created_at)).limit(limit).all()
    
    @staticmethod
    def create_chunk(
        session: Session,
        document_id: int,
        chunk_index: int,
        content: str,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeChunk:
        """Create knowledge chunk."""
        chunk = KnowledgeChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            chunk_metadata=metadata
        )
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
    
    @staticmethod
    def get_document_chunks(
        session: Session,
        document_id: int
    ) -> List[KnowledgeChunk]:
        """Get all chunks for a document."""
        return session.query(KnowledgeChunk).filter(
            KnowledgeChunk.document_id == document_id
        ).order_by(KnowledgeChunk.chunk_index).all()


class ImportOperations:
    """Operations for tracking imported content."""
    
    @staticmethod
    def is_already_imported(
        session: Session,
        title: str,
        created_date: str,
        content: str
    ) -> bool:
        """Check if content was already imported."""
        content_hash = calculate_content_hash(f"{title}{created_date}{content}")
        
        return session.query(Import).filter(
            Import.content_hash == content_hash
        ).first() is not None
    
    @staticmethod
    def mark_as_imported(
        session: Session,
        title: str,
        created_date: str,
        content: str
    ) -> Import:
        """Mark content as imported."""
        content_hash = calculate_content_hash(f"{title}{created_date}{content}")
        
        import_record = Import(
            title=title,
            created_date=created_date,
            note_content=content,
            content_hash=content_hash
        )
        session.add(import_record)
        session.commit()
        session.refresh(import_record)
        return import_record
    
    @staticmethod
    def get_import_history(
        session: Session,
        limit: int = 100
    ) -> List[Import]:
        """Get import history."""
        return session.query(Import).order_by(
            desc(Import.imported_at)
        ).limit(limit).all()


class AnalyticsOperations:
    """Database analytics and reporting operations."""
    
    @staticmethod
    def get_plant_statistics(session: Session) -> Dict[str, Any]:
        """Get plant statistics."""
        total_plants = session.query(Plant).count()
        total_plantings = session.query(Planting).count()
        
        # Current year plantings
        current_year = datetime.now().year
        current_plantings = session.query(Planting).filter(
            Planting.year == current_year
        ).count()
        
        # Most planted varieties
        popular_plants = session.query(
            Plant.name,
            func.count(Planting.id).label('planting_count')
        ).join(Planting).group_by(Plant.name).order_by(
            desc('planting_count')
        ).limit(5).all()
        
        return {
            'total_plants': total_plants,
            'total_plantings': total_plantings,
            'current_year_plantings': current_plantings,
            'popular_plants': [
                {'name': name, 'count': count} 
                for name, count in popular_plants
            ]
        }
    
    @staticmethod
    def get_seasonal_breakdown(
        session: Session,
        year: Optional[int] = None
    ) -> Dict[str, int]:
        """Get plantings by season."""
        target_year = year or datetime.now().year
        
        seasons = session.query(
            Planting.season,
            func.count(Planting.id).label('count')
        ).filter(
            Planting.year == target_year
        ).group_by(Planting.season).all()
        
        return {season: count for season, count in seasons}
    
    @staticmethod
    def get_knowledge_statistics(session: Session) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_docs = session.query(KnowledgeBase).count()
        total_chunks = session.query(KnowledgeChunk).count()
        
        categories = session.query(
            KnowledgeBase.category,
            func.count(KnowledgeBase.id).label('count')
        ).filter(
            KnowledgeBase.category.is_not(None)
        ).group_by(KnowledgeBase.category).all()
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'categories': [
                {'category': cat, 'count': count}
                for cat, count in categories
            ]
        }


class DatabaseOperations:
    """High-level database operations combining multiple entities."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
        self.plants = PlantOperations()
        self.plantings = PlantingOperations()
        self.conversations = ConversationOperations()
        self.knowledge = KnowledgeOperations()
        self.imports = ImportOperations()
        self.analytics = AnalyticsOperations()
    
    def get_plant_with_recent_plantings(
        self,
        plant_name: str,
        limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Get plant info with recent plantings."""
        plant = self.plants.get_by_name(self.session, plant_name)
        if not plant:
            return None
        
        recent_plantings = self.plantings.get_by_plant(
            self.session,
            plant.id,
            limit=limit
        )
        
        return {
            'plant': plant,
            'recent_plantings': recent_plantings
        }
    
    def create_garden_entry(
        self,
        plant_name: str,
        year: int,
        season: str = "Unknown",
        notes: Optional[str] = None,
        **planting_data
    ) -> Dict[str, Any]:
        """Create or update garden entry for a plant."""
        # Get or create plant
        plant = self.plants.get_by_name(self.session, plant_name)
        if not plant:
            plant = self.plants.create(self.session, name=plant_name)
        
        # Create planting
        planting_data.update({
            'plant_id': plant.id,
            'year': year,
            'season': season,
            'notes': notes
        })
        
        planting = self.plantings.create(self.session, **planting_data)
        
        return {
            'plant': plant,
            'planting': planting,
            'action': 'created'
        }


# Convenience function for getting database operations
def get_db_ops(session: Session) -> DatabaseOperations:
    """Get database operations instance."""
    return DatabaseOperations(session)