"""SQLAlchemy models for Garden Agent database."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Boolean, DateTime, Integer, String, Text, DECIMAL, ForeignKey, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Plant(Base):
    """Plant species and varieties."""
    __tablename__ = "plants"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    sun_requirement: Mapped[Optional[str]] = mapped_column(
        String(50), 
        nullable=True,
        comment="Full sun, Partial sun, Shade"
    )
    soil_ph_min: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 1), nullable=True)
    soil_ph_max: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 1), nullable=True)
    spacing_cm: Mapped[Optional[float]] = mapped_column(DECIMAL(5, 1), nullable=True)
    germination_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    maturity_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start_weeks_before_lf: Mapped[Optional[float]] = mapped_column(
        DECIMAL(3, 1), 
        nullable=True,
        comment="Weeks before last frost to start seeds indoors"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    plantings: Mapped[list["Planting"]] = relationship(
        "Planting", 
        back_populates="plant",
        cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["PlantEmbedding"]] = relationship(
        "PlantEmbedding",
        back_populates="plant",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Plant(id={self.id}, name='{self.name}')>"


class Season(Base):
    """Seasonal information for garden planning."""
    __tablename__ = "seasons"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    expected_last_frost: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    actual_last_frost: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    
    # Relationships (removed for now to simplify)
    
    __table_args__ = (
        Index('idx_seasons_year', 'year'),
    )
    
    def __repr__(self) -> str:
        return f"<Season(id={self.id}, year={self.year})>"


class Planting(Base):
    """Individual planting events and tracking."""
    __tablename__ = "plantings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plant_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("plants.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season: Mapped[str] = mapped_column(
        String(20), 
        nullable=False, 
        default="Unknown",
        comment="Winter, Spring, Summer, Fall, Unknown"
    )
    batch_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    seed_start_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    transplant_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    indoor: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    soil_ph: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 1), nullable=True)
    pests: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    disease: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fertilizer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    amendments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    plant: Mapped["Plant"] = relationship("Plant", back_populates="plantings")
    
    __table_args__ = (
        Index('idx_plantings_plant_year', 'plant_id', 'year'),
        Index('idx_plantings_season', 'season'),
    )
    
    def __repr__(self) -> str:
        return f"<Planting(id={self.id}, plant_id={self.plant_id}, year={self.year}, batch={self.batch_number})>"


class Conversation(Base):
    """User conversations and chat history."""
    __tablename__ = "conversations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class Message(Base):
    """Individual messages within conversations."""
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role: Mapped[str] = mapped_column(
        String(20), 
        nullable=False,
        comment="user, assistant, system"
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        index=True
    )
    
    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    __table_args__ = (
        Index('idx_messages_conversation_created', 'conversation_id', 'created_at'),
        Index('idx_messages_role', 'role'),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"


class KnowledgeBase(Base):
    """Knowledge base documents for RAG."""
    __tablename__ = "knowledge_base"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    doc_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        "KnowledgeChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_knowledge_category', 'category'),
        Index('idx_knowledge_created', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeBase(id={self.id}, title='{self.title}')>"


class KnowledgeChunk(Base):
    """Chunked content from knowledge base documents."""
    __tablename__ = "knowledge_chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    
    # Relationships
    document: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase",
        back_populates="chunks"
    )
    embeddings: Mapped[list["ChunkEmbedding"]] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_chunks_document_index', 'document_id', 'chunk_index'),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class PlantEmbedding(Base):
    """Vector embeddings for plant similarity search."""
    __tablename__ = "plant_embeddings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plant_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("plants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_data: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON array of floats")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    
    # Relationships
    plant: Mapped["Plant"] = relationship("Plant", back_populates="embeddings")
    
    __table_args__ = (
        Index('idx_plant_embeddings_model', 'embedding_model'),
    )
    
    def __repr__(self) -> str:
        return f"<PlantEmbedding(id={self.id}, plant_id={self.plant_id}, model='{self.embedding_model}')>"


class ChunkEmbedding(Base):
    """Vector embeddings for knowledge chunks."""
    __tablename__ = "chunk_embeddings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_data: Mapped[str] = mapped_column(Text, nullable=False, comment="JSON array of floats")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    
    # Relationships
    chunk: Mapped["KnowledgeChunk"] = relationship("KnowledgeChunk", back_populates="embeddings")
    
    __table_args__ = (
        Index('idx_chunk_embeddings_model', 'embedding_model'),
    )
    
    def __repr__(self) -> str:
        return f"<ChunkEmbedding(id={self.id}, chunk_id={self.chunk_id}, model='{self.embedding_model}')>"


class Import(Base):
    """Track imported notes and files to prevent duplicates."""
    __tablename__ = "imports"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_date: Mapped[str] = mapped_column(String(50), nullable=False)
    note_content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    
    __table_args__ = (
        Index('idx_imports_hash', 'content_hash'),
        Index('idx_imports_imported_at', 'imported_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Import(id={self.id}, title='{self.title}')>"