"""Vector storage with ChromaDB for semantic similarity search."""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from .models import Plant, KnowledgeBase, PlantEmbedding, ChunkEmbedding
from .connection import get_session

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding manager.
        
        Args:
            model_name: Sentence-transformers model name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to encode
            
        Returns:
            Numpy array embedding
        """
        return self.encode([text])[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if not self.model:
            return 384  # Default for all-MiniLM-L6-v2
        return self.model.get_sentence_embedding_dimension()


class ChromaDBManager:
    """Manages ChromaDB collections for vector storage."""
    
    def __init__(self, data_dir: str = "data/chroma"):
        """Initialize ChromaDB manager.
        
        Args:
            data_dir: Directory for ChromaDB persistence
        """
        self.data_dir = data_dir
        self.client = None
        self.collections = {}
        self._setup_client()
    
    def _setup_client(self):
        """Set up ChromaDB client."""
        try:
            # Ensure data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.data_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized with data dir: {self.data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def get_or_create_collection(
        self, 
        name: str, 
        embedding_dimension: int = 384,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Get or create a ChromaDB collection.
        
        Args:
            name: Collection name
            embedding_dimension: Embedding vector dimension
            metadata: Collection metadata
            
        Returns:
            ChromaDB collection object
        """
        if name in self.collections:
            return self.collections[name]
        
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
            self.collections[name] = collection
            logger.info(f"Collection '{name}' ready with {collection.count()} documents")
            return collection
        except Exception as e:
            logger.error(f"Failed to get/create collection '{name}': {e}")
            raise
    
    def delete_collection(self, name: str):
        """Delete a collection.
        
        Args:
            name: Collection name to delete
        """
        try:
            if name in self.collections:
                del self.collections[name]
            self.client.delete_collection(name=name)
            logger.info(f"Collection '{name}' deleted")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{name}': {e}")
    
    def reset_database(self):
        """Reset the entire ChromaDB database."""
        try:
            self.client.reset()
            self.collections.clear()
            logger.info("ChromaDB database reset")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB database: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            stats = {
                'data_directory': self.data_dir,
                'collections': {}
            }
            
            for name, collection in self.collections.items():
                stats['collections'][name] = {
                    'count': collection.count(),
                    'metadata': collection.metadata
                }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {'error': str(e)}


class PlantVectorStore:
    """Vector storage for plant similarity search."""
    
    def __init__(self, chroma_manager: ChromaDBManager, embedding_manager: EmbeddingManager):
        """Initialize plant vector store.
        
        Args:
            chroma_manager: ChromaDB manager instance
            embedding_manager: Embedding manager instance
        """
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self.collection_name = "plants"
        self.collection = None
        self._setup_collection()
    
    def _setup_collection(self):
        """Set up the plants collection."""
        self.collection = self.chroma_manager.get_or_create_collection(
            self.collection_name,
            embedding_dimension=self.embedding_manager.get_dimension(),
            metadata={"description": "Plant similarity search collection"}
        )
    
    def _create_plant_text(self, plant: Plant) -> str:
        """Create searchable text representation of a plant.
        
        Args:
            plant: Plant model instance
            
        Returns:
            Concatenated text for embedding
        """
        parts = [plant.name]
        
        if plant.sun_requirement:
            parts.append(f"Sun: {plant.sun_requirement}")
        
        if plant.soil_ph_min and plant.soil_ph_max:
            parts.append(f"pH: {plant.soil_ph_min}-{plant.soil_ph_max}")
        
        if plant.germination_days:
            parts.append(f"Germination: {plant.germination_days} days")
        
        if plant.maturity_days:
            parts.append(f"Maturity: {plant.maturity_days} days")
        
        return " | ".join(parts)
    
    def add_plant(self, plant: Plant) -> bool:
        """Add a plant to the vector store.
        
        Args:
            plant: Plant instance to add
            
        Returns:
            True if successful
        """
        try:
            # Create searchable text
            plant_text = self._create_plant_text(plant)
            
            # Generate embedding
            embedding = self.embedding_manager.encode_single(plant_text)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[plant_text],
                metadatas=[{
                    'plant_id': int(plant.id),
                    'plant_name': plant.name,
                    'sun_requirement': plant.sun_requirement or '',
                    'soil_ph_min': float(plant.soil_ph_min) if plant.soil_ph_min else 0.0,
                    'soil_ph_max': float(plant.soil_ph_max) if plant.soil_ph_max else 0.0,
                    'added_at': datetime.now().isoformat()
                }],
                ids=[f"plant_{plant.id}"]
            )
            
            logger.info(f"Added plant '{plant.name}' to vector store")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add plant '{plant.name}' to vector store: {e}")
            return False
    
    def add_plants_batch(self, plants: List[Plant]) -> int:
        """Add multiple plants to the vector store.
        
        Args:
            plants: List of Plant instances
            
        Returns:
            Number of plants successfully added
        """
        if not plants:
            return 0
        
        try:
            # Prepare data
            texts = [self._create_plant_text(plant) for plant in plants]
            embeddings = self.embedding_manager.encode(texts)
            
            metadatas = []
            ids = []
            
            for plant in plants:
                metadatas.append({
                    'plant_id': int(plant.id),
                    'plant_name': plant.name,
                    'sun_requirement': plant.sun_requirement or '',
                    'soil_ph_min': float(plant.soil_ph_min) if plant.soil_ph_min else 0.0,
                    'soil_ph_max': float(plant.soil_ph_max) if plant.soil_ph_max else 0.0,
                    'added_at': datetime.now().isoformat()
                })
                ids.append(f"plant_{plant.id}")
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(plants)} plants to vector store")
            return len(plants)
        
        except Exception as e:
            logger.error(f"Failed to add plants batch to vector store: {e}")
            return 0
    
    def search_similar_plants(
        self, 
        query: str, 
        n_results: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar plants using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar plants with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_single(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Process results
            similar_plants = []
            
            for i, (doc_id, distance, metadata, document) in enumerate(
                zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0],
                    results['documents'][0]
                )
            ):
                # Convert distance to similarity (ChromaDB uses L2 distance)
                similarity = 1.0 / (1.0 + distance)
                
                if similarity >= min_similarity:
                    similar_plants.append({
                        'plant_id': metadata['plant_id'],
                        'plant_name': metadata['plant_name'],
                        'similarity': round(similarity, 3),
                        'distance': round(distance, 3),
                        'document': document,
                        'metadata': metadata
                    })
            
            logger.info(f"Found {len(similar_plants)} similar plants for query: '{query}'")
            return similar_plants
        
        except Exception as e:
            logger.error(f"Failed to search similar plants: {e}")
            return []
    
    def update_plant(self, plant: Plant) -> bool:
        """Update a plant in the vector store.
        
        Args:
            plant: Updated plant instance
            
        Returns:
            True if successful
        """
        try:
            # Remove existing
            self.remove_plant(plant.id)
            
            # Add updated version
            return self.add_plant(plant)
        
        except Exception as e:
            logger.error(f"Failed to update plant '{plant.name}': {e}")
            return False
    
    def remove_plant(self, plant_id: int) -> bool:
        """Remove a plant from the vector store.
        
        Args:
            plant_id: ID of plant to remove
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[f"plant_{plant_id}"])
            logger.info(f"Removed plant {plant_id} from vector store")
            return True
        
        except Exception as e:
            logger.error(f"Failed to remove plant {plant_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plant vector store statistics."""
        return {
            'collection_name': self.collection_name,
            'total_plants': self.collection.count(),
            'embedding_dimension': self.embedding_manager.get_dimension()
        }


class KnowledgeVectorStore:
    """Vector storage for knowledge base documents."""
    
    def __init__(self, chroma_manager: ChromaDBManager, embedding_manager: EmbeddingManager):
        """Initialize knowledge vector store.
        
        Args:
            chroma_manager: ChromaDB manager instance
            embedding_manager: Embedding manager instance
        """
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self.collection_name = "knowledge"
        self.collection = None
        self._setup_collection()
    
    def _setup_collection(self):
        """Set up the knowledge collection."""
        self.collection = self.chroma_manager.get_or_create_collection(
            self.collection_name,
            embedding_dimension=self.embedding_manager.get_dimension(),
            metadata={"description": "Knowledge base retrieval collection"}
        )
    
    def add_document_chunk(
        self, 
        doc_id: int,
        chunk_id: int,
        chunk_text: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a document chunk to the vector store.
        
        Args:
            doc_id: Document ID
            chunk_id: Chunk ID
            chunk_text: Chunk content
            chunk_index: Chunk position in document
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Generate embedding
            embedding = self.embedding_manager.encode_single(chunk_text)
            
            # Prepare metadata
            chunk_metadata = {
                'doc_id': int(doc_id),
                'chunk_id': int(chunk_id),
                'chunk_index': int(chunk_index),
                'added_at': datetime.now().isoformat()
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[chunk_text],
                metadatas=[chunk_metadata],
                ids=[f"chunk_{chunk_id}"]
            )
            
            logger.info(f"Added chunk {chunk_id} from document {doc_id} to knowledge store")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add chunk {chunk_id}: {e}")
            return False
    
    def search_knowledge(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant chunks.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            category_filter: Optional category filter
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.encode_single(query)
            
            # Build where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause['category'] = category_filter
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Process results
            knowledge_chunks = []
            
            for i, (chunk_id, distance, metadata, document) in enumerate(
                zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0],
                    results['documents'][0]
                )
            ):
                similarity = 1.0 / (1.0 + distance)
                
                knowledge_chunks.append({
                    'chunk_id': metadata['chunk_id'],
                    'doc_id': metadata['doc_id'],
                    'similarity': round(similarity, 3),
                    'content': document,
                    'metadata': metadata
                })
            
            logger.info(f"Found {len(knowledge_chunks)} relevant chunks for query: '{query}'")
            return knowledge_chunks
        
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    def remove_document(self, doc_id: int) -> int:
        """Remove all chunks for a document.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            Number of chunks removed
        """
        try:
            # Query chunks for this document
            results = self.collection.get(where={'doc_id': doc_id})
            chunk_ids = results['ids']
            
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                logger.info(f"Removed {len(chunk_ids)} chunks for document {doc_id}")
                return len(chunk_ids)
            
            return 0
        
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge vector store statistics."""
        return {
            'collection_name': self.collection_name,
            'total_chunks': self.collection.count(),
            'embedding_dimension': self.embedding_manager.get_dimension()
        }


class VectorStoreManager:
    """Main manager for all vector storage operations."""
    
    def __init__(self, data_dir: str = "data/chroma"):
        """Initialize vector store manager.
        
        Args:
            data_dir: ChromaDB data directory
        """
        self.data_dir = data_dir
        self.embedding_manager = EmbeddingManager()
        self.chroma_manager = ChromaDBManager(data_dir)
        self.plant_store = PlantVectorStore(self.chroma_manager, self.embedding_manager)
        self.knowledge_store = KnowledgeVectorStore(self.chroma_manager, self.embedding_manager)
        
        logger.info("Vector store manager initialized")
    
    def sync_plants_from_database(self) -> int:
        """Sync all plants from database to vector store.
        
        Returns:
            Number of plants synced
        """
        try:
            with get_session() as session:
                from .operations import PlantOperations
                
                plant_ops = PlantOperations()
                all_plants = plant_ops.get_all(session)
                
                if all_plants:
                    return self.plant_store.add_plants_batch(all_plants)
                
                return 0
        
        except Exception as e:
            logger.error(f"Failed to sync plants from database: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector stores.
        
        Returns:
            Health status dictionary
        """
        try:
            health = {
                'overall_healthy': True,
                'checks': {}
            }
            
            # Check ChromaDB
            try:
                chroma_stats = self.chroma_manager.get_stats()
                health['checks']['chromadb'] = {
                    'status': 'ok',
                    'healthy': True,
                    'stats': chroma_stats
                }
            except Exception as e:
                health['checks']['chromadb'] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                health['overall_healthy'] = False
            
            # Check embedding model
            try:
                dimension = self.embedding_manager.get_dimension()
                health['checks']['embedding_model'] = {
                    'status': 'ok',
                    'healthy': True,
                    'model': self.embedding_manager.model_name,
                    'dimension': dimension
                }
            except Exception as e:
                health['checks']['embedding_model'] = {
                    'status': 'error',
                    'healthy': False,
                    'error': str(e)
                }
                health['overall_healthy'] = False
            
            # Check collections
            plant_stats = self.plant_store.get_stats()
            knowledge_stats = self.knowledge_store.get_stats()
            
            health['checks']['collections'] = {
                'status': 'ok',
                'healthy': True,
                'plants': plant_stats,
                'knowledge': knowledge_stats
            }
            
            return health
        
        except Exception as e:
            return {
                'overall_healthy': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics."""
        return {
            'embedding_model': self.embedding_manager.model_name,
            'embedding_dimension': self.embedding_manager.get_dimension(),
            'data_directory': self.data_dir,
            'chromadb': self.chroma_manager.get_stats(),
            'plants': self.plant_store.get_stats(),
            'knowledge': self.knowledge_store.get_stats()
        }
    
    def reset_all(self):
        """Reset all vector stores."""
        try:
            self.chroma_manager.reset_database()
            
            # Recreate stores
            self.plant_store = PlantVectorStore(self.chroma_manager, self.embedding_manager)
            self.knowledge_store = KnowledgeVectorStore(self.chroma_manager, self.embedding_manager)
            
            logger.info("All vector stores reset")
        except Exception as e:
            logger.error(f"Failed to reset vector stores: {e}")
            raise


# Global vector store manager instance
vector_store_manager = None


def get_vector_store_manager() -> VectorStoreManager:
    """Get global vector store manager instance."""
    global vector_store_manager
    
    if vector_store_manager is None:
        vector_store_manager = VectorStoreManager()
    
    return vector_store_manager


def initialize_vector_stores() -> bool:
    """Initialize vector stores and perform initial setup.
    
    Returns:
        True if successful
    """
    try:
        manager = get_vector_store_manager()
        
        # Perform health check
        health = manager.health_check()
        
        if not health['overall_healthy']:
            logger.error("Vector store initialization failed health check")
            return False
        
        logger.info("Vector stores initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize vector stores: {e}")
        return False