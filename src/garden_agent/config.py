"""Configuration settings for Garden Agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "tinyllama"
    timeout: int = 30
    temperature: float = 0.0
    max_retries: int = 3


@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    path: str = "garden.db"
    echo: bool = False  # SQLAlchemy echo mode


@dataclass
class VectorConfig:
    """Configuration for vector database."""
    persist_directory: str = "data/chromadb"
    collection_name: str = "garden_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig
    database: DatabaseConfig
    vector: VectorConfig
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            llm=LLMConfig(
                model=os.getenv("GARDEN_LLM_MODEL", "tinyllama"),
                timeout=int(os.getenv("GARDEN_LLM_TIMEOUT", "30")),
                temperature=float(os.getenv("GARDEN_LLM_TEMPERATURE", "0.0")),
                max_retries=int(os.getenv("GARDEN_LLM_MAX_RETRIES", "3"))
            ),
            database=DatabaseConfig(
                path=os.getenv("GARDEN_DB_PATH", "garden.db"),
                echo=os.getenv("GARDEN_DB_ECHO", "false").lower() == "true"
            ),
            vector=VectorConfig(
                persist_directory=os.getenv("GARDEN_VECTOR_PATH", "data/chromadb"),
                collection_name=os.getenv("GARDEN_VECTOR_COLLECTION", "garden_knowledge"),
                embedding_model=os.getenv("GARDEN_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
        )


# Default configuration instance
default_config = Config.from_env()