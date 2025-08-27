"""LLM client wrapper for Ollama integration and RAG system."""

import json
import logging
import subprocess
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import os

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.vectorstores import Chroma
    from langchain.embeddings.base import Embeddings
    from langchain.schema.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    success: bool
    error: Optional[str] = None


class OllamaClient:
    """Client for interacting with Ollama local LLM."""
    
    def __init__(self, model: str = "tinyllama", timeout: int = 30):
        """Initialize Ollama client.
        
        Args:
            model: Default model to use for generation
            timeout: Timeout in seconds for LLM calls
        """
        self.model = model
        self.timeout = timeout
        
    def is_available(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                return []
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.0
    ) -> LLMResponse:
        """Generate response from LLM.
        
        Args:
            prompt: User prompt
            model: Model to use (defaults to self.model)
            system: System prompt
            temperature: Sampling temperature
            
        Returns:
            LLM response
        """
        model = model or self.model
        
        # Build the request
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            # Use ollama's chat API via subprocess
            result = subprocess.run(
                ["ollama", "run", model, "--", prompt],
                input=json.dumps(request_data) if system else prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return LLMResponse(
                    content=result.stdout.strip(),
                    model=model,
                    success=True
                )
            else:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Ollama error: {error_msg}")
                return LLMResponse(
                    content="",
                    model=model,
                    success=False,
                    error=error_msg
                )
                
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama request timed out after {self.timeout} seconds")
            return LLMResponse(
                content="",
                model=model,
                success=False,
                error=f"Request timed out after {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return LLMResponse(
                content="",
                model=model,
                success=False,
                error=str(e)
            )
    
    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Generate JSON response from LLM.
        
        Args:
            prompt: User prompt
            model: Model to use
            system: System prompt
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response or error dict
        """
        json_system = (system or "") + "\n\nIMPORTANT: Respond only with valid JSON. No markdown formatting or extra text."
        
        response = self.generate(prompt, model, json_system, temperature)
        
        if not response.success:
            return {"error": response.error}
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Remove common markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response.content}")
            return {
                "error": f"Invalid JSON response: {e}",
                "raw_response": response.content
            }


def get_default_client() -> OllamaClient:
    """Get default Ollama client instance."""
    return OllamaClient()


def test_ollama_connection() -> bool:
    """Test if Ollama is working properly."""
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not available")
        return False
    
    models = client.list_models()
    if not models:
        print("❌ No models found. Please download a model first:")
        print("   ollama pull phi3.5:3.8b")
        return False
    
    print(f"✅ Ollama is available with models: {models}")
    
    # Test a simple generation
    print("Testing simple generation...")
    response = client.generate("Say hello in one word")
    
    if response.success:
        print(f"✅ Generation test successful: {response.content}")
        return True
    else:
        print(f"❌ Generation test failed: {response.error}")
        return False


# RAG System Classes (require LangChain)

class OllamaLangChainLLM(LLM):
    """LangChain wrapper for Ollama models using subprocess."""
    
    model: str = "tinyllama"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: float = 30.0
    ollama_client: Any = None
    
    def __init__(
        self, 
        model: str = "tinyllama",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: float = 30.0,
        **kwargs
    ):
        """Initialize Ollama LangChain LLM.
        
        Args:
            model: Ollama model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(**kwargs)
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain package required. Install with: pip install langchain")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.ollama_client = OllamaClient(model=model, timeout=int(timeout))
        
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Ollama model with prompt."""
        try:
            response = self.ollama_client.generate(
                prompt=prompt,
                temperature=self.temperature
            )
            
            if response.success:
                return response.content
            else:
                return f"Error: {response.error}"
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: Unable to generate response ({str(e)})"


class SimpleEmbeddings(Embeddings):
    """Simple embeddings using sentence-transformers for compatibility."""
    
    def __init__(self):
        """Initialize simple embeddings."""
        try:
            from garden_agent.vector_store import EmbeddingManager
            self.embedding_manager = EmbeddingManager()
        except ImportError:
            # Fallback to random embeddings for testing
            self.embedding_manager = None
            logger.warning("Vector store not available, using random embeddings")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if self.embedding_manager:
            embeddings = self.embedding_manager.encode(texts)
            return embeddings.tolist()
        else:
            # Random embeddings for testing
            import random
            return [[random.random() for _ in range(384)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if self.embedding_manager:
            embedding = self.embedding_manager.encode_single(text)
            return embedding.tolist()
        else:
            # Random embedding for testing
            import random
            return [random.random() for _ in range(384)]


class GardenKnowledgeRAG:
    """RAG system for gardening knowledge using local Ollama models."""
    
    def __init__(
        self,
        llm_model: str = "tinyllama",
        chroma_persist_dir: str = "data/chroma_rag",
        memory_k: int = 5
    ):
        """Initialize RAG system.
        
        Args:
            llm_model: Ollama model for generation
            chroma_persist_dir: ChromaDB persistence directory
            memory_k: Number of previous messages to remember
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain package required for RAG system")
        
        self.llm_model = llm_model
        self.chroma_persist_dir = chroma_persist_dir
        self.memory_k = memory_k
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.qa_chain = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components."""
        try:
            logger.info("Initializing RAG components...")
            
            # Ensure data directory exists
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            
            # Initialize LLM
            self.llm = OllamaLangChainLLM(
                model=self.llm_model,
                temperature=0.3,  # Lower for more factual responses
                max_tokens=500
            )
            
            # Initialize embeddings
            self.embeddings = SimpleEmbeddings()
            
            # Initialize or load vector store
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="gardening_knowledge"
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferWindowMemory(
                k=self.memory_k,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    def add_knowledge(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the knowledge base.
        
        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
        """
        try:
            if not metadatas:
                metadatas = [{"source": "manual", "timestamp": datetime.now().isoformat()} 
                           for _ in documents]
            
            # Add to vector store
            self.vectorstore.add_texts(
                texts=documents,
                metadatas=metadatas
            )
            
            # Persist to disk
            self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: User's gardening question
            
        Returns:
            Dictionary with answer and source information
        """
        try:
            # Use conversational chain
            result = self.qa_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "answer": f"I'm sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_memory_summary(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        if not self.memory or not self.memory.chat_memory:
            return []
        
        messages = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                messages.append({"role": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"role": "ai", "content": message.content})
        
        return messages
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "llm_model": self.llm_model,
                "persist_directory": self.chroma_persist_dir,
                "memory_window": self.memory_k
            }
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        health = {"healthy": True, "checks": {}}
        
        # Check Ollama connection
        try:
            client = OllamaClient()
            if client.is_available():
                health["checks"]["ollama"] = {"status": "ok"}
            else:
                health["checks"]["ollama"] = {"status": "error", "error": "Ollama not available"}
                health["healthy"] = False
        except Exception as e:
            health["checks"]["ollama"] = {"status": "error", "error": str(e)}
            health["healthy"] = False
        
        # Check models
        try:
            client = OllamaClient()
            models = client.list_models()
            
            llm_available = any(self.llm_model in name for name in models)
            
            health["checks"]["models"] = {
                "llm_model": self.llm_model,
                "llm_available": llm_available,
                "all_models": models
            }
            
            if not llm_available:
                health["healthy"] = False
                
        except Exception as e:
            health["checks"]["models"] = {"status": "error", "error": str(e)}
            health["healthy"] = False
        
        # Check vector store
        try:
            stats = self.get_knowledge_stats()
            health["checks"]["knowledge_base"] = {
                "status": "ok",
                "document_count": stats.get("total_documents", 0)
            }
        except Exception as e:
            health["checks"]["knowledge_base"] = {"status": "error", "error": str(e)}
            health["healthy"] = False
        
        return health


# Global RAG instance
_rag_system = None


def get_rag_system(llm_model: str = "tinyllama") -> GardenKnowledgeRAG:
    """Get or create global RAG system instance."""
    global _rag_system
    
    if _rag_system is None:
        _rag_system = GardenKnowledgeRAG(llm_model=llm_model)
    
    return _rag_system


def initialize_rag_system() -> bool:
    """Initialize RAG system and check health."""
    try:
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available - install with: pip install langchain")
            return False
            
        rag = get_rag_system()
        health = rag.health_check()
        
        if health["healthy"]:
            logger.info("RAG system initialized successfully")
            return True
        else:
            logger.error(f"RAG system health check failed: {health}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False


# Convenience functions for common operations
def ask_gardening_question(question: str) -> str:
    """Simple interface to ask a gardening question."""
    try:
        if not LANGCHAIN_AVAILABLE:
            return "Sorry, RAG system requires langchain package. Install with: pip install langchain"
            
        rag = get_rag_system()
        result = rag.query(question)
        return result["answer"]
    except Exception as e:
        return f"Sorry, I couldn't process your question: {str(e)}"


def add_gardening_knowledge(content: str, source: str = "manual") -> bool:
    """Add knowledge to the RAG system."""
    try:
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available for RAG system")
            return False
            
        rag = get_rag_system()
        rag.add_knowledge(
            documents=[content],
            metadatas=[{"source": source, "timestamp": datetime.now().isoformat()}]
        )
        return True
    except Exception as e:
        logger.error(f"Failed to add knowledge: {e}")
        return False


if __name__ == "__main__":
    # Test the connection when run directly
    logging.basicConfig(level=logging.INFO)
    test_ollama_connection()