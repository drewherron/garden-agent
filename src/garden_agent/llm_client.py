"""LLM client wrapper for Ollama integration."""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


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


if __name__ == "__main__":
    # Test the connection when run directly
    logging.basicConfig(level=logging.INFO)
    test_ollama_connection()