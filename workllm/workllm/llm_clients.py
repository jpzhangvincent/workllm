from typing import Protocol, Optional
import ollama
from openai import OpenAI
import os

class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients."""
    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = True) -> str: 
        """Generate text from a prompt using the LLM.
        
        Args:
            prompt: The input text prompt for the LLM
            system: Optional system message to guide the LLM's behavior
            stream: Whether to stream the response or return it all at once
            
        Returns:
            The generated text response from the LLM
        """
        ...

class OllamaClient:
    """Client for interacting with Ollama LLM models."""
    
    def __init__(self, model: str = "llama3"):
        """Initialize the Ollama client.
        
        Args:
            model: The name of the Ollama model to use (default: 'llama3')
        """
        self.model = model
        
    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = True) -> str:
        """Generate text using the Ollama model.
        
        Args:
            prompt: The input text prompt for the model
            system: Optional system message to guide the model's behavior
            stream: Whether to stream the response or return it all at once
            
        Returns:
            The generated text response from the model
        """
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            system=system or "",
            stream=stream
        )
        
        if not stream:
            return response["response"]
            
        full_response = ""
        for chunk in response:
            print(chunk["response"], end="", flush=True)
            full_response += chunk["response"]
            
        return full_response

class OpenAIClient:
    """Client for interacting with OpenAI's API."""
    
    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """Initialize the OpenAI client.
        
        Args:
            model: The name of the OpenAI model to use (default: 'gpt-4o-2024-11-20')
            
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = True) -> str:
        """Generate text using the OpenAI model.
        
        Args:
            prompt: The input text prompt for the model
            system: Optional system message to guide the model's behavior
            stream: Whether to stream the response or return it all at once
            
        Returns:
            The generated text response from the model
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt}
            ],
            stream=stream
        )
        
        if not stream:
            return response.choices[0].message.content
            
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
                
        return full_response
