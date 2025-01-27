import os
from typing import Protocol

import ollama
from botocore.exceptions import NoCredentialsError
from dotenv import find_dotenv, load_dotenv
from langchain_aws import ChatBedrock
from openai import OpenAI

load_dotenv(find_dotenv())

class LLMClient(Protocol):
    """Protocol defining the interface for LLM clients."""
    def generate(self, prompt: str, system: str | None = None, stream: bool = True) -> str:
        """Generate text from a prompt using the LLM.

        Args:
            prompt: The input text prompt for the LLM
            system: Optional system message to guide the LLM's behavior
            stream: Whether to stream the response or return it all at once

        Returns:
            The generated text response from the LLM
        """


class OllamaClient:
    """Client for interacting with Ollama LLM models."""

    def __init__(self, model: str = "llama3.2:3b"):
        """Initialize the Ollama client.
        Args:
            model: The name of the Ollama model to use (default: 'llama3.2:3b')
        """
        self.model = model

    def generate(self, prompt: str, system: str | None = None, stream: bool = True) -> str:
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


class BedrockClient:
    """Client for interacting with AWS Bedrock models."""

    def __init__(self, model: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """Initialize the Bedrock client.
        Args:
            model: The name of the Bedrock model to use (default: 'anthropic.claude-3-sonnet-20240229-v1:0')
        Raises:
            ValueError: If AWS credentials are not properly configured
        """
        try:
            self.client = ChatBedrock(
                model_id=model,
                streaming=True
            )
            self.model = model
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Please configure AWS credentials.") from None

    def generate(self, prompt: str, system: str | None = None, stream: bool = True) -> str:
        """Generate text using the Bedrock model.
        Args:
            prompt: The input text prompt for the model
            system: Optional system message to guide the model's behavior
            stream: Whether to stream the response or return it all at once
        Returns:
            The generated text response from the model
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [HumanMessage(content=prompt)]
        if system:
            messages.insert(0, SystemMessage(content=system))

        if not stream:
            response = self.client.invoke(messages)
            return response.content

        full_response = ""
        for chunk in self.client.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

        return full_response


def get_default_client() -> LLMClient:
    """Get a default LLM client instance.
    Returns:
        An LLMClient instance configured with default settings
    Raises:
        ValueError: If required environment variables are not set for the selected client
    """
    client_type = os.getenv("LLM_CLIENT", "ollama").lower()

    if client_type == "openai":
        return OpenAIClient()
    elif client_type == "deepseek":
        return DeepSeekClient()
    elif client_type == "openrouter":
        return OpenRouterClient()
    elif client_type == "bedrock":
        return BedrockClient()
    return OllamaClient()


class BaseOpenAIClient:
    """Base client for OpenAI-compatible APIs."""

    def __init__(self, model: str, api_key_env: str, base_url: str | None = None):
        """Initialize the base client.
        Args:
            model: The model name to use
            api_key_env: Environment variable name for the API key
            base_url: Optional base URL for the API
        Raises:
            ValueError: If the required API key environment variable is not set
        """
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable is required")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, system: str | None = None, stream: bool = True) -> str:
        """Generate text using the model.
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


class OpenAIClient(BaseOpenAIClient):
    """Client for interacting with OpenAI's API."""

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        """Initialize the OpenAI client.
        Args:
            model: The name of the OpenAI model to use (default: 'gpt-4o-2024-11-20')
        """
        super().__init__(model, "OPENAI_API_KEY")


class DeepSeekClient(BaseOpenAIClient):
    """Client for interacting with DeepSeek's API."""

    def __init__(self, model: str = "deepseek-chat"):
        """Initialize the DeepSeek client.
        Args:
            model: The name of the DeepSeek model to use (default: 'deepseek-chat')
        """
        super().__init__(model, "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1")


class OpenRouterClient(BaseOpenAIClient):
    """Client for interacting with OpenRouter's API."""

    def __init__(self, model: str = "openai/gpt-4"):
        """Initialize the OpenRouter client.
        Args:
            model: The name of the OpenRouter model to use (default: 'openai/gpt-4')
        """
        super().__init__(model, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1")
