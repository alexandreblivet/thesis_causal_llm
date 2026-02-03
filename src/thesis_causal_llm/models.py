"""
LLM interface for causal reasoning experiments.
Supports both local models (via Ollama) and API models (OpenAI).
"""

import os
from typing import Literal
from dotenv import load_dotenv
import ollama
from openai import OpenAI

load_dotenv()


class LLMInterface:
    """Unified interface for querying different LLMs."""

    def __init__(self, model_type: Literal["ollama", "openai"], model_name: str):
        """
        Initialize LLM interface.

        Args:
            model_type: Either "ollama" (local) or "openai" (API)
            model_name: Model identifier (e.g., "llama3.1:8b" or "gpt-4o-mini")
        """
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The question/prompt to send

        Returns:
            The model's text response
        """
        if self.model_type == "ollama":
            return self._query_ollama(prompt)
        elif self.model_type == "openai":
            return self._query_openai(prompt)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _query_ollama(self, prompt: str) -> str:
        """Query a local Ollama model."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is the Ollama service running?\n"
                "Start it with: ollama serve"
            )
        except ollama.ResponseError as e:
            if "model" in str(e).lower() and "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{self.model_name}' not found. Pull it with:\n"
                    f"ollama pull {self.model_name}"
                )
            raise RuntimeError(f"Ollama query failed: {e}")
        except Exception as e:
            if "connection refused" in str(e).lower():
                raise RuntimeError(
                    "Cannot connect to Ollama. Is the Ollama service running?\n"
                    "Start it with: ollama serve"
                )
            raise RuntimeError(f"Ollama query failed: {e}")

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic for reproducibility
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI query failed: {e}")


def get_available_models():
    """Return list of available models for the experiment."""
    return [
        ("ollama", "llama3.1:8b"),
        ("openai", "gpt-4o-mini")
    ]