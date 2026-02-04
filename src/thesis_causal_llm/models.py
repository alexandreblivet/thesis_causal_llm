"""
LLM interface for causal reasoning experiments.
Uses HuggingFace Inference API and Anthropic API.
"""

import os

import anthropic
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

ANTHROPIC_MODELS = {
    "claude-haiku",
    "claude-sonnet",
    "claude-opus",
}


class LLMInterface:
    """Interface for querying LLMs via HuggingFace or Anthropic API."""

    def __init__(self, model_name: str):
        """
        Initialize LLM interface.

        Args:
            model_name: Model identifier (HuggingFace or Anthropic)
        """
        self.model_name = model_name
        self.is_anthropic = any(m in model_name for m in ANTHROPIC_MODELS)

        if self.is_anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Add it to your .env file: ANTHROPIC_API_KEY=your_key_here"
                )
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            token = os.getenv("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN not found in environment. "
                    "Add it to your .env file: HF_TOKEN=your_token_here"
                )
            self.client = InferenceClient(token=token)

    def query(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The question/prompt to send
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            The model's text response
        """
        if self.is_anthropic:
            return self._query_anthropic(prompt, temperature)
        else:
            return self._query_huggingface(prompt, temperature)

    def _query_anthropic(self, prompt: str, temperature: float) -> str:
        """Query Anthropic Claude models."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature > 0 else 0.0,
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic query failed: {e}")

    def _query_huggingface(self, prompt: str, temperature: float) -> str:
        """Query HuggingFace models."""
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature if temperature > 0 else None,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"HuggingFace query failed: {e}")


def get_available_models() -> list[str]:
    """Return list of models to test."""
    return [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-7B-Instruct",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
    ]
