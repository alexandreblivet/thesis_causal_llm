"""
LLM interface for causal reasoning experiments.
Uses Anthropic API exclusively.
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMInterface:
    """Interface for querying LLMs via Anthropic API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Add it to your .env file: ANTHROPIC_API_KEY=your_key_here"
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def query(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The question/prompt to send
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            The model's text response
        """
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


def get_available_models() -> list[str]:
    """Return list of models to test."""
    return [
        "claude-haiku-4-5-20251001",    # Haiku 4.5
        "claude-sonnet-4-5-20250929",   # Sonnet 4.5
        "claude-sonnet-4-6",            # Sonnet 4.6
        "claude-opus-4-6",              # Opus 4.6
    ]
