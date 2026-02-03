"""
LLM interface for causal reasoning experiments.
Supports local models via Ollama.
"""

import ollama


class LLMInterface:
    """Interface for querying Ollama models."""

    def __init__(self, model_name: str):
        """
        Initialize LLM interface.

        Args:
            model_name: Ollama model identifier (e.g., "llama3.1:8b")
        """
        self.model_name = model_name

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
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
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


# Models to evaluate in the experiment
MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
]


def get_available_models() -> list[str]:
    """Return list of Ollama models to test."""
    return MODELS.copy()
