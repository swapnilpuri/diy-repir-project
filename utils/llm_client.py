"""
Provider-independent Instructor client.

Supports:
    ollama/qwen3.5:9b
    ollama/qwen3:14b
    openai/<model>
    anthropic/<model>

The same client factory is used by:
    Step 1 - Generator
    Step 4 - Judge
"""
import os
import instructor

from config.settings import (
    OLLAMA_HOST,
)

def create_instructor_client(
    provider_model: str,
):
    """
    Create an Instructor client using provider/model.

    Ollama uses the locally configured OLLAMA_HOST.
    Other providers use Instructor's native integrations.
    """

    if "/" not in provider_model:
        raise ValueError(
            "Model must use provider/model format. "
            f"Received: {provider_model}"
        )

    # Split only once so any further slashes remain part of the model identifier.
    provider, model = provider_model.split("/", 1)

    if not model.strip():
        raise ValueError(
            "Model name cannot be empty."
        )

    if provider.lower() == "ollama":

        # Ollama exposes an OpenAI-compatible LLM endpoint. JSON mode requests
        # structured output that Instructor parses against the response schema.
        return instructor.from_provider(
            provider_model,
            base_url=f"{OLLAMA_HOST}/v1",
            mode=instructor.Mode.JSON,
        )

    # Hosted providers use Instructor's native integrations and credential setup.
    return instructor.from_provider(
        provider_model
    )

def get_llm_client(model_name: str):
    """
    Legacy convenience wrapper around Instructor's provider factory.

    Unlike create_instructor_client(), this does not explicitly configure the
    project's Ollama host or JSON mode. Steps 1 and 4 use the newer factory.
    """
    client = instructor.from_provider(model_name)
    return client

