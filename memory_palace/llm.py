"""
LLM generation operations for Claude Memory Palace.

Provides functions for generating text using Ollama LLM models.
Uses aggressive VRAM management (keep_alive: 0) to allow model swapping.
"""

import requests
from typing import Optional

from .config import (
    get_ollama_url,
    get_llm_model,
    get_auto_link_config,
    PREFERRED_LLM_MODELS,
    PREFERRED_CLASSIFICATION_MODELS,
)


# Module-level cache for detected LLM model
_detected_llm_model: Optional[str] = None


def _detect_llm_model() -> Optional[str]:
    """
    Auto-detect an available LLM model from Ollama.

    Queries Ollama for available models and returns the first one
    from our preferred list that's available.

    Returns:
        Model name if found, None if Ollama unavailable or no suitable model
    """
    global _detected_llm_model

    if _detected_llm_model is not None:
        return _detected_llm_model

    ollama_url = get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        available_models = {m.get("name", "") for m in data.get("models", [])}

        # Find first preferred model that's available
        for preferred in PREFERRED_LLM_MODELS:
            if preferred in available_models:
                _detected_llm_model = preferred
                return preferred

            # Also check without tag suffix
            base_name = preferred.split(":")[0]
            for available in available_models:
                if available.startswith(base_name):
                    _detected_llm_model = available
                    return available

        # No preferred model found, skip embedding models and return first available
        for model in available_models:
            # Skip embedding-specific models
            if "embed" in model.lower():
                continue
            _detected_llm_model = model
            return model

        return None

    except requests.exceptions.RequestException:
        return None


def get_active_llm_model() -> Optional[str]:
    """
    Get the LLM model to use, either configured or auto-detected.

    Returns:
        Model name to use, or None if none available
    """
    # Check if explicitly configured
    configured = get_llm_model()
    if configured:
        return configured

    # Otherwise auto-detect
    return _detect_llm_model()


def generate_with_llm(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None
) -> Optional[str]:
    """
    Generate text using Ollama LLM.

    Args:
        prompt: The prompt to send to the LLM
        system: Optional system message to set model behavior
        model: Model to use (uses config/auto-detected if not specified)

    Returns:
        Generated text response, or None if Ollama unavailable or error
    """
    # Determine which model to use
    if model is None:
        model = get_active_llm_model()

    if model is None:
        # No model available
        return None

    ollama_url = get_ollama_url()

    try:
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": True,  # Enable Qwen3 thinking/reasoning mode
            "keep_alive": "0",  # Unload model immediately - aggressive VRAM strategy
            "options": {
                "num_ctx": 65536,   # 64K context window - full Qwen capacity
                "flash_attn": True  # Flash attention - ~2x KV cache efficiency
            }
        }
        if system:
            request_body["system"] = system

        response = requests.post(
            f"{ollama_url}/api/generate",
            json=request_body,
            timeout=180  # Transcripts can be long, thinking takes time
        )
        response.raise_for_status()
        data = response.json()
        # With think:true, response has "thinking" (reasoning) and "response" (answer)
        # We only return the final answer, but thinking trace is in data["thinking"]
        return data.get("response")
    except requests.exceptions.RequestException as e:
        # Ollama unavailable or error - fail gracefully
        print(f"LLM generation failed: {e}")
        return None
    except (KeyError, ValueError) as e:
        # Malformed response
        print(f"LLM response parsing failed: {e}")
        return None


def is_llm_available() -> bool:
    """
    Check if an LLM model is available for generation.

    Returns:
        True if a model is available, False otherwise
    """
    return get_active_llm_model() is not None


def clear_model_cache() -> None:
    """Clear the detected model cache, forcing re-detection on next call."""
    global _detected_llm_model, _detected_classification_model
    _detected_llm_model = None
    _detected_classification_model = None


# --- Edge Type Classification ---

# Valid edge types for normalization
VALID_EDGE_TYPES = {
    "relates_to", "supersedes", "derived_from",
    "contradicts", "exemplifies", "refines",
}

# Common LLM output variations → canonical edge type
_EDGE_TYPE_ALIASES = {
    "relates": "relates_to",
    "relates_to": "relates_to",
    "supersedes": "supersedes",
    "supersede": "supersedes",
    "derived_from": "derived_from",
    "derives_from": "derived_from",
    "derives": "derived_from",
    "derived": "derived_from",
    "contradicts": "contradicts",
    "contradict": "contradicts",
    "contradiction": "contradicts",
    "exemplifies": "exemplifies",
    "exemplify": "exemplifies",
    "example": "exemplifies",
    "refines": "refines",
    "refine": "refines",
    "refined": "refines",
}

CLASSIFICATION_PROMPT = """You are classifying the relationship between two memories in a knowledge graph. Return ONLY one word from the list below.

IMPORTANT: You must NEVER return "supersedes". Only a human can decide that one memory supersedes another. If two memories conflict, return "contradicts" — the user will decide how to resolve it.

Relationship types:
- relates_to: General topical similarity, no direct logical dependency between the two
- derived_from: Memory B was built from, implements, or extends Memory A
- contradicts: Memory A and Memory B make conflicting or incompatible claims about the same thing. This includes cases where Memory B appears to update, replace, or override Memory A — always use contradicts, never supersedes.
- exemplifies: Memory B describes a specific real-world event or instance that illustrates the abstract concept in Memory A. Memory A is a rule; Memory B is a case where the rule applied.
- refines: Memory B is an updated, more precise version of the SAME statement in Memory A. Both say the same thing, but B adds exact numbers, names, or details that A left vague.

Memory A: "{subject_a}"

Memory B: "{subject_b}"

Relationship type:"""


# Module-level cache for detected classification model
_detected_classification_model: Optional[str] = None


def _detect_classification_model() -> Optional[str]:
    """
    Auto-detect a small model suitable for edge classification.

    Prefers small, CPU-friendly models from PREFERRED_CLASSIFICATION_MODELS.
    Falls back to the main LLM model if no small model is available.
    """
    global _detected_classification_model

    if _detected_classification_model is not None:
        return _detected_classification_model

    # Check config override first
    auto_link_config = get_auto_link_config()
    configured = auto_link_config.get("classification_model")
    if configured:
        _detected_classification_model = configured
        return configured

    ollama_url = get_ollama_url()

    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        available_models = {m.get("name", "") for m in data.get("models", [])}

        # Find first preferred classification model that's available
        for preferred in PREFERRED_CLASSIFICATION_MODELS:
            if preferred in available_models:
                _detected_classification_model = preferred
                return preferred

            # Also check without tag suffix
            base_name = preferred.split(":")[0]
            for available in available_models:
                if available.startswith(base_name):
                    _detected_classification_model = available
                    return available

        # Fall back to the main LLM model
        fallback = get_active_llm_model()
        if fallback:
            _detected_classification_model = fallback
            return fallback

        return None

    except requests.exceptions.RequestException:
        return None


def _normalize_edge_type(raw: str) -> str:
    """
    Normalize LLM output to a valid edge type.

    Handles variations like 'derives' → 'derived_from', strips whitespace,
    lowercases, etc. Returns 'relates_to' if unrecognizable.
    """
    cleaned = raw.strip().lower().rstrip(".,;:!?")
    # Take first word only (model might be chatty)
    first_word = cleaned.split()[0] if cleaned.split() else ""

    # Check alias map
    if first_word in _EDGE_TYPE_ALIASES:
        resolved = _EDGE_TYPE_ALIASES[first_word]
    elif first_word in VALID_EDGE_TYPES:
        resolved = first_word
    else:
        # Fuzzy: check if any valid type starts with what we got
        resolved = "relates_to"  # default fallback
        for valid in VALID_EDGE_TYPES:
            if valid.startswith(first_word) and len(first_word) >= 4:
                resolved = valid
                break

    # supersedes is NEVER set by the classifier — only by human action.
    # If the model outputs it despite instructions, redirect to contradicts.
    if resolved == "supersedes":
        return "contradicts"

    return resolved


def classify_edge_type(
    subject_a: str,
    subject_b: str,
    model: Optional[str] = None
) -> str:
    """
    Classify the relationship between two memories using a small LLM.

    Args:
        subject_a: Subject/summary of the source memory
        subject_b: Subject/summary of the target memory
        model: Model override (uses auto-detected classification model if None)

    Returns:
        One of: relates_to, supersedes, derived_from, contradicts, exemplifies, refines
    """
    if model is None:
        model = _detect_classification_model()

    if model is None:
        return "relates_to"  # No model available, safe fallback

    ollama_url = get_ollama_url()
    prompt = CLASSIFICATION_PROMPT.format(
        subject_a=subject_a,
        subject_b=subject_b,
    )

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,   # Low but not zero; reasoning models need sampling
                    "num_predict": 2000,  # Room for reasoning + output
                    "keep_alive": "0",    # Aggressive VRAM management
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "")
        return _normalize_edge_type(raw)

    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"Edge classification failed: {e}")
        return "relates_to"  # Graceful fallback
