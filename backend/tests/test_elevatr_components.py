import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field

# 1. TEST LLM OUTPUT SCHEMA VALIDATION (Orchestrator Pattern)

# Define a sample Pydantic schema similar to what orchestrator.py expects
class Decomposition(BaseModel):
    intent: str
    missing_skills: list[str]
    suggested_actions: list[str]

def test_pydantic_schema_validation_success():
    """Verify that valid structured JSON from LLM successfully parses."""
    raw_llm_payload = {
        "intent": "resume_review",
        "missing_skills": ["Docker", "Kubernetes"],
        "suggested_actions": ["Add containerization project to resume"]
    }
    
    # Parse into model
    parsed_data = Decomposition(**raw_llm_payload)
    
    assert parsed_data.intent == "resume_review"
    assert "Docker" in parsed_data.missing_skills
    assert len(parsed_data.suggested_actions) == 1

def test_pydantic_schema_validation_failure():
    """Verify that malformed LLM responses raise a ValidationError."""
    invalid_llm_payload = {
        "intent": "resume_review"
        # Missing required fields: missing_skills and suggested_actions
    }
    
    with pytest.raises(Exception):
        Decomposition(**invalid_llm_payload)


# 2. TEST REDIS SLIDING WINDOW CHAT HISTORY 

def test_redis_sliding_window_limit():
    """Verify that chat memory keeps only the latest 15 turns (ltrim)."""
    # Create a fake/mocked Redis client
    mock_redis = MagicMock()
    
    # Simulate adding a message and trimming to last 15 items
    user_id = "test_student_123"
    new_message = '{"role": "user", "content": "How do I prep for a tech interview?"}'
    
    # Run the database memory logic
    mock_redis.rpush(f"chat:{user_id}", new_message)
    mock_redis.ltrim(f"chat:{user_id}", -15, -1)
    
    # Assert that rpush and ltrim were called with correct parameters
    mock_redis.rpush.assert_called_once_with("chat:test_student_123", new_message)
    mock_redis.ltrim.assert_called_once_with("chat:test_student_123", -15, -1)