import pytest
from time import time, sleep
from src.agent.conversation import PreDecisionConversationMixin, ConversationTurn

class TestAgent(PreDecisionConversationMixin):
    """Test implementation of an agent using the mixin"""
    def __init__(self, max_turns=3):
        super().__init__(max_turns=max_turns)
        self.prompt_responses = {}  # For mocking LLM responses
    
    def _get_prompt_from_template(self, template_name: str, **kwargs) -> str:
        return f"Mock prompt for {template_name}"
    
    def _get_llm_response(self, prompt: str) -> str:
        return self.prompt_responses.get(prompt, "Default response")

def test_conversation_basic_flow():
    agent = TestAgent()
    agent.prompt_responses = {
        "Mock prompt for init_conversation.txt": "Hello, let's cooperate",
        "Mock prompt for response.txt": "I agree to cooperate"
    }
    
    # Test conversation start
    response = agent.start_conversation(round_num=1)
    assert response == "Hello, let's cooperate"
    assert len(agent._current_conversation) == 1
    assert agent._current_conversation[0].speaker == "initiator"
    
    # Test response
    response = agent.respond_to_message("I also want to cooperate", round_num=1)
    assert response == "I agree to cooperate"
    assert len(agent._current_conversation) == 2
    assert agent._current_conversation[1].speaker == "responder"

def test_conversation_turn_limit():
    agent = TestAgent(max_turns=2)
    
    # Fill up turns
    agent.start_conversation(round_num=1)
    agent.respond_to_message("Message 1", round_num=1)
    agent.respond_to_message("Message 2", round_num=1)
    agent.respond_to_message("Message 3", round_num=1)
    
    # Try to exceed limit
    response = agent.respond_to_message("Message 4", round_num=1)
    assert response is None
    assert len(agent._current_conversation) == 4  # 2 turns * 2 messages per turn

def test_conversation_storage():
    agent = TestAgent()
    
    # First conversation
    agent.start_conversation(round_num=1)
    agent.respond_to_message("Test message", round_num=1)
    agent.end_conversation()
    
    # Second conversation
    agent.start_conversation(round_num=2)
    agent.respond_to_message("Another message", round_num=2)
    agent.end_conversation()
    
    assert len(agent._conversation_history) == 2
    assert len(agent._current_conversation) == 0

def test_conversation_summary():
    agent = TestAgent()
    # Add mock responses
    agent.prompt_responses = {
        "Mock prompt for init_conversation.txt": "Let's cooperate",
        "Mock prompt for response.txt": "I agree to cooperate"
    }
    
    agent.start_conversation(round_num=1)
    agent.respond_to_message("Sure!", round_num=1)
    
    summary = agent.get_current_conversation_summary()
    assert "initiator: Let's cooperate" in summary
    assert "responder: I agree to cooperate" in summary

def test_conversation_history_format():
    agent = TestAgent()
    
    # Have a conversation
    agent.start_conversation(round_num=1)
    agent.respond_to_message("Test message", round_num=1)
    agent.end_conversation()
    
    # Check history structure
    assert len(agent._conversation_history) == 1
    conversation = agent._conversation_history[0]
    assert len(conversation) == 2
    
    # Check turn format
    turn = conversation[0]
    assert isinstance(turn, ConversationTurn)
    assert hasattr(turn, 'speaker')
    assert hasattr(turn, 'message')
    assert hasattr(turn, 'timestamp')

def test_timestamp_tracking():
    agent = TestAgent()
    
    start_time = time()
    agent.start_conversation(round_num=1)
    
    turn = agent._current_conversation[0]
    assert turn.timestamp >= start_time
    assert turn.timestamp <= time()