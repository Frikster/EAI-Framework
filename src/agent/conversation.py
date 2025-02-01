from typing import List, Dict, Optional
from dataclasses import dataclass
from time import time

@dataclass
class ConversationTurn:
    speaker: str
    message: str
    timestamp: float

class PreDecisionConversationMixin:
    def __init__(self, max_turns: int = 3):
        self.max_conversation_turns = max_turns
        self._conversation_history: List[List[ConversationTurn]] = []
        self._current_conversation: List[ConversationTurn] = []
    
    def start_conversation(self, round_num: int) -> str:
        """Initiates a conversation for the current round."""
        self._current_conversation = []
        prompt = self._get_prompt_from_template("init_conversation.txt", round=round_num)
        response = self._get_llm_response(prompt)
        self._add_to_conversation("initiator", response)
        return response

    def respond_to_message(self, other_message: str, round_num: int) -> Optional[str]:
        """Generates a response to the other agent's message."""
        if len(self._current_conversation) >= self.max_conversation_turns * 2:
            return None
            
        prompt = self._get_prompt_from_template(
            "response.txt",
            last_message=other_message,
            round=round_num
        )
        response = self._get_llm_response(prompt)
        self._add_to_conversation("responder", response)
        return response

    def _add_to_conversation(self, role: str, message: str) -> None:
        """Adds a message to the current conversation."""
        turn = ConversationTurn(
            speaker=role,
            message=message,
            timestamp=time()
        )
        self._current_conversation.append(turn)

    def end_conversation(self) -> None:
        """Stores the current conversation in history."""
        if self._current_conversation:
            self._conversation_history.append(self._current_conversation)
            self._current_conversation = []

    def get_current_conversation_summary(self) -> str:
        """Returns a formatted summary of the current conversation."""
        if not self._current_conversation:
            return ""
        
        summary = []
        for turn in self._current_conversation:
            summary.append(f"{turn.speaker}: {turn.message}")
        return "\n".join(summary)

    def _get_prompt_from_template(self, template_name: str, **kwargs) -> str:
        """Helper method to load and format prompt templates."""
        # This will be implemented when we add the prompt templates
        raise NotImplementedError()

    def _get_llm_response(self, prompt: str) -> str:
        """Helper method to get LLM response."""
        # This will use the existing LLM infrastructure from the agent
        raise NotImplementedError() 