from typing import List, Dict, Optional

# from langchain.chat_models import ChatOpenAI
# from langchain import LLMChain, PromptTemplate

from src.agent.agent import BasicAgent
from src.utils import get_llm_response
from src.agent.conversation import PreDecisionConversationMixin


class LLMAgent(BasicAgent):
    def __init__(
        self,
        llm_name,
        game_description,
        has_emotion,
        memory_update_format,
        memory_update_addintional_keys,
        round_question_format,
        do_scratchpad_step,
    ):
        super().__init__()
        self.llm_name = llm_name
        self._game_description = game_description
        self._history: List[Dict[str, str]] = []
        self.emotion_memory: List[str] = []
        self.has_emotion = has_emotion
        self._memory_update_format = memory_update_format
        self._round_question_format = round_question_format
        self.memory_update_addintional_keys = memory_update_addintional_keys
        self.do_scratchpad_step = do_scratchpad_step

        self.reflection_request = '''
Explain your decision step by step, be very short and clear: 
'''

    def _add_to_history(self, role, content):
        self._history.append({"role": role, "content": content})

    def init_memory(self):
        self._history.clear()
        self._add_to_history("system", self._game_description)
        return self._game_description

    def update_memory(
        self, my_step, opponent_step, my_reward, opponent_reward, step_num, *args
    ):
        current_update = self._memory_update_format.format(
            round=step_num,
            my_step=my_step,
            opponent_step=opponent_step,
            my_reward=my_reward,
            opponent_reward=opponent_reward,
        )
        self._add_to_history("user", current_update)
        return {"user": current_update}

    def update_emotion_memory(self, emotion):
        self.emotion_memory.append(emotion)

    def _make_step(self, step_num):
        game_question = self._round_question_format.format(round=step_num)
        self._history.append({"role": "user", "content": game_question})
        predicted_step = get_llm_response(self.llm_name, self._history)
        predicted_step = self.postprocess_step(predicted_step)
        self._history.pop()
        return predicted_step

    def make_step(self, step_num):
        scratchpad_step = ''
        if self.do_scratchpad_step:
            self._history.append({"role": "user", "content": self.reflection_request})
            scratchpad_step = get_llm_response(self.llm_name, self._history)
            self._history.append({"role": "assistant", "content": scratchpad_step})
        predicted_step = self._make_step(step_num)
        if self.do_scratchpad_step:
            self._history.pop()
            self._history.pop()
        print(scratchpad_step)
        return predicted_step, scratchpad_step


    # def _make_step_division(self, game_question=None):
    #     if game_question is not None:
    #         self._history.append({"role": "user", "content": game_question})
    #     predicted_step = get_llm_response(self.llm_name, self._history)
    #     predicted_step = self.postprocess_step(predicted_step)
    #     return predicted_step
    #
    # def make_step_division(self, game_question=None):
    #     scratchpad_step = ''
    #     if self.do_scratchpad_step:
    #         self._history.append({"role": "user", "content": self.reflection_request})
    #         scratchpad_step = get_llm_response(self.llm_name, self._history)
    #         self._history.append({"role": "assistant", "content": scratchpad_step})
    #     predicted_step = self._make_step_division(game_question)
    #     # print(scratchpad_step)
    #     return predicted_step, scratchpad_step

    def postprocess_step(self, predicted_step):
        return predicted_step.strip("., \n\t")


class EmotionReflectionLLMAgent(LLMAgent, PreDecisionConversationMixin):
    def __init__(
        self,
        llm_name,
        game_description,
        has_emotion,
        memory_update_format,
        memory_update_addintional_keys,
        round_question_format,
        do_scratchpad_step,
        emotion_update_format=None,
        emotion_question_format=None,
        outer_emotion_update_format=None,
        outer_emotions_question_format=None,
        outer_opponent_emotion_update_format=None,
        # New conversation parameters
        max_conversation_turns: int = 3,
    ):
        LLMAgent.__init__(
            self,
            llm_name=llm_name,
            game_description=game_description,
            has_emotion=has_emotion,
            memory_update_format=memory_update_format,
            memory_update_addintional_keys=memory_update_addintional_keys,
            round_question_format=round_question_format,
            do_scratchpad_step=do_scratchpad_step
        )
        PreDecisionConversationMixin.__init__(
            self,
            max_turns=max_conversation_turns,
        )
        # Existing emotion-related initialization
        self._emotion_update_format = emotion_update_format
        self._inner_emotion_question_format = emotion_question_format
        self._outer_emotion_update_format = outer_emotion_update_format
        self._outer_emotion_question_format = outer_emotions_question_format
        self._outer_opponent_emotion_update_format = outer_opponent_emotion_update_format

    def _get_prompt_from_template(self, template_name: str, **kwargs) -> str:
        """Implementation of the abstract method from PreDecisionConversationMixin"""
        template_path = f"prompts/english/agent/conversation/{template_name}"
        with open(template_path, 'r') as f:
            template = f.read()
        return template.format(**kwargs)

    def _get_llm_response(self, prompt: str) -> str:
        """Implementation of the abstract method from PreDecisionConversationMixin"""
        self._history.append({"role": "user", "content": prompt})
        response = get_llm_response(self.llm_name, self._history)
        self._history.pop()  # Remove the prompt to keep history clean
        return response.strip()

    def _get_emotion_state(self, request_format):
        emoton_question = request_format  # self._emotion_question_format
        self._history.append({"role": "user", "content": emoton_question})
        emotion = get_llm_response(self.llm_name, self._history)
        emotion = self.postprocess_step(emotion)
        self._history.pop()
        return emotion

    def get_inner_emotion(self):
        return self._get_emotion_state(self._inner_emotion_question_format)

    def get_outer_emotion(self):
        return self._get_emotion_state(self._outer_emotion_question_format)

    def update_memory(
        self,
        my_step,
        opponent_step,
        my_reward,
        opponent_reward,
        step_num,
        inner_emotion=None,
        outer_emotion=None,
        outer_opponent_emotion=None,
    ):
        # Get conversation summary if available
        conversation_summary = self.get_current_conversation_summary()
        
        # Format base update
        current_update = self._memory_update_format.format(
            round=step_num,
            my_step=my_step,
            opponent_step=opponent_step,
            my_reward=my_reward,
            opponent_reward=opponent_reward,
            currency=self.memory_update_addintional_keys['currency'],
            conversation_summary=conversation_summary if conversation_summary else "No pre-decision conversation occurred."
        )
        
        # Add emotion updates
        if inner_emotion is not None:
            current_update += self._emotion_update_format.format(emotion=inner_emotion)
        if outer_emotion is not None:
            current_update += self._outer_emotion_update_format.format(
                emotion=outer_emotion
            )
        if outer_opponent_emotion is not None:
            current_update += self._outer_opponent_emotion_update_format.format(
                emotion=outer_opponent_emotion
            )
        
        self._add_to_history("user", current_update)
        return {"user": current_update}
