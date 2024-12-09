class BasicAgent:
    def __init__(self):
        pass

    def init_memory(self):
        pass

    def update_memory(self, my_step, opponent_step, my_reward, opponent_reward, step_num, **kwargs):
        pass

    def update_emotion_memory(self, emotion):
        pass

    def make_step(self, step_num):
        pass

    def perceive_opponent_emotion(self, opponent_emotion):
        pass

    def demonstrate_emotion(self):
        pass

    def _get_emotion_state(self, request_format):
        pass
    
    def get_inner_emotion(self):
        pass

    def get_outer_emotion(self):
        pass
