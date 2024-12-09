from src.agent.agent import BasicAgent


class PDPredefinedAgent(BasicAgent):
    def __init__(self, ego_move, coop_move):
        super().__init__()
        self.coop_move = coop_move
        self.ego_move = ego_move


class DeflectingAgent(PDPredefinedAgent):
    def make_step(self, step_num):
        return self.ego_move


class NaiveCooperativeAgent(PDPredefinedAgent):
    def make_step(self, step_num):
        return self.coop_move

class AlteratingAgent(PDPredefinedAgent):
    def make_step(self, step_num):
        if step_num % 2 == 0:
            return self.coop_move
        else:
            return self.ego_move
    def get_outer_emotion(self):
        return "angry"

class VindictiveAgent(PDPredefinedAgent):
    """
    Starts with cooperation and cooperates until the opponent deflects. After that it always deflects
    """
    def __init__(self, ego_move, coop_move):
        super().__init__(ego_move, coop_move)
        self.opponent_deflected = False

    def make_step(self, step_num):
        if self.opponent_deflected:
            return self.ego_move
        return self.coop_move

    def update_memory(self, my_step, opponent_step, my_reward, opponent_reward, step_num, **kwargs):
        if opponent_step == self.ego_move:
            self.opponent_deflected = True


class ImitativeAgent(PDPredefinedAgent):
    """
    Imitator
    """
    def __init__(self, ego_move, coop_move):
        super().__init__(ego_move, coop_move)
        self.last_opponent_step = None

    def make_step(self, step_num):
        if self.last_opponent_step is None:
            return self.coop_move
        return self.last_opponent_step

    def update_memory(self, my_step, opponent_step, my_reward, opponent_reward, step_num, **kwargs):
        self.last_opponent_step = opponent_step
