from src.utils import EmotionBuffer, print_emotion_evolution
from tqdm import tqdm

class TableGame:
    def __init__(self, reward_map):
        self.moves_to_rewards = reward_map
    def run(self, agent1, agent2, logger):
        pass


class RepeatedTableGame(TableGame):
    def __init__(
            self,
            reward_map,
            n_steps,
            need_check_emotions,
            need_demonstrate_emotions,
            memorize_demonstrated_emotions,
            memorize_seen_emotions,
    ):
        super().__init__(reward_map)
        self.n_steps = n_steps
        self.need_check_emotions = need_check_emotions
        self.need_demonstrate_emotions = need_demonstrate_emotions
        self.need_check_emotions = need_check_emotions
        self.need_demonstrate_emotions = need_demonstrate_emotions
        self.memorize_demonstrated_emotions = memorize_demonstrated_emotions
        self.memorize_seen_emotions = memorize_seen_emotions

    def run(self, agent1, agent2, logger):
        agent1.init_memory()
        agent2.init_memory()

        for step_num in range(self.n_steps): # tqdm(
            self._run_step(agent1, agent2, step_num, logger)

    def check_for_scatchpad(self, step):
        if isinstance(step, tuple):
            step, scratchpad_step = step
        else:
            step, scratchpad_step = step, None
        return step, scratchpad_step

    def _run_step(self, agent1, agent2, step_num, logger):
        # make current step
        step1, scratchpad_step1 = self.check_for_scatchpad(agent1.make_step(step_num)) 
        step2, scratchpad_step2 = self.check_for_scatchpad(agent2.make_step(step_num))
        try:
            reward1, reward2 = self.moves_to_rewards[step1 + step2]
        except KeyError as err:
            print(err)
            return
        logger.log({"decisions": {"agent1": step1, "agent2": step2}})
        logger.log({"decisions_scratchpad": {"agent1": scratchpad_step1, "agent2": scratchpad_step2}})
        # reflect on self emotions
        additional_args1 = dict.fromkeys(
            ["inner_emotion", "outer_emotion", "opponent_outer_emotion"]
        )
        additional_args2 = dict.fromkeys(
            ["inner_emotion", "outer_emotion", "opponent_outer_emotion"]
        )
        if self.need_check_emotions:
            inner_emotion1 = agent1.get_inner_emotion()
            inner_emotion2 = agent2.get_inner_emotion()
            agent1.update_emotion_memory(inner_emotion1)
            agent2.update_emotion_memory(inner_emotion2)
            additional_args1["inner_emotion"] = inner_emotion1
            additional_args2["inner_emotion"] = inner_emotion2
            logger.log(
                {
                    "inner_emotions": {
                        "agent1": inner_emotion1,
                        "agent2": inner_emotion2,
                    }
                }
            )
        # perceive and demonstrate emotions
        if self.need_demonstrate_emotions:
            outer_emotion1 = agent1.get_outer_emotion()
            outer_emotion2 = agent2.get_outer_emotion()
            # agent1.perceive_opponent_emotion(outer_emotion2)
            # agent2.perceive_opponent_emotion(outer_emotion1)
            logger.log(
                {"outer_emotions": {"agent1": outer_emotion1, "agent2": outer_emotion2}}
            )
            if self.memorize_demonstrated_emotions:
                additional_args1["outer_emotion"] = outer_emotion1
                additional_args2["outer_emotion"] = outer_emotion2
            if self.memorize_seen_emotions:
                additional_args1["opponent_outer_emotion"] = outer_emotion1
                additional_args2["opponent_outer_emotion"] = outer_emotion2
        # Update memory: reformatted according to memory_update.txt answer
        # (optional) + self emotions + seen emotions
        memory_update1 = agent1.update_memory(
            step1,
            step2,
            reward1,
            reward2,
            step_num,
            inner_emotion=additional_args1["inner_emotion"],
            outer_emotion=additional_args1["outer_emotion"],
            outer_opponent_emotion=additional_args2["opponent_outer_emotion"],
        )
        memory_update2 = agent2.update_memory(
            step2,
            step1,
            reward2,
            reward1,
            step_num,
            inner_emotion=additional_args2["inner_emotion"],
            outer_emotion=additional_args2["outer_emotion"],
            outer_opponent_emotion=additional_args1["opponent_outer_emotion"],
        )
        # print(step1, step2)
        # print(f"agent1: {memory_update1}")
        # print(f"agent2: {memory_update2}")
        logger.log({"memory": {"agent1": memory_update1, "agent2": memory_update2}})
