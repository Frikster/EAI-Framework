from src.agent.llm_agent import LLMAgent, EmotionReflectionLLMAgent
from src.agent.predefined_division_agent import PredefinedRatioDivisionAgent
from src.agent.predefined_table_game import NaiveCooperativeAgent, DeflectingAgent, VindictiveAgent, ImitativeAgent, AlteratingAgent


def init_agent(agent_name, agent_config):
    if agent_name == 'naive_cooperative':
        return NaiveCooperativeAgent(ego_move=agent_config['ego_move'], coop_move=agent_config['coop_move'])
    if agent_name == 'deflecting':
        return DeflectingAgent(ego_move=agent_config['ego_move'], coop_move=agent_config['coop_move'])
    if agent_name == 'alterating':
        return AlteratingAgent(ego_move=agent_config['ego_move'], coop_move=agent_config['coop_move'])
    if agent_name == 'vindictive':
        return VindictiveAgent(ego_move=agent_config['ego_move'], coop_move=agent_config['coop_move'])
    if agent_name == 'imitative':
        return ImitativeAgent(ego_move=agent_config['ego_move'], coop_move=agent_config['coop_move'])
    if agent_name == 'ratio_division':
        return PredefinedRatioDivisionAgent(ratio=agent_config['ratio'], total_sum=agent_config['total_sum'])
    if agent_name == 'llm':
        return LLMAgent(llm_name=agent_config['llm_name'],
                        game_description=agent_config['game_description'],
                        has_emotion=agent_config['has_emotion'],
                        memory_update_format=agent_config['memory_update_format'],
                        round_question_format=agent_config['round_question_format'],
                        do_scratchpad_step=agent_config['do_scratchpad_step'],
                        memory_update_addintional_keys=agent_config['memory_update_addintional_keys']
                        )
    if agent_name == 'emotion_reflection_llm':
        return EmotionReflectionLLMAgent(llm_name=agent_config['llm_name'],
                                         game_description=agent_config['game_description'],
                                         has_emotion=agent_config['has_emotion'],
                                         memory_update_format=agent_config['memory_update_format'],
                                         memory_update_addintional_keys=agent_config['memory_update_addintional_keys'],
                                         round_question_format=agent_config['round_question_format'],
                                         do_scratchpad_step=agent_config['do_scratchpad_step'],
                                         emotion_update_format=agent_config['emotion_update_format'],
                                         emotion_question_format=agent_config['emotion_question_format'],
                                         outer_emotion_update_format=agent_config['outer_emotion_update_format'],
                                         outer_emotions_question_format=agent_config['outer_emotions_question_format'],
                                         outer_opponent_emotion_update_format=agent_config['outer_opponent_emotion_update_format']
        )
    raise NotImplementedError(f'No such agent name {agent_name}')