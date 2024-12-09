from src.dirs import PROMPT_PATH, GAME_SET_PATH, INSTRUCT_PATH, AGENT_PATH
from src.utils import read_json, read_text
from copy import deepcopy


def prepare_division_game_config(game_config):
    res = deepcopy(game_config)
    return res

def prepare_agent_config(config, game_config, naming_config, agent_ind: int):
    emotion_prompt = ""
    if config['has_emotion']:
        emotion_prompt = (read_text(PROMPT_PATH / f"emotions/{config['emotion']}.txt").
                          format(coplayer=naming_config['coplayer']))

    rules_file = f"rules{agent_ind}"
    if config['has_emotion']:
        rules_file += f"_emotion"
    rules_file += '.txt'

    rules = read_text(PROMPT_PATH / f"games/{game_config['name']}/{rules_file}").format(
        total_sum=game_config['total_sum'],
        coplayer=naming_config['coplayer'],
        emotion=emotion_prompt
    )

    return {
        "agent_name": config["agent_name"],
        "llm_name": config["llm_name"],
        "has_emotion": config['has_emotion'],
        "game_description": rules,
        "memory_update_addintional_keys": config["memory_update_addintional_keys"],
        "round_question_format": read_text(PROMPT_PATH / f"games/{game_config['name']}/{config['summary_step']}.txt"),
        "do_scratchpad_step": config['do_scratchpad_step'],
        "memory_update_format": "",
        "emotion_update_format": "",
        "emotion_question_format": "",
        "outer_emotion_update_format": "",
        "outer_emotions_question_format": "",
        "outer_opponent_emotion_update_format": "",
        "ratio": config['ratio'],
        "total_sum": game_config['total_sum'],
        "emotion_prompt_file": config['emotion']
    }
