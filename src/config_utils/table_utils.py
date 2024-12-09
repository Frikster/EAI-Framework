from src.dirs import PROMPT_PATH, GAME_SET_PATH, INSTRUCT_PATH, AGENT_PATH
from src.utils import read_json, read_text
from copy import deepcopy


def prepare_game_description(config, naming_config):

    reward_format_map = read_json(PROMPT_PATH / f"games/{config['name']}/rewards.json")
    reward_map = {
        k.format(move1=naming_config['move1'], move2=naming_config['move2']): v
        for k, v in reward_format_map.items()
    }

    res = deepcopy(config)
    res['reward_map'] = reward_map

    return res


def prepare_agent_config(config, game_name, naming_config, agent_ind: int):
    if "llm" not in config["agent_name"]:
        return {
            "agent_name": config["agent_name"],
            "ego_move": naming_config[f"move{agent_ind}"],
            "coop_move": naming_config[f"move{1 + agent_ind % 2}"],
        }

    memory_update_format = read_text(AGENT_PATH / "memory_update.txt")
    emotion_update_format = read_text(AGENT_PATH / "emotions/emotion_update_format.txt")
    emotion_question_format = read_text(AGENT_PATH / "emotions/emotions_question.txt")

    outer_emotion_update_format = read_text(AGENT_PATH / "outer_emotions/outer_emotion_update_format.txt")
    outer_emotions_question_format = read_text(AGENT_PATH / "outer_emotions/outer_emotions_question.txt")
    outer_opponent_emotion_update_format = read_text(AGENT_PATH / "outer_emotions/outer_opponent_emotion_update_format.txt")

    emotion_prompt = ""
    if config['has_emotion']:
        emotion_prompt = (read_text(PROMPT_PATH / f"emotions/{config['emotion']}.txt").
                          format(coplayer=naming_config['coplayer']))

    game_desc_template = prepare_game_desc_template(config['game_setting'], game_name, emotion_prompt, agent_ind)
    game_desc = game_desc_template.format(currency=naming_config['currency'],
                                          coplayer=naming_config['coplayer'],
                                          move1=naming_config['move1'],
                                          move2=naming_config['move2'],
                                          emotion=emotion_prompt)

    round_question_format = read_text(GAME_SET_PATH / f"round/{config['game_setting']['round_question']}{agent_ind}.txt")
    round_question_format = round_question_format.format(round="{round}",
                                                         move1=naming_config["move1"],
                                                         move2=naming_config["move2"])

    return {
        "agent_name": config["agent_name"],
        "llm_name": config["llm_name"],
        "has_emotion": config['has_emotion'],
        "game_description": game_desc,
        "memory_update_addintional_keys": config["memory_update_addintional_keys"],
        "round_question_format": round_question_format,
        "do_scratchpad_step": config['do_scratchpad_step'],
        "memory_update_format": memory_update_format,
        "emotion_update_format": emotion_update_format,
        "emotion_question_format": emotion_question_format,
        "outer_emotion_update_format": outer_emotion_update_format,
        "outer_emotions_question_format": outer_emotions_question_format,
        "outer_opponent_emotion_update_format": outer_opponent_emotion_update_format,
    }


def prepare_game_desc_template(config, game_name, emotion_prompt, agent_ind):
    general_template = read_text(GAME_SET_PATH / 'general_template' / f"{config['general_template']}.txt")
    environment_file = f"{config['environment']}"
    if config['emotions_info'] != '':
        environment_file += f"_{config['emotions_info']}"
    environment_desc_format = read_text(GAME_SET_PATH / 'environment' / f"{environment_file}.txt")
    final_instructions = read_text(INSTRUCT_PATH / f"{config['final_instruction']}.txt")
    game_rules = read_text(PROMPT_PATH / f"games/{game_name}/rules{agent_ind}.txt")
    game_desc_template = general_template.format(enviroment=environment_desc_format,
                                                 game_rules=game_rules,
                                                 emotion=emotion_prompt,
                                                 final_instructions=final_instructions)
    return game_desc_template
