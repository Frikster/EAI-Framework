from copy import deepcopy

from tqdm import tqdm

from src.agent.init_agent import init_agent
from src.config_utils.division_utils import prepare_agent_config, prepare_division_game_config
from src.dirs import LOG_PATH
from src.dvision_game import DivisionGame
from src.utils import (
    init_openai_client,
    TwoAgentsLogger,
    save_readable_config,
    print_config, read_text,
)
from dotenv import load_dotenv
import os
import argparse
import pickle

load_dotenv(".env")
assert "OPENAI_API_KEY" in os.environ

init_openai_client(os.environ["OPENAI_API_KEY"])


def read_cached_configs():
    with open('cached/cached_configs_1.pkl', 'rb') as f:
        res1 = pickle.load(f)
    with open('cached/cached_configs_2.pkl', 'rb') as f:
        res2 = pickle.load(f)
    return res1, res2



def generate_emotions(possible_emotions, possible_prompts):
    res = []
    for emotion in possible_emotions:
        for prompt_file in possible_prompts:
            res.append(f'{emotion}/{prompt_file}')
    return res


total_sums = [10**7,  10**3] # [10**2, 10**4, 10**6, 10**8]
coplayers = ["another person", "colleague", "opponent"]
llms = ["gpt-4-0125-preview", "gpt-3.5-turbo-0125"] # ["gpt-3.5-turbo-0125"]  # "gpt-4-0125-preview"
has_emotions = [True, False]


possible_emotions = ["anger", "fear", "disgust", "happiness", "sadness", "surprise"]
possible_prompts = ["simple", "situation_coplayer", "situation_external"]
emotions = generate_emotions(possible_emotions, possible_prompts)
do_scratchpad_steps = [True, False]

predefined_split_ratios = [0.2, 0.3, 0.4] # [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]


# game_basic_config = {
#     "name": "dictator",  # dictator, ultimatum
#     "total_sum": 1000,
# }
#
# naming_config = {
#     "currency": "dollars",  # points, dollars, cents,...
#     "coplayer": "opponent",  # coplayer, opponent, colleague,...
# }

agent_basic_config = {
    "agent_name": "llm",
    "llm_name": "gpt-3.5-turbo",
    "has_emotion": False,
    "emotion": "surprise/situation_coplayer",
    "do_scratchpad_step": False,
    "memory_update_addintional_keys": {},
    "summary_step": "",
    "ratio": 0,
}


def generate_game_configs(game_name):
    return [ {"name": game_name, "total_sum": total_sum, "do_second_step": False} for total_sum in total_sums]


def generate_name_configs():
    return [ {"currency": "dollars", "coplayer": coplayer} for coplayer in coplayers]


def generate_agent_configs():
    cur_agent_basic_config = deepcopy(agent_basic_config)
    agent_configs = []
    for llm in llms:
        cur_agent_basic_config['llm_name'] = llm
        for do_scratchpad_step in do_scratchpad_steps:
            cur_agent_basic_config['do_scratchpad_step'] = do_scratchpad_step
            cur_agent_basic_config['has_emotion'] = True
            for emotion in emotions:
                cur_agent_basic_config['emotion'] = emotion
                agent_configs.append(deepcopy(cur_agent_basic_config))
            cur_agent_basic_config['has_emotion'] = False
            cur_agent_basic_config['emotion'] = ''
            agent_configs.append(deepcopy(cur_agent_basic_config))
    return agent_configs


def generate_predefined_agent_configs():
    res = []
    for ratio in predefined_split_ratios:
        cur_config = deepcopy(agent_basic_config)
        cur_config['agent_name'] = 'ratio_division'
        cur_config['summary_step'] = 'summary_step1'
        cur_config['ratio'] = ratio
        res.append(cur_config)
    return res

def run_game(game_config, naming_config, agent1_config, agent2_config, logger):
    game = DivisionGame(
        name=game_config['name'],
        total_sum=game_config['total_sum'],
        do_second_step=game_config['do_second_step'],
        coplayer_name=naming_config['coplayer']
    )

    agent1 = init_agent(agent1_config["agent_name"], agent1_config)
    agent2 = init_agent(agent2_config["agent_name"], agent2_config)

    # logger.log_json(
    #     {
    #         "agent1_config": agent1_config,
    #         "agent2_config": agent2_config,
    #     }
    # )

    logger.log_json(
        {
            "agent1_config": agent1_config,
            "agent2_config": agent2_config,
            "naming_config": naming_config,
            "game_config": game_config,
            # "agent1_prompt": agent1._history[0],
            # "agent2_prompt": agent2._history[0],
        }
    )

    game.run(agent1, agent2, logger)


def run_pipeline(game_basic_config, naming_config, agent1_basic_config, agent2_basic_config):
    game_config = prepare_division_game_config(game_basic_config)
    agent1_config = prepare_agent_config(
        config=agent1_basic_config,
        game_config=game_basic_config,
        naming_config=naming_config,
        agent_ind=1,
    )
    agent2_config = prepare_agent_config(
        config=agent2_basic_config,
        game_config=game_basic_config,
        naming_config=naming_config,
        agent_ind=2,
    )

    logger = TwoAgentsLogger.construct_from_configs(
        agent1_config, agent2_config, LOG_PATH, game_name=game_basic_config['name']
    )
    if args.verbose:
        print_config(game_config)
        print("==================")
        print_config(agent1_config)
        print("==================")
        print_config(agent2_config)
        raise Exception()

    run_game(game_config, naming_config, agent1_config, agent2_config, logger)

    save_readable_config(game_config, logger.run_name, LOG_PATH)
    save_readable_config(agent1_config, logger.run_name, LOG_PATH)
    save_readable_config(agent2_config, logger.run_name, LOG_PATH)


# Anger, Disgust, Fear, Happiness, Sadness, Surprise

parser = argparse.ArgumentParser(description='Run single experiment')
parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")
args = parser.parse_args()

if __name__ == "__main__":

    # TODO: missing
    # cached_configs_1, cached_configs_2 = read_cached_configs()

    game_dictator_configs = generate_game_configs("dictator")
    game_ultimatum_configs = generate_game_configs("ultimatum")
    name_configs = generate_name_configs()
    agent_configs = generate_agent_configs()
    predefined_agent_configs = generate_predefined_agent_configs()

    print(len(game_dictator_configs), len(game_ultimatum_configs))
    print(len(name_configs))
    print(len(agent_configs), len(predefined_agent_configs))

    # ----dictator & ultimatum for 1st----

    all_configs_1 = []

    for game_basic_config in tqdm(game_dictator_configs + game_ultimatum_configs, desc='Iterating GAME conf'):
        game_basic_config["do_second_step"] = False
        for naming_config in tqdm(name_configs, desc='Iterating NAME conf'):
            for cur_agent_basic_config in tqdm(agent_configs, desc='Iterating AGENT conf'):
                cur_agent_basic_config1 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config2 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config1["summary_step"] = "summary_step1"
                cur_agent_basic_config2["summary_step"] = "summary_step2"
                #     configs.append((row.game_name, row.total_sum, row.coplayer, row.has_emotion, row.llm_name, row.emotion, row.emotion_prompt_file, row.do_scratchpad_step))
                emotion_prompt_file_parts = cur_agent_basic_config1["emotion"].split('/')
                if len(emotion_prompt_file_parts) != 2:
                    if cur_agent_basic_config1["emotion"] != '':
                        raise Exception()
                    emotion_prompt_file_parts = ['no_emotion', '']
                cur_config = (
                    game_basic_config['name'],
                    game_basic_config['total_sum'],
                    naming_config['coplayer'],
                    cur_agent_basic_config1['has_emotion'],
                    cur_agent_basic_config1['llm_name'],
                    emotion_prompt_file_parts[0],
                    emotion_prompt_file_parts[1],
                    cur_agent_basic_config1['do_scratchpad_step']
                )
                # if cur_config not in cached_configs_1:
                #     run_pipeline(game_basic_config, naming_config, cur_agent_basic_config1, cur_agent_basic_config2)
                run_pipeline(game_basic_config, naming_config, cur_agent_basic_config1, cur_agent_basic_config2)

    # ----ultimatum----


    print('ULTIMATUM 2')
    all_configs_2 = []

    # --2--
    for game_basic_config in tqdm(game_ultimatum_configs, desc='Iterating GAME conf'):
        game_basic_config["do_second_step"] = True
        for naming_config in tqdm(name_configs, desc='Iterating NAME conf'):
            for cur_agent_basic_config in tqdm(agent_configs, desc='Iterating AGENT conf'):
                cur_agent_basic_config2 = deepcopy(cur_agent_basic_config)
                cur_agent_basic_config2["summary_step"] = "summary_step2"
                emotion_prompt_file_parts = cur_agent_basic_config2["emotion"].split('/')
                if len(emotion_prompt_file_parts) != 2:
                    if cur_agent_basic_config2["emotion"] != '':
                        raise Exception()
                    emotion_prompt_file_parts = ['no_emotion', '']
                for predefined_agent_config in predefined_agent_configs:
                    cur_config = (
                        game_basic_config['name'],
                        game_basic_config['total_sum'],
                        naming_config['coplayer'],
                        cur_agent_basic_config2['has_emotion'],
                        cur_agent_basic_config2['llm_name'],
                        emotion_prompt_file_parts[0],
                        emotion_prompt_file_parts[1],
                        cur_agent_basic_config2['do_scratchpad_step'],
                        predefined_agent_config['ratio']
                    )
                    all_configs_2.append(cur_config)
                    run_pipeline(game_basic_config, naming_config, predefined_agent_config, cur_agent_basic_config2)
                    # if cur_config not in cached_configs_2:
                    #     run_pipeline(game_basic_config, naming_config, predefined_agent_config, cur_agent_basic_config2)
    #
    # with open('testing/possible_configs_1.pkl', 'wb') as f:
    #     pickle.dump(all_configs_1, f)

    with open('testing/possible_configs_22.pkl', 'wb') as f:
        pickle.dump(all_configs_2, f)