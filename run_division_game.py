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

load_dotenv(".env")
assert "OPENAI_API_KEY" in os.environ

init_openai_client(os.environ["OPENAI_API_KEY"])

game_basic_config = {
    "name": "dictator",  # dictator, ultimatum
    "total_sum": 1000,
}

naming_config = {
    "currency": "dollars",  # points, dollars, cents,...
    "coplayer": "another person",  # coplayer, opponent, colleague,...
}

agent1_basic_config = {
    "agent_name": "llm",
    "llm_name": "gpt-3.5-turbo-0125",
    "has_emotion": False,
    "emotion": "surprise/situation_coplayer",
    "do_scratchpad_step": False,
    "memory_update_addintional_keys": {},
    "summary_step": "summary_step1",
    "ratio": 0,
}

agent2_basic_config = {
    "agent_name": "llm",
    "llm_name": "gpt-3.5-turbo-0125",
    "has_emotion": False,
    "emotion": "surprise/situation_coplayer",
    "do_scratchpad_step": True,
    "memory_update_addintional_keys": {},
    "summary_step": "summary_step2",
    "ratio": 0,
}


def run_game(game_config, naming_config, agent1_config, agent2_config, logger):
    game = DivisionGame(
        name=game_config['name'],
        total_sum=game_config['total_sum'],
        coplayer_name=naming_config['coplayer'],
        do_second_step=True
    )

    agent1 = init_agent(agent1_config["agent_name"], agent1_config)
    agent2 = init_agent(agent2_config["agent_name"], agent2_config)

    logger.log_json(
        {
            "agent1_config": agent1_config,
            "agent2_config": agent2_config,
        }
    )

    game.run(agent1, agent2, logger)


# Anger, Disgust, Fear, Happiness, Sadness, Surprise

parser = argparse.ArgumentParser(description='Run single experiment')
parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose mode")
args = parser.parse_args()

if __name__ == "__main__":
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
