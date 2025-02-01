import json
import os
from collections import defaultdict
import os
import csv
from datetime import datetime
from typing import Dict, List

from openai import OpenAI
import os

openai_client = None


def init_openai_client(OPENAI_API_KEY):
    global openai_client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

def print_config(config):
    for k in config:
        print(k)
        print(config[k])
        print('------------')

def save_readable_config(config, run_name, log_path):
    with open(os.path.join(log_path, run_name, 'readable_summary.txt'), 'a') as f:
        for k in config:
            f.write(f'{k}\n')
            f.writelines(f'{config[k]}\n')
            f.write('------------\n')
        f.write('============\n')

class EmotionBuffer:
    def __init__(self):
        self.agent_id2emotions = defaultdict(list)

    def add_emotion(self, agent_id, emotion):
        self.agent_id2emotions[agent_id].append(emotion)


class BasicLogger:
    """
    A Logger class for writing logs_exps to CSV files.

    Attributes:
        keys (List[str]): A list of keys for which logs_exps need to be created.
        logs_path (str): The base path where log directories will be created.
        now (str): The current timestamp used for creating unique directories.

    Methods:
        log_to_csv(values_dict: Dict[str, str]): Logs the provided values to their respective CSV files.
    """

    def __init__(
        self, keys: List[str], logs_path: str = "", run_name: str = None, game_name: str = ''
    ) -> None:
        """
        Initializes the Logger instance.

        Args:
            keys (List[str]): A list of keys (categories) for logging.
            logs_path (str): The base directory path where logs_exps will be stored.

        Creates a new directory for each key under the 'logs_path/now' directory,
        where 'now' is the current timestamp.
        """
        self.keys: List[str] = keys
        self.logs_path: str = logs_path

        if run_name is None:
            self.run_name = datetime.now().strftime("%d_%m_%H%M%S")
            if game_name != '':
                self.run_name = game_name + '_' + self.run_name
        else:
            self.run_name = run_name

        os.makedirs(os.path.join(logs_path, self.run_name), exist_ok=True)

    def log_json(self, configs: Dict[str, Dict[str, any]]) -> None:
        """
        Logs each nested dictionary in the given dictionary of dictionaries to separate JSON files.
        The filenames are derived from the keys of the outer dictionary.

        Args:
            configs (Dict[str, Dict[str, any]]): A dictionary of configuration dictionaries.
        """
        for config_name, config in configs.items():
            self._write_config_to_file(config, config_name)

    def _write_config_to_file(self, config: Dict[str, any], filename: str) -> None:
        """
        Writes the given configuration dictionary to a JSON file.

        Args:
            config (Dict[str, any]): The configuration dictionary to log.
            filename (str): The filename (without extension) to save the configuration.
        """
        file_path = os.path.join(self.logs_path, self.run_name, f"{filename}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)

    @classmethod
    def construct_from_configs(
        cls,
        agent1_config: dict,
        agent2_config: dict,
        logs_path: str = "",
        run_name: str = None,
        game_name: str = None,
    ):
        keys = ["decisions", "memory", "decisions_scratchpad", "inner_emotions", "outer_emotions", "div_decisions",
                "div_decisions_scratchpad", "conversation"]
        # if (
        #     "inner_emotions" in agent1_config.keys()
        #     or "inner_emotions" in agent2_config.keys()
        # ):
        #     keys.append("inner_emotions")
        # if (
        #     "outer_emotions" in agent1_config.keys()
        #     or "outer_emotions" in agent2_config.keys()
        # ):
        #     keys.append("outer_emotions")

        # Create an instance of cls (BasicLogger in this case) with the computed keys
        return cls(keys=keys, logs_path=logs_path, run_name=run_name, game_name=game_name)

    def log(self, values_dict: Dict[str, str], format='csv') -> None:
        """
        Logs the given values to files, one file per key.

        Args:
            values_dict (Dict[str, str])
        """
        pass


class TwoAgentsLogger(BasicLogger):
    """
    A Logger class for writing logs_exps to CSV files.

    Attributes:
        keys (List[str]): A list of keys for which logs_exps need to be created.
        logs_path (str): The base path where log directories will be created.
        run_name (str): Specific name of the experiment.
        game_name (str): Name of the game
        The current timestamp is used if no name is provided.

    Methods:
        log_to_csv(values_dict: Dict[str, str]): Logs the provided values to their respective CSV files.
    """

    def __init__(
        self, keys: List[str], logs_path: str = "", run_name: str = None, game_name: str = ''
    ) -> None:
        super().__init__(keys=keys, logs_path=logs_path, run_name=run_name, game_name=game_name)

    def log(self, values_dict: Dict[str, Dict[str, str]]) -> None:
        """
        Logs the given values to CSV files, one file per key. Each key's value is expected
        to be a dictionary with 'agent1' and 'agent2' as keys.

        Args:
            values_dict (Dict[str, Dict[str, str]]): A dictionary where each key-value pair represents
            a log entry. Each value is another dictionary with keys 'agent1' and 'agent2'.

        For each key in 'values_dict', this method appends a new row to the corresponding CSV file.
        The row contains the current timestamp, and the values for 'agent1' and 'agent2'.
        """
        for key, agents in values_dict.items():
            if key in self.keys:
                file_path = os.path.join(
                    self.logs_path, self.run_name, f"{key}.csv"
                )
                with open(file_path, "a", newline="") as file:
                    writer = csv.writer(file)

                    # Check if the value for the key is a dictionary with 'agent1' and 'agent2'
                    if isinstance(agents, dict) and 'agent1' in agents and 'agent2' in agents:
                        writer.writerow(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), agents['agent1'], agents['agent2']]
                        )
                    else:
                        raise ValueError(f'Value for key {key} must be a dictionary with "agent1" and "agent2"')
            else:
                raise ValueError(f'Uninitialized key: {key}')


def print_emotion_evolution(emotion_buffer):
    pass


def get_llm_response(model_name, messages):
    for x in messages:
        print(x)
    print('----------------------------')
    response = openai_client.chat.completions.create(
        model=model_name, messages=messages, temperature=0
    )
    print(response.choices[0].message.content)
    print('============================')
    return response.choices[0].message.content


def read_text(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return f.read()


def read_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)
