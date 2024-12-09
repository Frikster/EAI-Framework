import pandas as pd
from typing import List, Tuple, Dict
import os

class BasicEvaluation:
    """
    A basic class for evaluating logs.

    Attributes:
        keys (List[str]): A list of keys for which logs are to be evaluated.
        run_name (str): The name of the run for which logs are being evaluated.
        log_folder (str): The folder where log files are stored.
        logs (Dict[str, pd.DataFrame]): A dictionary mapping keys to their corresponding log dataframes.
    """

    def __init__(self, keys: List[str], run_name: str, log_folder: str = '', header: int = None) -> None:
        """
        Initializes a BasicEvaluation instance.

        Args:
            keys (List[str]): A list of keys for log evaluation.
            run_name (str): The name of the run for log files.
            log_folder (str): The folder where log files are located.
            header (int, optional): The row number to use as the column names. None means that the first row is not treated as a header.
        """
        self.keys = keys
        self.run_name = run_name
        self.log_folder = log_folder

        self.logs = {key: pd.read_csv(os.path.join(log_folder, run_name, f'{key}.csv'), header=header) for key in keys}

    def get_metric(self) -> None:
        """
        A method to get metrics from the logs. To be implemented in subclasses.
        """
        pass

class DecisionStatistics(BasicEvaluation):
    """
    A subclass of BasicEvaluation focused on decision statistics.

    Inherits all attributes from BasicEvaluation.
    """

    def __init__(self, run_name: str, log_folder: str = '', run_key: str = 'decisions') -> None:
        """
        Initializes a DecisionStatistics instance.

        Args:
            run_name (str): The name of the run for log files.
            log_folder (str): The folder where log files are located.
        """
        super().__init__([run_key], run_name, log_folder=log_folder)
        self.run_key = run_key

    def get_metric(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculates and returns decision statistics.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A dataframe of count statistics for each agent and a series of count combinations.
        """
        decision_log = self.logs[self.run_key]
        count_1 = decision_log.iloc[:, 1].value_counts()
        count_2 = decision_log.iloc[:, 2].value_counts()
        decision_stats = pd.DataFrame([count_1, count_2], index=['agent_1', 'agent_2'])

        decision_log['combination'] = decision_log.iloc[:, 1] + decision_log.iloc[:, 2]
        count_combinations = decision_log['combination'].value_counts()

        return decision_stats, count_combinations