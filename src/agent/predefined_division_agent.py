class PredefinedRatioDivisionAgent:
    def __init__(self, ratio, total_sum):
        self.ratio = ratio
        self.total_sum = total_sum
        self._round_question_format = ''

    def init_memory(self):
        pass

    def make_step(self, step_num):
        keep_sum = int(self.total_sum * self.ratio)
        give_sum = self.total_sum - keep_sum
        return f"{keep_sum};{give_sum}"

