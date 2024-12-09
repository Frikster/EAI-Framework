class DivisionGame:
    def __init__(self, name, total_sum, do_second_step, coplayer_name):
        self.name = name
        self.total_sum = total_sum
        self.do_second_step = do_second_step
        self.coplayer_name = coplayer_name
        self.update_format = """Your {coplayer} splitted {total_sum} as following: he keeps {keep_sum} dollars and gives you {give_sum} dollars.
Do you accept the split?"""

    def run(self, agent1, agent2, logger):
        agent1.init_memory()
        agent2.init_memory()
        agent1._round_question_format = agent1._round_question_format.format(total_sum=self.total_sum,
                                                                             coplayer=self.coplayer_name)
        step1, scratchpad_step1 = self.check_for_scatchpad(agent1.make_step(""))
        agent1_sum_part, agent2_sum_part = self.parse_answer1(step1)
        step2, scratchpad_step2 = '', ''
        if self.name == 'ultimatum' and self.do_second_step:
            cur_update = self.update_format.format(coplayer=self.coplayer_name,
                                                  total_sum=self.total_sum,
                                                  keep_sum=agent1_sum_part,
                                                  give_sum=agent2_sum_part)
            agent2._round_question_format = agent2._round_question_format.format(coplayer=self.coplayer_name,
                                                                                  total_sum=self.total_sum,
                                                                                  keep_sum=agent1_sum_part,
                                                                                  give_sum=agent2_sum_part)
            agent2._add_to_history("user", cur_update)
            step2, scratchpad_step2 = self.check_for_scatchpad(agent2.make_step(""))
            is_accepted = self.parse_answer2(step2)
        logger.log({"div_decisions": {"agent1": step1, "agent2": step2}})
        logger.log({"div_decisions_scratchpad": {"agent1": scratchpad_step1, "agent2": scratchpad_step2}})

    def parse_answer1(self, answer: str):
        def to_digit(agent_sum):
            agent_sum = agent_sum.replace(',', '')
            agent_sum = agent_sum.strip(' .,:\t\n$')
            try:
                return int(agent_sum)
            except:
                return float(agent_sum)
        answer = answer.strip(' .,:\t\n')
        answer_splitted = answer.split(';')
        # print(answer)
        # if len(answer_splitted) > 2:
        #     new_answer_splitted = [answer_splitted[0]]
        #     for i in range(1, len(answer_splitted)):
        #         if not answer_splitted[i].startswith('0'):
        if len(answer_splitted) != 2:
            print(answer)
        assert len(answer_splitted) == 2
        agent1_sum_part, agent2_sum_part = answer_splitted
        return to_digit(agent1_sum_part), to_digit(agent2_sum_part)

    def parse_answer2(self, answer: str):
        answer = answer.strip(' .,:\t\n$')

        if answer.upper() == 'ACCEPT':
            return True
        elif answer.upper() == 'REJECT':
            return False
        raise Exception()

    def check_for_scatchpad(self, step):
        if isinstance(step, tuple):
            step, scratchpad_step = step
        else:
            step, scratchpad_step = step, None
        return step, scratchpad_step
