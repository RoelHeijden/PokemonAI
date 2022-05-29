import ujson

from game_parser.state import Battle
from game_parser.state_updater import update_state
from Showdown_Pmariglia import constants


class GameLog:
    def __init__(self, info, battle_id):
        self.battle_id = battle_id

        self.format = info['format']
        self.turns = info['turns']
        self.winner = info['winner']
        self.score = info['score']

        self.p1 = info['p1']
        self.p1_team = info['p1team']
        self.p2 = info['p2']
        self.p2_team = info['p2team']

        self.seed = info['seed']
        self.timestamp = info['timestamp']
        self.end_type = info['endType']
        self.ladder_error = True if info.get('ladderError') else False

        self.input_log = info['inputLog']
        self.log = info['log']

        self.rated_battle = self.is_rated_battle(self.input_log)
        self.p1_rating = self.get_rating('p1', self.input_log, self.rated_battle)
        self.p2_rating = self.get_rating('p2', self.input_log, self.rated_battle)
        self.average_rating = self.get_avg_rating(self.p1_rating, self.p2_rating, self.rated_battle)

    def is_rated_battle(self, input_log):
        for string in input_log:
            if string.startswith('>start'):
                d = ujson.loads(string.strip('>start '))
                return d.get('rated') == 'Rated battle'

        raise KeyError("key '>start' not found in input_log of battle {}".format(self.battle_id))

    def get_rating(self, player, input_log, rated_battle):
        if not rated_battle:
            return 0

        for string in input_log:
            if string.startswith('>player ' + player):
                string = string.strip(('>player ' + player + ' '))

                # for some reason it's stored as: {""name": ..} but idk if that is always the case
                if string[1] == '"' and string[2] == '"':
                    string = string[:1] + string[2:]

                d = ujson.loads(string.strip('>player ' + player + ' '))
                return d.get("rating")

        raise KeyError("key '>player {}' not found in input_log of battle {}".format(player, self.battle_id))

    def get_avg_rating(self, p1_rating, p2_rating, rated_battle):
        if not rated_battle:
            return 0

        return int((p1_rating + p2_rating) / 2)

    def parse_replay(self):
        state = self.init_state()

        if self.winner == self.p1:
            winner = 'p1'
        elif self.winner == self.p2:
            winner = 'p2'
        else:
            winner = None

        basic_info = {
            "winner": winner,
            "p1rating": self.p1_rating,
            "p2rating": self.p2_rating,
            "average_rating": self.average_rating,
            "rated_battle": self.rated_battle,
            "roomid": self.battle_id,
        }

        team_preview_state = state.to_dict()
        team_preview_state.update(basic_info)

        # initialize states with team preview state
        game_states = [team_preview_state]

        # parsing the log line by line
        for i, line in enumerate(self.log):
            split_msg = line.split('|')

            if len(split_msg) < 2:
                continue

            if split_msg[1] == "win" or split_msg[1] == "tie":
                break

            # update the battle state
            else:
                bug_free_game = update_state(state, split_msg)
                if not bug_free_game:
                    return False

            # extract game state at the end of a turn, or when a Pokemon fainted
            if split_msg[1] == "turn" or split_msg[1] == "upkeep" and \
                    (state.p1.active.fainted or state.p2.active.fainted):
                d = state.to_dict()
                d.update(basic_info)
                game_states.append(d)

        return game_states

    def init_state(self):
        state = Battle(self.battle_id)

        state.p1.account_name = self.p1
        state.p2.account_name = self.p2

        state.p1.name = 'p1'
        state.p2.name = 'p2'

        state.battle_type = constants.RANDOM_BATTLE if 'randombattle' in self.format else constants.STANDARD_BATTLE
        state.generation = self.format[:4]

        state.initialize_team_preview(self.p1_team, self.p2_team)

        return state


