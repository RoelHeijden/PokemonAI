import ujson

from state import Battle
from state_updater import update_state
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
        d = self.__dict__
        battle = self.init_battle()

        d['team_preview_state'] = battle.to_dict()

        # parsing the log line by line
        game_states = {}
        for i, line in enumerate(self.log):
            split_msg = line.split('|')

            if len(split_msg) < 2:
                continue

            if split_msg[1] == "win" or split_msg[1] == "tie":
                break

            # update the battle state
            else:
                update_state(battle, split_msg)

            # extract game state
            if split_msg[1] == "turn":
                game_states["turn " + str(battle.turn)] = battle.extract_game_state()

        d['game_states'] = game_states
        return d

    def init_battle(self):
        battle = Battle(self.battle_id)

        # battle.p1.account_name = self.p1
        # battle.p2.account_name = self.p2
        battle.p1.name = 'p1'
        battle.p2.name = 'p2'

        battle.battle_type = constants.RANDOM_BATTLE if 'randombattle' in self.format else constants.STANDARD_BATTLE
        battle.generation = self.format[:4]

        battle.initialize_team_preview(self.p1_team, self.p2_team)
        return battle
