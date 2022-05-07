import os
import shutil
import time
import ujson

from battle import Battle
from battle_modifier import update_battle
from Showdown_Pmariglia import constants


class ReplayData:
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
                update_battle(battle, split_msg)

            # extract game state
            if split_msg[1] == "turn":
                game_states["turn " + str(battle.turn)] = battle.extract_game_state()

        d['game_states'] = game_states
        return d

    def init_battle(self):
        battle = Battle(self.battle_id)

        battle.user.account_name = self.p1
        battle.opponent.account_name = self.p2
        battle.user.name = 'p1'
        battle.opponent.name = 'p2'

        battle.battle_type = constants.RANDOM_BATTLE if 'randombattle' in self.format else constants.STANDARD_BATTLE
        battle.generation = self.format[:4]

        battle.initialize_team_preview(self.p1_team, self.p2_team)
        return battle


def parse_all(folder_path, save_path):
    """ Parses the game states, writes new json files to folders sorted on average rating """

    print("creating file names")

    files = sorted(
        [
            (
                os.path.join(folder_path, file_name),
                file_name.replace("battle", "game-states").replace('.log', ''),
                int(file_name.split('-')[2].strip('.log.json'))
            )
            for file_name in os.listdir(folder_path)
        ]
    )

    print(f'{len(files)} files found')
    print("starting parsing")

    match_count = 0
    tic = time.time()

    for path_in, file_out, battle_id in files:
        with open(path_in, "r") as f_in:
            for line in f_in:

                print("battle ID:", battle_id)

                info = ujson.loads(line)
                date = info['timestamp'].split(" ")

                # only check games post dynamax ban, which was december 17 I think
                if date[1] == 'Dec' and int(date[2]) < 18:
                    continue

                replay_data = ReplayData(info, battle_id)
                parsed_replay = replay_data.parse_replay()

                if parsed_replay.get('rated_battle'):
                    rating = parsed_replay['average_rating']
                    if 1000 <= rating <= 1199:
                        folder = "rated_1000_1199"
                    elif 1200 <= rating <= 1399:
                        folder = "rated_1200_1399"
                    elif 1400 <= rating <= 1599:
                        folder = "rated_1400_1599"
                    elif 1600 <= rating <= 1799:
                        folder = "rated_1600_1799"
                    elif 1800 <= rating:
                        folder = "rated_1800+"
                    else:
                        raise ValueError(
                            "Rated battle rating {} not between 999 and inf -- room id: {}".format(rating, battle_id)
                        )
                else:
                    folder = "unrated"

                path_out = os.path.join(save_path, folder, file_out)
                f_out = open(path_out, "w+")
                f_out.write(ujson.dumps(parsed_replay))

                match_count += 1

                if match_count % 100 == 0:
                    print(f'{match_count} games processed, {time.time() - tic} passed')

    toc = time.time()

    print("Match count: {}".format(match_count))
    print("Total time: {:.2f}s".format(toc - tic))


def pretty_print(save_path: str, pretty_path: str, max_turns=20):
    """ Writes new readable file """
    with open(save_path, "r") as f_in, open(pretty_path, 'w') as f_out:
        for line in f_in:
            info = ujson.loads(line)

            f_out.write(ujson.dumps(info['log'], indent=2))

            for i, state in enumerate(info['game_states'].values(), start=1):

                f_out.write("\n\n{} {} {} \n\n".format("#" * 60, ' turn ' + str(state['turn']) + ' ', "#" * 60))
                f_out.write(ujson.dumps(state, indent=4))

                if i == max_turns:
                    break


def copy_to_batches(data_folder, batch_location, n_batches=10, max_batch_size=100):
    """ copies the files, divided over n folders of folder size m """

    for i, file_name in enumerate(os.listdir(data_folder)):

        # break if batches are full
        if i >= max_batch_size * n_batches:
            break

        # create save path
        batch = "batch" + str(i % n_batches)
        batch_folder = os.path.join(batch_location, batch)

        # create directory
        if not os.path.exists(batch_folder):
            os.mkdir(batch_folder)

        # copy file
        file_path = os.path.join(data_folder, file_name)
        save_path = os.path.join(batch_folder, file_name)
        shutil.copy(file_path, save_path)


if __name__ == "__main__":
    # path_in = os.path.join("granted_data_testing", "raw_data", "batch1")
    # path_out = os.path.join("granted_data_testing", "processed_data", "anonymized-ou-incomplete")
    # parse_all(path_in, path_out)

    # path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/anonymized-ou-Dec2019-Feb2020/anonymized-ou-incomplete"
    # path_out = os.path.join("granted_data_testing", "raw_data")
    # copy_to_batches(path_in, path_out)

    path_in = os.path.join("granted_data_testing", "raw_data", "batch1")
    file = os.path.join(path_in, 'battle-gen8ou-100161.log.json')
    f = open(file, 'r')

    info = None
    for line in f:
        info = ujson.loads(line)

    for line in info['log']:
        print(line)

