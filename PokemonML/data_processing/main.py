
import json
import os
import numpy as np
import time
import ujson
import math

from game_parser.game_log import GameLog


def main():
    # mode = 'read from folder'
    # mode = 'pre-process data'
    mode = 'extract training states'
    # mode = 'inspect a log'

    battle_id = 1021737

    if mode == 'read from folder':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete/all_rated_1200+"
        path_out = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_games'
        parse_all(path_in, path_out)

    if mode == 'pre-process data':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/anonymized-ou-Dec2019-Feb2020/anonymized-ou-incomplete"
        path_out = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete"
        pre_pre_processing(path_in, path_out)

    if mode == 'extract training states':
        path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_games'
        path_out = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states'
        extract_training_states(path_in, path_out)

    if mode == 'inspect a log':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete/all_rated_1200+"

        file = os.path.join(path_in, 'battle-gen8ou-' + str(battle_id) + '.log.json')
        f = open(file, 'r')

        info = None
        for line in f:
            info = ujson.loads(line)
        for line in info['log']:
            print(line)
        print()
        for mon in info['p1team']:
            print(mon)
        print()
        for mon in info['p2team']:
            print(mon)


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

    print(f'{len(files)} files found\n')
    print("starting parsing")

    match_count = 0
    tic = time.time()

    ignore_list = json.load(open('games_to_ignore.json', 'r'))

    for path_in, file_out, battle_id in files:

        # skip battle if game is in the ignore list
        if ignore_list.get(str(battle_id)):
            continue

        with open(path_in, "r", encoding='utf-8') as f_in:
            for line in f_in:

                info = ujson.loads(line)

                # only check games post dynamax ban, which I think was on december 17
                date = info['timestamp'].split(" ")
                if date[1] == 'Dec' and int(date[2]) < 18:
                    continue

                # extract game states
                game = GameLog(info, battle_id)
                parsed_replay = game.parse_replay()

                # write parsed file
                path_out = os.path.join(save_path, file_out)
                f_out = open(path_out, "w+")
                f_out.write(ujson.dumps(parsed_replay))

                match_count += 1

                if match_count % 10000 == 0:
                    print(f'{match_count} games processed, {round(time.time() - tic)}s')

    toc = time.time()

    print("Match count: {}".format(match_count))
    print("Total time: {:.2f}s".format(toc - tic))


def pre_pre_processing(folder_path, save_path):
    """ does some preprocessing:
        - removes games played before december 18th (pre dmax ban)
        - sort games to folders based on average rating
        - changes 'null' variables
    """

    print("creating file names")

    files = sorted(
        [
            (
                os.path.join(folder_path, file_name),
                file_name,
                int(file_name.split('-')[2].strip('.log.json'))
            )
            for file_name in os.listdir(folder_path)
        ]
    )

    print(f'{len(files)} files found\n')

    match_count = 0
    tic = time.time()

    for path_in, file_out, battle_id in files:
        with open(path_in, "r", encoding='utf-8') as f_in:
            for line in f_in:

                info = ujson.loads(line)

                # only check games post dynamax ban, which I think was on december 17
                date = info['timestamp'].split(" ")
                if date[1] == 'Dec' and int(date[2]) < 18:
                    continue

                # get rating and edit null variables
                rated_battle, p1_rating, p2_rating = get_player_ratings(info['inputLog'], battle_id)
                info['rated_battle'] = rated_battle
                info['p1rating'] = p1_rating
                info['p2rating'] = p2_rating
                info['roomid'] = battle_id
                info['average_rating'] = int((p1_rating + p2_rating) / 2)

                # sort based on average rating
                if info.get('rated_battle'):
                    rating = info['average_rating']
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
                            "Rated battle rating {} not between 999 and inf -- battle id: {}".format(rating, battle_id)
                        )
                else:
                    folder = "unrated"

                # create directory if it doesn't exist
                folder_out = os.path.join(save_path, folder)
                if not os.path.exists(folder_out):
                    os.mkdir(folder_out)

                # write new file
                path_out = os.path.join(folder_out, file_out)
                f_out = open(path_out, "w+")
                f_out.write(ujson.dumps(info))

                match_count += 1

                if match_count % 10000 == 0:
                    print(f'{match_count} games processed, {round(time.time() - tic)} seconds passed')

    toc = time.time()

    print("Match count: {}".format(match_count))
    print("Total time: {:.2f}s".format(toc - tic))


def get_player_ratings(game_input_log, battle_id):
    """" extract average rating from a game, using the inputLog """

    def is_rated_battle(input_log):
        for string in input_log:
            if string.startswith('>start'):
                d = ujson.loads(string.strip('>start '))
                return d.get('rated') == 'Rated battle'
        raise KeyError("key '>start' not found in input_log of battle {}".format(battle_id))

    def get_rating(player, input_log, rated_battle):
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
        raise KeyError("key '>player {}' not found in input_log of battle {}".format(player, battle_id))

    is_rated_battle = is_rated_battle(game_input_log)
    p1_rating = get_rating('p1', game_input_log, is_rated_battle)
    p2_rating = get_rating('p2', game_input_log, is_rated_battle)

    return is_rated_battle, p1_rating, p2_rating


def extract_training_states(path_in, path_out, min_game_length=3):
    """ converts each game into one or two number-converted game states """

    print("starting\n")

    # collect file names
    files = sorted([os.path.join(path_in, file_name)
                    for file_name in os.listdir(path_in)])

    print(f'{len(files)} files found\n')

    start_time = time.time()
    n_states = 0
    n_games = 0

    games_too_short = {}
    ignore_list = json.load(open('games_to_ignore.json', 'r'))

    # open each game_states file
    for f in files:
        with open(f) as f_in:

            all_states = json.load(f_in)
            room_id = all_states[0]['roomid']

            # skip game if in the ignore list
            if ignore_list.get(str(room_id)):
                continue

            # skip game if game doesn't last long enough
            if len(all_states) < min_game_length:
                games_too_short[f] = len(all_states)
                continue

            # select two random states to write
            states, state_nums = pick_random_states(all_states, min_turns_apart=math.floor(min_game_length/3))

            # write each extracted state to separate file
            for i, (s, n) in enumerate(zip(states, state_nums)):

                file_name = os.path.join(path_out, 'gen8ou-' + str(room_id) + '-state-' + str(n) + '.json')
                with open(file_name, 'w') as f_out:
                    f_out.write(ujson.dumps(s))

                n_states += 1

            n_games += 1

            if n_games % 2000 == 0:
                print(f'{n_games} games opened')
                print(f'{n_states} states extracted')
                print(f'runtime: {round(time.time() - start_time, 1)}s\n')

    print("finished\n")

    print(f"{len(games_too_short)} games skipped because they lasted shorter than {min_game_length} turns")
    print(ujson.dumps(games_too_short))

    print(f'\n{n_games} games opened, {n_states} states extracted')
    print(f'runtime: {round(time.time() - start_time, 1)}s\n')


def pick_random_states(all_states, min_turns_apart=1):
    """ selects two random non-preview states from a game_state list,
        one for each of the players POV and at least n turns apart """

    # rolling from (1, ..) in order to skip the team_preview state
    state_idx1 = np.random.randint(1, len(all_states) - min_turns_apart)
    state_idx2 = np.random.randint(state_idx1 + min_turns_apart, len(all_states))

    state1 = all_states[state_idx1]
    state2 = all_states[state_idx2]

    return [state1, reverse_pov(state2)], [state_idx1, state_idx2]


def reverse_pov(state):
    """ flips the POV of a state """
    if state['winner'] == 'p1':
        state['winner'] = 'p2'
    elif state['winner'] == 'p2':
        state['winner'] = 'p1'
    else:
        # tie
        pass

    hold_my_beer = state['p1']
    state['p1'] = state['p2']
    state['p2'] = hold_my_beer

    hold_my_beer = state['p1rating']
    state['p1rating'] = state['p2rating']
    state['p2rating'] = hold_my_beer

    return state


if __name__ == "__main__":
    main()

