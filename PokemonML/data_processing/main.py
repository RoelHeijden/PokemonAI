import json
import os
import numpy as np
import time
import ujson
import math
import random
import shutil

from game_parser.game_log import GameLog


"""

TO DO:
    - implement last_used_move
    - implement Zoroark/Illusion
    - implement turn counts for Encore, Lightscreen, etc.

NOTEWORTHY:
    - states includes moments in a turn where one of the sides has a fainted Pokemon
    - team preview is not extracted as a state
    
"""


def main():
    # mode = 'parse all games'
    # mode = 'create test split'
    # mode = 'create training batches'
    mode = 'inspect a game'

    inspect_battle_id = 5685788

    if mode == 'parse all games':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/raw-ou-incomplete"
        path_out = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/training_games'
        parse_all(path_in, path_out)

    if mode == 'create test split':
        train_path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/training_games'
        test_path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games'
        create_test_split(train_path, test_path, test_split=0.10)

    if mode == 'create training batches':
        path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/test_games/1000+'
        path_out = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/experiment_data/game_length'
        f_out_name = 'ou_game_states'
        create_training_batches(path_in, path_out, f_out_name, file_size=10000, min_game_length=3)

    if mode == 'inspect a game':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/raw-ou-incomplete"
        file = os.path.join(path_in, 'battle-gen8ou-' + str(inspect_battle_id) + '.log.json')
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

    print("collecting file names\n")

    # collect files
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

        # open raw battle json
        with open(path_in, "r", encoding='utf-8') as f_in:
            for line in f_in:

                info = ujson.loads(line)

                # only check games post dynamax ban, which was on december 17th. Checking 18th to be sure
                date = info['timestamp'].split(" ")
                if date[1] == 'Dec' and int(date[2]) <= 18:
                    continue

                # replace the 'null' values
                rated_battle, p1_rating, p2_rating = get_player_ratings(info['inputLog'], battle_id)
                info['rated_battle'] = rated_battle
                info['p1rating'] = p1_rating
                info['p2rating'] = p2_rating
                info['roomid'] = battle_id
                info['average_rating'] = int((p1_rating + p2_rating) / 2)

                # parse game and extract game states
                game = GameLog(info, battle_id)
                parsed_game = game.parse_replay()

                # parse_replay() returns False if the game is bugged
                if not parsed_game:
                    continue

                # write parsed game_states file
                path_out = os.path.join(save_path, file_out)
                f_out = open(path_out, "w+")
                f_out.write(ujson.dumps(parsed_game))

                match_count += 1

                if match_count % 10000 == 0:
                    print(f'{match_count} games processed, {round(time.time() - tic)}s')

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


def create_test_split(train_path, test_path, test_split=0.10):
    """ moves files from the training folder to test folder """

    # collect files
    files = [
        (
            os.path.join(train_path, file_name),
            file_name
        )
        for file_name in os.listdir(train_path)]

    print(f'{len(files)} files collected')

    # output folders
    folder_all = 'all'
    folder_1100 = '1100+'
    folder_1300 = '1300+'
    folder_1500 = '1500+'
    folder_1700 = '1700+'

    # create random test split
    random.shuffle(files)
    n_test_files = int(test_split * len(files))
    test_files = files[: n_test_files]

    files_moved = 0

    # iterate over allocated test files
    for f, file_name in test_files:

        # get rating
        with open(f, 'r') as f_in:
            game_states = json.load(f_in)
            rating = game_states[0]['average_rating']

        # copy file to rating folders
        if rating >= 1100:
            shutil.copy(f, os.path.join(test_path, folder_1100, file_name))
        if rating >= 1300:
            shutil.copy(f, os.path.join(test_path, folder_1300, file_name))
        if rating >= 1500:
            shutil.copy(f, os.path.join(test_path, folder_1500, file_name))
        if rating >= 1700:
            shutil.copy(f, os.path.join(test_path, folder_1700, file_name))

        # move file to the 'all' test folder
        shutil.move(f, os.path.join(test_path, folder_all, file_name))

        files_moved += 1

        if files_moved % 10000 == 0:
            print(f'{files_moved} files moved')

    print(f'finished, {files_moved} files moved')

    # show current file split
    train = [os.path.join(train_path, file_name)
             for file_name in os.listdir(train_path)]

    test = [os.path.join(test_path, file_name)
            for file_name in os.listdir(test_path)]

    print(f'train files: {len(train)}')
    print(f'test files: {len(test)}')


def create_training_batches(path_in, path_out, f_out_name, file_size=10000, min_game_length=3):
    """
    - extracts n random game states from each game-states file
    - write states to .jsonl files
    """

    print("collecting file names\n")

    # collect file names
    files = sorted([os.path.join(path_in, file_name)
                    for file_name in os.listdir(path_in)])

    random.shuffle(files)

    print(f'{len(files)} files found\n')

    # these games won't be represented in the batches
    games_too_short = {}
    ignore_list = json.load(open('games_to_ignore.json', 'r'))

    # # output folders
    # folder_all = 'all'
    # folder_1000_1199 = '1000-1199'
    # folder_1200_1399 = '1200-1399'
    # folder_1400_1599 = '1400-1599'
    # folder_1600 = '1600+'
    #
    # # initialize and open first out files
    # f_out_all = open(os.path.join(path_out, folder_all, f_out_name + '0.jsonl'), 'w')
    # f_out_1000_1199 = open(os.path.join(path_out, folder_1000_1199, f_out_name + '0.jsonl'), 'w')
    # f_out_1200_1399 = open(os.path.join(path_out, folder_1200_1399, f_out_name + '0.jsonl'), 'w')
    # f_out_1400_1599 = open(os.path.join(path_out, folder_1400_1599, f_out_name + '0.jsonl'), 'w')
    # f_out_1600 = open(os.path.join(path_out, folder_1600, f_out_name + '0.jsonl'), 'w')
    #
    # n_states_all = 0
    # n_states_1000_1199 = 0
    # n_states_1200_1399 = 0
    # n_states_1400_1599 = 0
    # n_states_1600 = 0

    # output folders
    folder_2_9 = '2-9'
    folder_10_19 = '10-19'
    folder_20_29 = '20-29'
    folder_30_39 = '30-39'
    folder_40_49 = '40-49'
    folder_50 = '50+'

    # initialize and open first out files
    f_out_2_9 = open(os.path.join(path_out, folder_2_9, f_out_name + '0.jsonl'), 'w')
    f_out_10_19 = open(os.path.join(path_out, folder_10_19, f_out_name + '0.jsonl'), 'w')
    f_out_20_29 = open(os.path.join(path_out, folder_20_29, f_out_name + '0.jsonl'), 'w')
    f_out_30_39 = open(os.path.join(path_out, folder_30_39, f_out_name + '0.jsonl'), 'w')
    f_out_40_49 = open(os.path.join(path_out, folder_40_49, f_out_name + '0.jsonl'), 'w')
    f_out_50 = open(os.path.join(path_out, folder_50, f_out_name + '0.jsonl'), 'w')

    n_states_2_9 = 0
    n_states_10_19 = 0
    n_states_20_29 = 0
    n_states_30_39 = 0
    n_states_40_49 = 0
    n_states_50 = 0

    n_games = 0
    total_game_length = 0
    total_n_states = 0

    start_time = time.time()

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
                games_too_short[room_id] = len(all_states)
                continue

            rating = min(all_states[0]['p1rating'], all_states[0]['p2rating'])
            game_length = all_states[-1]['turn']

            n_games += 1
            total_game_length += game_length
            total_n_states += len(all_states) - 1

            n_samples = min(10, len(all_states) - 1)
            states = random.sample(all_states[1:], n_samples)

            # 2-9
            if 2 <= game_length <= 9:
                f_out_2_9 = write_to_batch(states, f_out_2_9, n_states_2_9, file_size, f_out_name, path_out, folder_2_9)
                n_states_2_9 += len(states)

            # 10-19
            if 10 <= game_length <= 19:
                f_out_10_19 = write_to_batch(states, f_out_10_19, n_states_10_19, file_size, f_out_name, path_out, folder_10_19)
                n_states_10_19 += len(states)

            # 20-29
            if 20 <= game_length <= 29:
                f_out_20_29 = write_to_batch(states, f_out_20_29, n_states_20_29, file_size, f_out_name, path_out, folder_20_29)
                n_states_20_29 += len(states)

            # 30-39
            if 30 <= game_length <= 39:
                f_out_30_39 = write_to_batch(states, f_out_30_39, n_states_30_39, file_size, f_out_name, path_out, folder_30_39)
                n_states_30_39 += len(states)

            # 40-49
            if 40 <= game_length <= 49:
                f_out_40_49 = write_to_batch(states, f_out_40_49, n_states_40_49, file_size, f_out_name, path_out, folder_40_49)
                n_states_40_49 += len(states)

            # 50+
            if 50 <= game_length:
                f_out_50 = write_to_batch(states, f_out_50, n_states_50, file_size, f_out_name, path_out, folder_50)
                n_states_50 += len(states)

            # updates
            if n_games % 2000 == 0:
                print(f'{n_games} games sampled')
                print(f'states extracted: \n'
                      f'  2-9: {n_states_2_9}, \n'
                      f'  10-19: {n_states_10_19}, \n'
                      f'  20-29: {n_states_20_29}, \n'
                      f'  30-39: {n_states_30_39}, \n'
                      f'  40-49: {n_states_40_49}, \n'
                      f'  50+: {n_states_50}')
                print(f'runtime: {round(time.time() - start_time, 1)}s\n')

            # # all
            # f_out_all = write_to_batch(states, f_out_all, n_states_all, file_size, f_out_name, path_out, folder_all)
            # n_states_all += len(states)
            #
            # 1000-1199
            # if 1000 <= rating <= 1199:
            #     f_out_1000_1199 = write_to_batch(states, f_out_1000_1199, n_states_1000_1199, file_size, f_out_name, path_out, folder_1000_1199)
            #     n_states_1000_1199 += len(states)
            #
            # # 1200-1399
            # if 1200 <= rating <= 1399:
            #     f_out_1200_1399 = write_to_batch(states, f_out_1200_1399, n_states_1200_1399, file_size, f_out_name, path_out, folder_1200_1399)
            #     n_states_1200_1399 += len(states)
            #
            # # 1400-1599
            # if 1400 <= rating <= 1599:
            #     f_out_1400_1599 = write_to_batch(states, f_out_1400_1599, n_states_1400_1599, file_size, f_out_name, path_out, folder_1400_1599)
            #     n_states_1400_1599 += len(states)
            #
            # # 1600+
            # if 1600 <= rating:
            #     f_out_1600 = write_to_batch(states, f_out_1600, n_states_1600, file_size, f_out_name, path_out, folder_1600)
            #     n_states_1600 += len(states)
            #
            # # updates
            # if n_games % 2000 == 0:
            #     print(f'{n_games} games sampled')
            #     print(f'states extracted: \n'
            #           f'  all: {n_states_all}, \n'
            #           f'  1000-1199: {n_states_1000_1199}, \n'
            #           f'  1200-1399: {n_states_1200_1399}, \n'
            #           f'  1400-1599: {n_states_1400_1599}, \n'
            #           f'  1600+: {n_states_1600}')
            #     print(f'runtime: {round(time.time() - start_time, 1)}s\n')

    print(ujson.dumps(games_too_short, indent=2), '\n')
    print(f"{len(games_too_short)} games skipped because they lasted shorter than {min_game_length} turns\n")

    # print(f'{n_games} games sampled')
    # print(f'states extracted: \n'
    #       f'  all: {n_states_all}, \n'
    #       f'  1000-1199: {n_states_1000_1199}, \n'
    #       f'  1200-1399: {n_states_1200_1399}, \n'
    #       f'  1400-1599: {n_states_1400_1599}, \n'
    #       f'  1600+: {n_states_1600}')

    print(f'{n_games} games sampled')
    print(f'states extracted: \n'
          f'  2-9: {n_states_2_9}, \n'
          f'  10-19: {n_states_10_19}, \n'
          f'  20-29: {n_states_20_29}, \n'
          f'  30-39: {n_states_30_39}, \n'
          f'  40-49: {n_states_40_49}, \n'
          f'  50+: {n_states_50}')

    print(f'average n states: {total_n_states / n_games:.3f}')
    print(f'average game length: {total_game_length / n_games:.3f}\n')

    print(f'runtime: {round(time.time() - start_time, 1)}s\n')


def write_to_batch(states, f_out, n_states, file_size, f_out_name, path_out, folder):

    # write each extracted state to batch file
    for s in states:
        f_out.write(ujson.dumps(s))
        f_out.write('\n')

        n_states += 1

        # close previous, and initialize new batch file
        if n_states % file_size == 0:
            f_out.close()

            # open new file
            f_out_name = f_out_name + str(int(n_states / file_size)) + '.jsonl'
            f_out = open(os.path.join(path_out, folder, f_out_name), 'w')

    return f_out


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

