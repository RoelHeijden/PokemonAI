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
    - team preview is not extracted as a states
    
"""


def main():
    # mode = 'parse all games'
    # mode = 'create test split'
    # mode = 'create training batches'
    mode = 'inspect a game'

    inspect_battle_id = 4421651

    if mode == 'parse all games':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/raw-ou-incomplete"
        path_out = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/training_games'
        parse_all(path_in, path_out)

    if mode == 'create test split':
        train_path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/training_games'
        test_path = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/test_games'
        create_test_split(train_path, test_path, test_split=0.10)

    if mode == 'create training batches':
        path_in = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/test_games'
        path_out = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/test_states/1500+'
        f_out_name = 'ou_game_states'
        create_training_batches(path_in, path_out, f_out_name, min_rating=1500, batch_size=10000, min_game_length=3)

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
    files = [
        (
            os.path.join(train_path, file_name),
            os.path.join(test_path, file_name)
        )
        for file_name in os.listdir(train_path)]

    print(f'{len(files)} files collected')

    random.shuffle(files)
    n_test_files = int(test_split * len(files))
    test_files = files[: n_test_files]

    files_moved = 0
    for f_in, f_out in test_files:
        shutil.move(f_in, f_out)
        files_moved += 1
        if files_moved % 10000 == 0:
            print(f'{files_moved} files moved')

    print(f'finished, {files_moved} files moved')

    train = [os.path.join(train_path, file_name)
             for file_name in os.listdir(train_path)]

    test = [os.path.join(test_path, file_name)
            for file_name in os.listdir(test_path)]

    print(f'train files: {len(train)}')
    print(f'test files: {len(test)}')


def create_training_batches(path_in, path_out, f_out_name, min_rating=1200, batch_size=10000, min_game_length=3):
    """
    - extracts two game states from each game-states file
    - switches player's POV for one of the states
    - writes as batches to .jsonl files
    """

    print("collecting file names\n")

    # collect file names
    files = sorted([os.path.join(path_in, file_name)
                    for file_name in os.listdir(path_in)])

    random.shuffle(files)

    print(f'{len(files)} files found\n')

    start_time = time.time()

    n_games = 0
    n_states = 0
    n_written = 0

    games_too_short = {}
    ignore_list = json.load(open('games_to_ignore.json', 'r'))

    # initialize first out file
    f_out = open(os.path.join(path_out, f_out_name + '0.jsonl'), 'w')
    n_written += 1

    # open each game_states file
    for f in files:
        with open(f) as f_in:

            all_states = json.load(f_in)
            room_id = all_states[0]['roomid']

            n_games += 1

            # skip game if in the ignore list
            if ignore_list.get(str(room_id)):
                continue

            # skip if rating is too low
            if all_states[0]['average_rating'] < min_rating:
                continue

            # skip game if game doesn't last long enough
            if len(all_states) < min_game_length:
                games_too_short[room_id] = len(all_states)
                continue

            # select two random states to write
            states = pick_random_states(all_states, min_turns_apart=math.floor(min_game_length/3))

            # write each extracted state to batch file
            for s in states:
                f_out.write(ujson.dumps(s))
                f_out.write('\n')

                n_states += 1

            # close previous, and initialize new batch file
            if n_states % batch_size == 0:
                f_out.close()
                file = f_out_name + str(int(n_states / batch_size)) + '.jsonl'
                f_out = open(os.path.join(path_out, file), 'w')

                n_written += 1

            # keeping you updated
            if n_states % 5000 == 0:
                print(f'{n_games} games opened')
                print(f'{n_states} states extracted')
                print(f'{n_written} batch files created')
                print(f'runtime: {round(time.time() - start_time, 1)}s\n')

    print(ujson.dumps(games_too_short, indent=2), '\n')
    print(f"{len(games_too_short)} games skipped because they lasted shorter than {min_game_length} turns\n")

    print(f'{n_games} games opened')
    print(f'{n_states} states extracted')
    print(f'{n_written} batch files created with {batch_size} states each')
    print(f'final file contains {n_states % batch_size} states\n')

    print(f'runtime: {round(time.time() - start_time, 1)}s\n')


def pick_random_states(all_states, min_turns_apart=1):
    """ selects two random non-preview states from a game_state list,
        one for each of the players POV and at least n turns apart """

    # rolling from (1, ..) in order to skip the team_preview state
    state_idx1 = np.random.randint(1, len(all_states) - min_turns_apart)
    state_idx2 = np.random.randint(state_idx1 + min_turns_apart, len(all_states))

    state1 = all_states[state_idx1]
    state2 = all_states[state_idx2]

    return [state1, reverse_pov(state2)]


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

