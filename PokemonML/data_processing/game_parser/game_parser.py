import os
import shutil
import time
import ujson
import json

from game_log import GameLog


def main():
    mode = 'read from folder'
    # mode = 'read from batch'
    # mode = 'create batches'
    # mode = 'pre-process data'
    # mode = 'inspect a log'

    folder = 'all_rated_1200+'
    battle_id = 00000000

    batch = 0

    n_batches = 4
    batch_size = 50

    if mode == 'read from folder':
        folder_path = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete"
        path_in = os.path.join(folder_path, folder)
        path_out = 'C:/Users/RoelH/Documents//Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+'
        parse_all(path_in, path_out)

    if mode == 'read from batch':
        data_folder = "../data/granted_data_testing"
        path_in = os.path.join(data_folder, "raw_data", "batch" + str(batch))
        path_out = os.path.join(data_folder, "processed_data")
        parse_all(path_in, path_out)

    if mode == 'create batches':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete"
        path_in = os.path.join(path_in, folder)
        path_out = "../data/granted_data_testing/raw_data"
        copy_to_batches(path_in, path_out, n_batches, batch_size)

    if mode == 'pre-process data':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/anonymized-ou-Dec2019-Feb2020/anonymized-ou-incomplete"
        path_out = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete"
        pre_pre_processing(path_in, path_out)

    if mode == 'inspect a log':
        folder_path = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/pre-processed-ou-dec2019-feb2022/anonymized-ou-incomplete"
        path_in = os.path.join(folder_path, folder)

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

                if match_count % 20000 == 0:
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
                    print(f'{match_count} games processed, {round(time.time() - tic)} seconds passed')

    toc = time.time()

    print("Match count: {}".format(match_count))
    print("Total time: {:.2f}s".format(toc - tic))


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
    main()

