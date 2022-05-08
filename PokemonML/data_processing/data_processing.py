import os
import shutil
import time
import ujson

from game_parser.game_log import GameLog

"""
UNFINISHED PROGRAM
- Transform, Illusion and other form changes cause bugs
- Game state information needs to be extended


TO DO:
    1. test nicknamed pokemon
    2. add move inputs to state
    3. check ability if statements in start_volatile_status()
    4. actually edit the Transform function
    5. check switch_or_drag transform handling (it resets the pokemon..)
    6. test zoroark
"""


def main():
    # mode = 'read from batch'
    mode = 'create batches'
    # mode = 'test a file'

    batch = 0
    battle_id = 100165

    if mode == 'read from batch':
        data_folder = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/python/PokemonML/data/granted_data_testing"
        path_in = os.path.join(data_folder, "raw_data", "batch" + str(batch))
        path_out = os.path.join(data_folder, "processed_data", "anonymized-ou-incomplete")
        parse_all(path_in, path_out)

    if mode == 'create batches':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/anonymized-ou-Dec2019-Feb2020/anonymized-ou-incomplete"
        path_out = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/python/PokemonML/data/granted_data_testing/raw_data"
        copy_to_batches(path_in, path_out, 10, 50)

    if mode == 'test a file':
        path_in = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/python/PokemonML/data/granted_data_testing/raw_data"
        file = os.path.join(path_in, 'batch' + str(batch), 'battle-gen8ou-' + str(battle_id) + '.log.json')
        f = open(file, 'r')

        info = None
        for line in f:
            info = ujson.loads(line)

        for line in info['log']:
            print(line)


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

    for path_in, file_out, battle_id in files:
        with open(path_in, "r", encoding='utf-8') as f_in:
            for line in f_in:

                info = ujson.loads(line)
                print(info)

                # only check games post dynamax ban, which I think was on december 17
                date = info['timestamp'].split(" ")
                if date[1] == 'Dec' and int(date[2]) < 18:
                    continue

                # extract game states
                game = GameLog(info, battle_id)
                parsed_replay = game.parse_replay()

                # sort based on average rating
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

                # create directory if it doesn't exist
                folder_out = os.path.join(save_path, folder)
                if not os.path.exists(folder_out):
                    os.mkdir(folder_out)

                # write parsed file
                path_out = os.path.join(folder_out, file_out)
                f_out = open(path_out, "w+")
                f_out.write(ujson.dumps(parsed_replay))

                match_count += 1

                if match_count % 100 == 0:
                    print(f'{match_count} games processed, {time.time() - tic} passed')

    toc = time.time()

    print("Match count: {}".format(match_count))
    print("Total time: {:.2f}s".format(toc - tic))


def write_readable(file_path, save_path, max_turns=20):
    """ Writes new readable file """
    with open(file_path, "r") as f_in, open(save_path, 'w') as f_out:
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
    main()

