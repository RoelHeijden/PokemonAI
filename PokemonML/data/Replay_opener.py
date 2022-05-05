from urllib.request import Request, urlopen
import io
import os


def write_to_json(input_file, output_folder):
    """ Opens a list of replay urls as .json files writes to a folder location """
    replays = open(input_file, "rt")
    json_list = []

    for i, replay in enumerate(replays):
        if '\n' in replay:
            replay = replay.rstrip('\n')

        # opens replay as json file
        replay = replay + ".json"
        req = Request(replay, headers={'User-Agent': 'Mozilla/5.0'})
        replay_json = urlopen(req).read()

        # convert from bytes to string
        replay_json = replay_json.decode("utf-8")

        json_list.append(replay_json + '\n')

        if (i+1) % 20 == 0:
            print(f'{i+1} replays converted')

        # write to location file
        output_file = os.path.join(output_folder, 'replay' + str(i) + ".txt")
        file = io.open(output_file, "w", encoding="utf-8")
        file.write(replay_json)
        file.close()


if __name__ == '__main__':
    filename = "replay_urls/vgc19_ultra_replays.txt"
    location = "replay_jsons/vgc19_ultra_jsons"
    write_to_json(filename, location)



