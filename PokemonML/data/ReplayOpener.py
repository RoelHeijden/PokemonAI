from urllib.request import Request, urlopen
import io


def write_to_json(input_file, output_file):
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
    file = io.open(output_file, "w", encoding="utf-8")
    file.writelines(json_list)
    file.close()


if __name__ == '__main__':
    filename = "replay_urls/all_replays.txt"
    location = "replay_jsons/all_jsons.txt"
    write_to_json(filename, location)



