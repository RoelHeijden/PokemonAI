from tqdm import tqdm

from data_loader import data_loader

"""
INPUT LAYER

- Weather type [8]
- Weather count [1]
- Terrain type [5]
- Terrain count [1]
- Trick room [1]
- Trick room count [1]

- For each player:
    - Side conditions [11]
    - Future sight [1]
    - Wish [2]
    - Healing wish [1]
    - Has active [1]
    - N pokemon [1]
    - Active [1]
    - Reserve [5]

    - For each Pokemon
        - Species [1033] -> embedding: [64]
        - Abilities [263] -> embedding: [16]
        - Item [196] -> embedding: [16]
        - Has item [1]
        - Types [18]
        - Stats [6]
        - Level [1]

        - Status conditions: [7]
        - Volatile status [27]
        - Sleep countdown [1]
        - Stat changes [7]
        - Is alive [1]
        - Health [1]
        - N moves [1]
        - Moves [4]

        - For each move:
            - Name [757] -> embedding: [64]
            - Type [18]
            - Move category [3]
            - Base power [1]
            - Max PP [4]

            - Current PP [4]
            - Disabled [1]
            - Used [1]
            - Targets self [1]
            - Priority [1]


Attributes that may be unavailable in pmariglia's state simulation:
    - weather, terrain and trick room count
    - sleep countdown
    - healing wish
    - choicelock in volatile status



---------------------- TO DO ----------------------

0. check if normalize name is fine, and if gravity, terrain, etc work

1. create training to test split
2. create new jsonl batches

3. implement gravity, magic room and wonder room in transformer

4. see how embedding works with 2dim arrays [2*6 pokemon] when splitting afterwards
5. finish transformer with tensors for:
    - result [1x1]
    - field state [1xn]
    - player state conditions [2xn]
    - pokemon names [2x6]
    - pokemon abilities [2x6]
    - pokemon items [2x6]
    - pokemon moves [2x6x4]
    - pokemon attributes [2x6xn]
    

6. scale data (e.g. stats = stats / 250)
7. shuffle moves and reserve pokemon (as final step)
8. split game results from dataset


"""


def main():
    training_files = 'C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/processed-ou-incomplete/all_rated_1200+/training_states_testfolder'
    loader = data_loader(training_files, batch_size=100)

    for batch in tqdm(loader):
        # print(batch['fields'])
        pass


if __name__ == '__main__':
    main()

