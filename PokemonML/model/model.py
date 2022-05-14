import torch
import numpy as np
import pandas as pd

"""
INPUT LAYER

- Weather type [8]
- Weather count [1]
- Terrain type [5]
- Terrain count [1]
- Trick room [1]
- Trick room count [1]

- For each player:
    - Side conditions [?]
    - Future sight [1]
    - Wish [2]
    - Has active [1]
    - Active [1]
    - Reserve [5]

    - For each Pokemon
        - Species [1033]
        - Abilities [263]
        - Item [196]
        - Has item [1]
        - Types [18]
        - Stats [6]
        - Level [1]

        - Status conditions: [28]
        - Volatile status [?]
        - Stat changes [7]
        - Fainted [1]
        - Health [1]
        - First turn out [1]
        - Moves [4]

        - For each move:
            - Name [757]
            - Type [18]
            - Move category [3]
            - Base power [1]
            - Max PP [4]

            - Current PP [4]
            - Disabled [1]
            - Last used [1]
            - Used [1]
            - Targets self [1]
            - Priority [1]


Attributes that may be unavailable in pmariglia's state simulation:
    - weather, terrain and trick room count
    - Pokemon: First turn out
    - Move: Last used


To add to state:
    Game
    - Turn
    - P1 rating
    - P2 rating
    - Average rating
    - Rated battle
    - Room ID

    Field
    - Trick room [boolean]
    - Weather turn count
    - Terrain turn count
    - Trick room turn count

    Pokemon
    - First_turn_out [boolean]
    - Toxic turn count (if not already in status representation)
    - Sleep turn count (if not already in status or volatile_status representation)

    Move
    - Last_used [boolean]

"""

if torch.cuda.is_available():
    print("hey")
