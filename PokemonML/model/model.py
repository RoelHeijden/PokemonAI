import torch
import numpy as np
import pandas as pd

"""
INPUT LAYER [1 x ???]

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

        - Status conditions: [7]
        - Volatile status [27]
        - Sleep countdown [1]
        - Stat changes [7]
        - Fainted [1]
        - Health [1]
        - N moves [1]
        - Moves [4]

        - For each move:
            - Name [757]
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

"""




