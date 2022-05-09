"""
INPUT LAYER

- Weather type: [4]
- Terrain type: [4]
- Trick room: [1]
- Other field types: [???]

- For each player:
    - Side conditions: [???]

    - For each Pokemon
        - Species: [1023]  # reduce to gen8ou legal pokemon?
        - Types [18]
        - Active [1]
        - Alive [1]
        - Stats [6]
        - Stat changes [6]
        - Max HP [1]
        - Current HP [1]
        - Item: [368]  # reduce to relevant items only?
        - Moves: [731]  # reduce to gen8 legal moves?
        - Last used move: [731]
        - Max PP [4]
        - current PP [4]
        - Status [???]  # how many representations for badly poisoned, normal sleep turns and Rest sleep turns
        - Volatile status [23]
"""