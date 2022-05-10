"""
INPUT LAYER

- Weather type: [7] or [4*8 + 3]
- Terrain type: [4] or [4*8]
- Trick room: [1]
- Other field types: [?]

- For each player:
    - Side conditions: [?]

    - For each Pokemon
        - Species: [1032]  # reduce to gen8ou legal pokemon?
        - Moves: [757]  # reduce to gen8 legal moves?
        - Last used move: [757]
        - abilities [263]
        - Item: [196]
        - Status conditions: [28]
        - Volatile status [?]
        - Types [18]
        - Max PP [4]
        - current PP [4]
        - Stats [6]
        - Stat changes [6]
        - Active [1]
        - Alive [1]
        - Max HP [1]
        - Current HP [1]

"""