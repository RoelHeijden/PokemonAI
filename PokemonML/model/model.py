"""
INPUT LAYER

- Weather type: [7] or [4*8 + 3]
- Terrain type: [4] or [4*8]
- Trick room: [1]
- Other field types: [?]  # remove?

- For each player:
    - Side conditions: [?]
    - Future sight [2]
    - Wish: [2]
    - Active: [1]
    - Reserve [5]

    - For each Pokemon
        - Species: [1032]  # reduce to gen8ou legal pokemon?
        - Abilities [263]
        - Item: [196]
        - Types [18]
        - Stats [6]

        - Status conditions: [28]
        - Volatile status [?]
        - Stat changes [6]
        - Fainted [1]
        - Max HP [1]
        - Current HP [1]
        - Last used move: [4]
        - Moves [4]

        - For each move:
            - Name: [757]  # reduce to gen8 legal moves?
            - Type: [18]
            - Move category [3]
            - Base power [1]
            - Max PP [4]

            - Current PP [4]
            - Disabled [1]
            - Disabled source [?]  # remove?

"""