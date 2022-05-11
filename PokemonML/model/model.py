"""
INPUT LAYER

- Weather type [7]
- Weather count [1]
- Terrain type [4]
- Terrain count [1]
- Trick room [1]
- Trick room count [1]

- For each player:
    - Side conditions [?]
    - Future sight [1]
    - Wish [2]
    - Active [1]
    - Reserve [5]

    - For each Pokemon
        - Species [1032]  # reduce to gen8ou legal pokemon?
        - Abilities [263]
        - Item [196]
        - Has item [1]
        - Types [18]
        - Stats [5]

        - Status conditions: [28]
        - Volatile status [?]
        - Stat changes [6]
        - Fainted [1]
        - Max HP [1]
        - Current HP [1]
        - Moves [4]

        - For each move:
            - Name [757]  # reduce to gen8 legal moves?
            - Type [18]
            - Move category [3]
            - Base power [1]
            - Max PP [4]

            - Current PP [4]
            - Disabled [1]
            - Last used [1]
            - Used [1]
            - Target self [1]

"""