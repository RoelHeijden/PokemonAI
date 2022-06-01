import os
from tqdm import tqdm
import ujson

from helpers import normalize_name, StatCalc
from PokemonML.value_network.data.categories import (
    SPECIES,
    ITEMS,
    ABILITIES,
    MOVES,
    WEATHERS,
    TERRAINS,
    TYPES,
    STATUS,
    MOVE_CATEGORIES,
    VOLATILE_STATUS,
    SIDE_CONDITIONS,
)

from PokemonML.value_network.data.lookups import (
    MOVE_LOOKUP,
    POKEMON_LOOKUP,
    FORM_LOOKUP,
    VOLATILES_TO_IGNORE,
    INVULNERABLE_STAGES,
    VULNERABLE_STAGES
)


def main():
    writer = SentWriter()

    # writer.write_poke_sents(min_rating=1200)
    # writer.write_move_sents()


class SentWriter:
    def __init__(self):
        self.data_folder = "C:/Users/RoelH/Documents/Uni/Bachelor thesis/data/raw-ou-incomplete/"
        self.out_folder = "sent_files/"

    def write_poke_sents(self, file_name='sents_per_poke.txt', min_rating=1100):
        files = sorted(
            [
                os.path.join(self.data_folder, file_name)
                for file_name in os.listdir(self.data_folder)
            ]
        )
        print(f'{len(files)} files collected')

        out_file = os.path.join(self.out_folder, file_name)

        with open(out_file, "w+") as out:
            for file in tqdm(files):
                with open(file, "r", encoding='utf-8') as f:
                    for line in f:
                        d = ujson.loads(line)

                        # get rating
                        _, p1_rating, p2_rating = self._get_player_ratings(d['inputLog'], file)

                        pokes = []
                        if p1_rating >= min_rating:
                            pokes += d["p1team"]
                        if p2_rating >= min_rating:
                            pokes += d["p2team"]

                        for poke in pokes:
                            sent = self._create_poke_sent(poke)
                            out.write(f"{sent}\n")

    def write_move_sents(self, file_name='sents_per_move.txt'):
        out_file = os.path.join(self.out_folder, file_name)
        with open(out_file, "w+") as out:
            for move in MOVES:
                sent = self._create_move_sent(move)
                out.write(f"{sent}\n")

    def _get_player_ratings(self, game_input_log, battle_id):
        """" extract average rating from a game, using the inputLog """

        def is_rated_battle(input_log):
            for string in input_log:
                if string.startswith('>start'):
                    d = ujson.loads(string.strip('>start '))
                    return d.get('rated') == 'Rated battle'
            raise KeyError("key '>start' not found in input_log of battle {}".format(battle_id))

        def get_rating(player, input_log, rated_battle):
            if not rated_battle:
                return 0

            for string in input_log:
                if string.startswith('>player ' + player):
                    string = string.strip(('>player ' + player + ' '))

                    # for some reason it's stored as: {""name": ..} but idk if that is always the case
                    if string[1] == '"' and string[2] == '"':
                        string = string[:1] + string[2:]

                    d = ujson.loads(string.strip('>player ' + player + ' '))
                    return d.get("rating")
            raise KeyError("key '>player {}' not found in input_log of battle {}".format(player, battle_id))

        is_rated_battle = is_rated_battle(game_input_log)
        p1_rating = get_rating('p1', game_input_log, is_rated_battle)
        p2_rating = get_rating('p2', game_input_log, is_rated_battle)

        return is_rated_battle, p1_rating, p2_rating

    def _create_poke_sent(self, poke):
        word_list = []

        name = normalize_name(poke["species"])
        if FORM_LOOKUP.get(name):
            name = FORM_LOOKUP[name]

        # species
        word_list.append(normalize_name(name))

        # item
        word_list.append(normalize_name(poke["item"]))

        # ability
        word_list.append(normalize_name(poke["ability"]))

        # moves
        for move in poke["moves"]:
            word_list.append(normalize_name(move))

        # typing
        for typing in POKEMON_LOOKUP[name]['types']:
            word_list.append(normalize_name(typing))

        # # stats
        # base_stats = POKEMON_LOOKUP[name]['baseStats']
        # nature = to_id_str("Serious" if poke["nature"] == "" else poke["nature"])
        # stats = StatCalc.calculate_stats(base_stats, poke['ivs'], poke['evs'], nature, poke['level'])

        return " ".join(word_list)

    def _create_move_sent(self, move):
        word_list = []
        move_data = MOVE_LOOKUP[move]

        # move name
        word_list.append(move)

        # base power
        bp = move_data['basePower']
        if bp != 0:
            if bp < 60:
                word_list.append('lowbasepower')
            if 60 <= bp < 100:
                word_list.append('midbasepower')
            if 100 <= bp:
                word_list.append('highbasepower')

        # accuracy
        acc = move_data['accuracy']
        if type(acc) == bool:
            word_list.append('noaccuracy')
        else:
            if acc == 100:
                word_list.append('100accuracy')
            if 100 > acc <= 80:
                word_list.append('midaccuracy')
            if 80 > acc:
                word_list.append('lowaccuracy')

        # type
        word_list.append(normalize_name(move_data['type']))

        # priority positive/-/negative
        priority = move_data['priority']
        if priority > 0:
            word_list.append('positivepriority')
        if priority < 0:
            word_list.append('negativepriority')

        # category: status/phys/spec
        word_list.append(normalize_name(move_data['category']))

        # target: self/opponent
        target = normalize_name(move_data['target'])
        if target == 'self':
            word_list.append('targetself')
        if target == 'all':
            word_list.append('targetall')
        else:
            word_list.append('targetopponent')

        # boosts: self/opp, positive/negative
        boosts = MOVE_LOOKUP[move].get('boosts')
        if boosts:
            for _, amount in boosts.items():
                if amount > 0:
                    word_list.append('statboost')
                if amount < 0:
                    word_list.append('statdrop')

        # flags: heal/recharge
        for flag in MOVE_LOOKUP[move]['flags']:
            if flag == 'heal':
                word_list.append('heals')
            if flag == 'recharge':
                word_list.append('mustrecharge')

        # multihit
        if MOVE_LOOKUP[move].get('multihit'):
            word_list.append('multihit')

        # future move
        if MOVE_LOOKUP[move].get('isFutureMove'):
            word_list.append('futuremove')

        # ohko
        if MOVE_LOOKUP[move].get('ohko'):
            word_list.append('ohko')

        # sets status
        if MOVE_LOOKUP[move].get('status'):
            word_list.append('setsstatus')

        return " ".join(word_list)


if __name__ == "__main__":
    main()


