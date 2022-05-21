from copy import deepcopy
import logging
import math
import json
import re

from Showdown_Pmariglia import constants
from Showdown_Pmariglia.data import all_move_json
from Showdown_Pmariglia.data import pokedex
from Showdown_Pmariglia.showdown.engine.helpers import normalize_name as normalize_name_include_nums

from game_parser.helpers import normalize_name
from PokemonML.data_processing.game_parser.state import Pokemon, LastUsedMove


logger = logging.getLogger(__name__)

MOVE_END_STRINGS = {'move', 'switch', 'upkeep', ''}


def find_pokemon_in_reserves(pkmn_name, reserves):
    for reserve_pkmn in reserves:
        if pkmn_name.startswith(reserve_pkmn.name) or reserve_pkmn.name.startswith(pkmn_name) or reserve_pkmn.base_name == pkmn_name:
            return reserve_pkmn
    return None


def is_p2(battle, split_msg):
    return not split_msg[2].startswith(battle.p1.name)


def heal_or_damage(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    pkmn = side.active

    if constants.FNT in split_msg[3]:
        pkmn.hp = 0
    else:
        pkmn.hp = float(split_msg[3].split('/')[0])

    # increase the amount of turns toxic has been active
    if len(split_msg) == 5 and constants.TOXIC in split_msg[3] and '[from] psn' in split_msg[4]:
        side.side_conditions[constants.TOXIC_COUNT] += 1

    if len(split_msg) == 5 and ('Healing Wish' in split_msg[4] or 'Lunar Dance' in split_msg[4]):
        side.active.status = None


def faint(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    side.active.hp = 0
    side.active.fainted = True


def move(battle, split_msg):
    if '[from]' in split_msg[-1] and split_msg[-1] != "[from]lockedmove":
        return
    if '[still]' in split_msg[-1] or '[miss]' in split_msg[-1]:
        if '[from]' in split_msg[-2] and split_msg[-2] != "[from]lockedmove":
            return

    move_name = normalize_name(split_msg[3].strip().lower())

    if is_p2(battle, split_msg):
        side = battle.p2
        pkmn = battle.p2.active
    else:
        side = battle.p1
        pkmn = battle.p1.active

    if move_name == 'healingwish' or move_name == 'lunardance':
        side.healing_wish_incoming = True

    # remove volatile status if they have it
    # this is for preparation moves like Phantom Force
    do_not_remove = ['substitute', 'yawn', 'disable', 'throatchop', 'uproar']
    if move_name in pkmn.volatile_statuses and move_name not in do_not_remove:
        logger.debug("Removing volatile status {} from {}".format(move_name, pkmn.name))
        pkmn.volatile_statuses.remove(move_name)

    # there's a bug in the data where in rare occasions numbers are concatenated with 'action' strings,
    # as well as with other variables. This may result in things like: shadowball31788
    if pkmn.get_move(move_name) is None and move_name != 'struggle':
        logger.warning(f'battle {battle.battle_tag} - turn {battle.turn}\n'
                       f'move {move_name} from {pkmn.name} not in pokemon move list')

        file = open('games_to_ignore.json')
        ignore_json = json.load(file)
        ignore_json[str(battle.battle_tag)] = battle.battle_tag
        open('games_to_ignore.json', 'w').write(json.dumps(ignore_json, indent=2))

    elif move_name == 'struggle':
        pass

    # decrement the PP by one
    else:
        move_object = pkmn.get_move(move_name)
        move_object.current_pp -= 1
        logger.debug("{} already has the move {}. Decrementing the PP by 1".format(pkmn.name, move_name))

    # if this pokemon used two different moves without switching,
    # set a flag to signify that it cannot have a choice item
    if (
            side.last_used_move.pokemon_name == side.active.name and
            side.last_used_move.move != move_name
    ):
        logger.debug("{} used two different moves - it cannot have a choice item".format(pkmn.name))
        pkmn.can_have_choice_item = False

    # if the p2 uses a boosting status move, they cannot have a choice item
    # this COULD be set for any status move, but some pkmn uncommonly run things like specs + wisp
    try:
        if constants.BOOSTS in all_move_json[move_name] and all_move_json[move_name][constants.CATEGORY] == constants.STATUS:
            logger.debug("{} used a boosting status-move. Setting can_have_choice_item to False".format(pkmn.name))
            pkmn.can_have_choice_item = False
    except KeyError:
        pass

    try:
        if all_move_json[move_name][constants.CATEGORY] == constants.STATUS:
            logger.debug("{} used a status-move. Setting can_have_assultvest to False".format(pkmn.name))
            pkmn.can_have_assaultvest = False
    except KeyError:
        pass

    try:
        category = all_move_json[move_name][constants.CATEGORY]
        logger.debug("Setting {}'s last used move: {}".format(pkmn.name, move_name))
        side.last_used_move = LastUsedMove(
            pokemon_name=pkmn.name,
            move=move_name,
            turn=battle.turn
        )
    except KeyError:
        category = None
        side.last_used_move = LastUsedMove(
            pokemon_name=pkmn.name,
            move=constants.DO_NOTHING_MOVE,
            turn=battle.turn
        )

    # if this pokemon used a damaging move, eliminate the possibility of it having a lifeorb
    # the lifeorb will reveal itself if it has it
    if category in constants.DAMAGING_CATEGORIES and not any([normalize_name(a) in ['sheerforce', 'magicguard'] for a in pokedex[pkmn.name][constants.ABILITIES].values()]):
        logger.debug("{} used a damaging move - not guessing lifeorb anymore".format(pkmn.name))
        pkmn.can_have_life_orb = False

    # there is nothing special in the protocol for "wish" - it must be extracted here
    if move_name == constants.WISH and 'still' not in split_msg[4]:
        logger.debug("{} used wish - expecting {} health of recovery next turn".format(side.active.name, side.active.max_hp/2))
        side.wish = (2, side.active.max_hp/2)


def boost(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    amount = int(split_msg[4].strip())

    pkmn.boosts[stat] = min(pkmn.boosts[stat] + amount, constants.MAX_BOOSTS)


def unboost(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    amount = int(split_msg[4].strip())

    pkmn.boosts[stat] = max(pkmn.boosts[stat] - amount, -1*constants.MAX_BOOSTS)


def status(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    status_name = split_msg[3].strip()
    logger.debug("{} got status: {}".format(pkmn.name, status_name))
    pkmn.status = status_name

    if status_name == 'slp':
        if split_msg[-1] == '[from] move: Rest':
            pkmn.sleep_countdown = 2
        else:
            pkmn.sleep_countdown = 3


def cant(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    if split_msg[-1] == 'slp' and pkmn.sleep_countdown > 0:
        pkmn.sleep_countdown -= 1


def activate(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
        other_pkmn = battle.p1.active
    else:
        pkmn = battle.p1.active
        other_pkmn = battle.p2.active

    if split_msg[3].lower() == 'move: skill swap':
        pkmn.ability = normalize_name(split_msg[4])
        other_pkmn.ability = normalize_name(split_msg[5])

    # check if pokemon is trapped
    if 'move: ' in split_msg[3]:
        move = normalize_name(split_msg[3].strip('move: '))
        if move in constants.BINDING_MOVES:
            pkmn.volatile_statuses.append(constants.PARTIALLY_TRAPPED)


def prepare(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    being_prepared = normalize_name_include_nums(split_msg[3])

    if (being_prepared == 'solarbeam' or being_prepared == 'solarblade') and battle.weather == constants.SUN:
        return

    if being_prepared in pkmn.volatile_statuses:
        logger.warning("{} already has the volatile status {}".format(pkmn.name, being_prepared))
    else:
        pkmn.volatile_statuses.append(being_prepared)


def start_volatile_status(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
        side = battle.p2
        opp_side = battle.p1
    else:
        pkmn = battle.p1.active
        side = battle.p1
        opp_side = battle.p2

    volatile_status = normalize_name_include_nums(split_msg[3].split(":")[-1])

    if volatile_status == 'ingrain' or volatile_status == 'noretreat' :
        pkmn.volatile_statuses.append(constants.TRAPPED)
        return

    # for some reason futuresight is sent with the `-start` message
    # `-start` is typically reserved for volatile statuses
    if volatile_status == "futuresight":
        side.future_sight = (3, pkmn.name)
        return

    # TBD
    if volatile_status == "doomdesire":
        return

    if volatile_status not in pkmn.volatile_statuses:
        logger.debug("Starting the volatile status {} on {}".format(volatile_status, pkmn.name))
        pkmn.volatile_statuses.append(volatile_status)

    if volatile_status == constants.DYNAMAX:
        pkmn.hp *= 2
        pkmn.max_hp *= 2
        logger.debug("{} started dynamax - doubling their HP to {}/{}".format(pkmn.name, pkmn.hp, pkmn.max_hp))

    if volatile_status == constants.TYPECHANGE:
        if split_msg[4] == "[from] move: Reflect Type":
            new_types = deepcopy(pokedex[opp_side.active.name][constants.TYPES])
        else:
            new_types = [normalize_name(t) for t in split_msg[4].split("/")]

        logger.debug("Setting {}'s types to {}".format(pkmn.name, new_types))
        pkmn.types = new_types


def end_volatile_status(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    volatile_status = normalize_name_include_nums(split_msg[3].split(":")[-1])

    if volatile_status == 'futuresight' or volatile_status == "doomdesire":
        return

    if volatile_status == 'octolock':
        return

    if volatile_status in constants.BINDING_MOVES:
        volatile_status = constants.PARTIALLY_TRAPPED

    if volatile_status not in pkmn.volatile_statuses:
        logger.warning("Pokemon '{}' does not have the volatile status '{}'\n"
                       "battle: {}, turn: {}".format(pkmn.to_dict(), volatile_status, battle.battle_tag, battle.turn))
    else:
        logger.debug("Removing the volatile status {} from {}".format(volatile_status, pkmn.name))
        pkmn.volatile_statuses.remove(volatile_status)
        if volatile_status == constants.DYNAMAX:
            pkmn.hp /= 2
            pkmn.max_hp /= 2
            logger.debug("{} ended dynamax - halving their HP to {}/{}".format(pkmn.name, pkmn.hp, pkmn.max_hp))


def curestatus(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    side.active.status = None
    side.side_conditions[constants.TOXIC_COUNT] = 0
    side.active.sleep_countdown = 0


def cureteam(battle, split_msg):
    """Cure every pokemon on the p2's team of it's status"""
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    side.active.status = None
    side.side_conditions[constants.TOXIC_COUNT] = 0
    side.active.sleep_countdown = 0

    for pkmn in filter(lambda p: isinstance(p, Pokemon), side.reserve):
        pkmn.status = None
        pkmn.sleep_countdown = 0


def weather(battle, split_msg):
    weather_name = normalize_name(split_msg[2].split(':')[-1].strip())
    battle.weather = weather_name

    if split_msg[-1] == '[upkeep]':
        battle.weather_count += 1
    else:
        battle.weather_count = 0


def fieldstart(battle, split_msg):
    """Set the battle's field condition"""
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())

    if field_name == constants.TRICK_ROOM:
        battle.trick_room = True
        battle.trick_room_count = 0

    elif field_name == 'gravity':
        battle.gravity = True
        battle.gravity_count = 0

    elif field_name == 'magicroom':
        battle.magic_room = True
        battle.magic_room_count = 0

    elif field_name == 'wonderroom':
        battle.wonder_room = True
        battle.wonder_room_count = 0

    elif field_name in constants.TERRAIN:
        battle.terrain = field_name
        battle.terrain_count = 0

    else:
        logger.warning(f'field {field_name} not recognized, battle id {battle.battle_tag}')


def fieldend(battle, split_msg):
    """Remove the battle's field condition"""
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())

    # trick room shows up as a `-fieldend` item but is separate from the other fields
    if field_name == constants.TRICK_ROOM:
        battle.trick_room = False
        battle.trick_room_count = 0

    elif field_name == 'gravity':
        battle.gravity = False
        battle.gravity_count = 0

    elif field_name == 'magicroom':
        battle.magic_room = False
        battle.magic_room_count = 0

    elif field_name == 'wonderroom':
        battle.wonder_room = False
        battle.wonder_room_count = 0

    elif field_name in constants.TERRAIN:
        battle.terrain = None
        battle.terrain_count = 0

    else:
        logger.warning(f'field {field_name} not recognized, battle id {battle.battle_tag}')


def sidestart(battle, split_msg):
    """Set a side effect such as stealth rock or sticky web"""
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)

    if is_p2(battle, split_msg):
        logger.debug("Side condition {} starting for p2".format(condition))
        battle.p2.side_conditions[condition] += 1
    else:
        logger.debug("Side condition {} starting for bot".format(condition))
        battle.p1.side_conditions[condition] += 1


def sideend(battle, split_msg):
    """Remove a side effect such as stealth rock or sticky web"""
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)

    if is_p2(battle, split_msg):
        logger.debug("Side condition {} ending for p2".format(condition))
        battle.p2.side_conditions[condition] = 0
    else:
        logger.debug("Side condition {} ending for bot".format(condition))
        battle.p1.side_conditions[condition] = 0


def swapsideconditions(battle, _):
    p1_sc = battle.p1.side_conditions
    opponent_sc = battle.p2.side_conditions
    for side_condition in constants.COURT_CHANGE_SWAPS:
        p1_sc[side_condition], opponent_sc[side_condition] = opponent_sc[side_condition], p1_sc[side_condition]


def set_item(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    # if p2 is being given a choice scarf (via Trick or Switcheroo)
    item = normalize_name(split_msg[3].strip())
    logger.debug("Setting {}'s item to {}".format(side.active.name, item))
    side.active.item = item


def remove_item(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    logger.debug("Removing {}'s item".format(side.active.name))
    side.active.item = None


def set_ability(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    for msg in split_msg:
        if constants.ABILITY in normalize_name(msg):
            ability = normalize_name(msg.split(':')[-1])
            logger.debug("Setting {}'s ability to {}".format(side.active.name, ability))
            side.active.ability = ability


def clearnegativeboost(battle, split_msg):
    if is_p2(battle, split_msg):
        pkmn = battle.p2.active
    else:
        pkmn = battle.p1.active

    for stat, value in pkmn.boosts.items():
        if value < 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0


def clearallboost(battle, _):
    pkmn = battle.p1.active
    for stat, value in pkmn.boosts.items():
        if value != 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0

    pkmn = battle.p2.active
    for stat, value in pkmn.boosts.items():
        if value != 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0


def singleturn(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    move_name = normalize_name_include_nums(split_msg[3].split(':')[-1])
    if move_name in constants.PROTECT_VOLATILE_STATUSES:
        # set to 2 because the `upkeep` function will decrement by 1 on every end-of-turn
        side.side_conditions[constants.PROTECT] = 2
        logger.debug("{} used protect".format(side.active.name))


def upkeep(battle, _):
    if battle.p1.side_conditions[constants.PROTECT] > 0:
        battle.p1.side_conditions[constants.PROTECT] -= 1
        logger.debug("Setting protect to {} for the bot".format(battle.p1.side_conditions[constants.PROTECT]))

    if battle.p2.side_conditions[constants.PROTECT] > 0:
        battle.p2.side_conditions[constants.PROTECT] -= 1
        logger.debug("Setting protect to {} for the p2".format(battle.p2.side_conditions[constants.PROTECT]))

    if battle.p1.wish[0] > 0:
        battle.p1.wish = (battle.p1.wish[0] - 1, battle.p1.wish[1])
        logger.debug("Decrementing wish to {} for the bot".format(battle.p1.wish[0]))

    if battle.p2.wish[0] > 0:
        battle.p2.wish = (battle.p2.wish[0] - 1, battle.p2.wish[1])
        logger.debug("Decrementing wish to {} for the p2".format(battle.p2.wish[0]))

    if battle.p1.future_sight[0] > 0:
        battle.p1.future_sight = (battle.p1.future_sight[0] - 1, battle.p1.future_sight[1])
        logger.debug("Decrementing future_sight to {} for the bot".format(battle.p1.future_sight[0]))

    if battle.p2.future_sight[0] > 0:
        battle.p2.future_sight = (battle.p2.future_sight[0] - 1, battle.p2.future_sight[1])
        logger.debug("Decrementing future_sight to {} for the p2".format(battle.p2.future_sight[0]))

    if battle.p1.active.status != constants.TOXIC:
        battle.p1.side_conditions[constants.TOXIC_COUNT] = 0

    if battle.p2.active.status != constants.TOXIC:
        battle.p2.side_conditions[constants.TOXIC_COUNT] = 0

    if battle.terrain:
        battle.terrain_count += 1

    if battle.trick_room:
        battle.trick_room_count += 1

    if battle.magic_room:
        battle.magic_room_count += 1

    if battle.gravity:
        battle.gravity += 1

    if battle.wonder_room:
        battle.wonder_room += 1



def turn(battle, split_msg):
    battle.turn = int(split_msg[2])


def switch_or_drag(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    if side.healing_wish_incoming:
        side.healing_wish_incoming = False

    if side.active is not None:
        # set the pkmn's types back to their original value if the types were changed
        if constants.TYPECHANGE in side.active.volatile_statuses:
            original_types = pokedex[side.active.name][constants.TYPES]
            logger.debug("{} had it's type changed - changing its types back to {}".format(side.active.name, original_types))
            side.active.types = original_types

        # if the target was transformed, reset its transformed attributes
        if constants.TRANSFORM in side.active.volatile_statuses:
            logger.debug("{} was transformed. Resetting its transformed attributes".format(side.active.name))
            side.active.stats = side.active.original_attributes['stats']
            side.active.moves = side.active.original_attributes['moves']
            side.active.types = pokedex[side.active.name][constants.TYPES]

        # reset the ability to original
        side.active.ability = side.active.original_attributes['ability']

        # reset the boost of the pokemon being replaced
        side.active.boosts = {key: 0 for key in side.active.boosts}

        # reset the volatile statuses of the pokemon being replaced
        side.active.volatile_statuses.clear()

        # reset toxic count for this side
        side.side_conditions[constants.TOXIC_COUNT] = 0

        # if the side is alive and has regenerator, give it back 1/3 of it's maxhp
        if side.active.hp > 0 and not side.active.fainted and side.active.ability == "regenerator":
            health_healed = int(side.active.max_hp / 3)
            side.active.hp = min(side.active.hp + health_healed, side.active.max_hp)
            logger.debug(
                "{} switched out with regenerator. Healing it to {}/{}".format(
                    side.active.name, side.active.hp, side.active.max_hp
                )
            )

    name = normalize_name(split_msg[3].split(',')[0])
    pkmn = find_pokemon_in_reserves(name, side.reserve)

    side.reserve.remove(pkmn)

    side.last_used_move = LastUsedMove(
        pokemon_name=None,
        move='switch {}'.format(pkmn.name),
        turn=battle.turn
    )

    # pkmn != active is a special edge-case for Zoroark
    if side.active is not None and pkmn != side.active:
        side.reserve.append(side.active)

    side.active = pkmn


def transform(battle, split_msg):
    if is_p2(battle, split_msg):
        opp_pkmn = battle.p1.active
        user_pkmn = battle.p2.active
    else:
        opp_pkmn = battle.p2.active
        user_pkmn = battle.p1.active

    if opp_pkmn is None:
        raise ValueError(f"Pokemon {user_pkmn} cannot transform into '{opp_pkmn}'")

    user_pkmn.boosts = deepcopy(opp_pkmn.boosts)
    user_pkmn.stats = deepcopy(opp_pkmn.stats)
    user_pkmn.ability = deepcopy(opp_pkmn.ability)
    user_pkmn.moves = deepcopy(opp_pkmn.moves)
    user_pkmn.types = deepcopy(opp_pkmn.types)

    for m in user_pkmn.moves:
        m.max_pp = 5
        m.current_pp = 5

    if constants.TRANSFORM not in user_pkmn.volatile_statuses:
        user_pkmn.volatile_statuses.append(constants.TRANSFORM)


def form_change(battle, split_msg):
    if is_p2(battle, split_msg):
        side = battle.p2
    else:
        side = battle.p1

    previous_boosts = side.active.boosts
    prev_base_name = side.active.base_name
    prev_hp = side.active.hp
    prev_status = side.active.status
    prev_volatile_status = side.active.volatile_statuses
    prev_moves = side.active.moves

    name = split_msg[3].split(',')[0]

    if normalize_name(name) == "zoroark":
        prev_active = deepcopy(side.active)
        side.active = find_pokemon_in_reserves(normalize_name(name), side.reserve)

        side.active.hp = math.ceil((prev_active.hp / prev_active.max_hp) * side.active.max_hp)

    else:
        new_pokemon = Pokemon(
            name=name,
            nickname=side.active.nickname,
            level=side.active.level,
            gender=side.active.gender,
            nature=side.active.nature,
            evs=side.active.evs,
            ivs=side.active.ivs,
            ability=side.active.ability,
            moves=[],
            item=side.active.item
        )
        side.active = new_pokemon
        side.active.moves = prev_moves
        side.active.hp = prev_hp
        side.active.base_name = prev_base_name

    side.active.boosts = previous_boosts
    side.active.status = prev_status
    side.active.volatile_statuses = prev_volatile_status


def update_state(battle, split_msg):
    action = split_msg[1].strip()

    # some games are bugged where action strings are followed by the players anonymized name.
    # e.g. move79888
    if any(i.isdigit() for i in action) and not action.startswith('-side'):
        file = open('games_to_ignore.json')
        ignore_json = json.load(file)
        ignore_json[str(battle.battle_tag)] = battle.battle_tag
        open('games_to_ignore.json', 'w').write(json.dumps(ignore_json, indent=2))

        logger.warning(f"game {battle.battle_tag} contained action '{action}'\n"
                       f"game aborted and added to games_to_ignore")
        return False

    # old games have something like: -sidestart38555, message.
    # split this into: -sidestart, p1/p2, message.
    if action.startswith('-side') and any(i.isdigit() for i in action):
        action, player_name = re.split('(\d+)', action)[0:2]

        if battle.p1.account_name == player_name:
            player = 'p1'
        elif battle.p2.account_name == player_name:
            player = 'p2'
        else:
            raise ValueError(f"none of the players match player name '{player_name}'\n"
                             f"battle: {battle.battle_tag}, turn: {battle.turn}")

        split_msg = [split_msg[0], action, player + ': ' + str(player_name)] + split_msg[2:]

    battle_modifiers_lookup = {
        'switch': switch_or_drag,
        'faint': faint,
        'drag': switch_or_drag,
        '-heal': heal_or_damage,
        '-damage': heal_or_damage,
        'move': move,
        '-boost': boost,
        '-unboost': unboost,
        '-status': status,
        'cant': cant,
        '-activate': activate,
        '-prepare': prepare,
        '-start': start_volatile_status,
        '-end': end_volatile_status,
        '-curestatus': curestatus,
        '-cureteam': cureteam,
        '-weather': weather,
        '-fieldstart': fieldstart,
        '-fieldend': fieldend,
        '-sidestart': sidestart,
        '-sideend': sideend,
        '-swapsideconditions': swapsideconditions,
        '-item': set_item,
        '-enditem': remove_item,
        '-immune': set_ability,
        'detailschange': form_change,
        'replace': form_change,
        '-formechange': form_change,
        '-transform': transform,
        '-clearnegativeboost': clearnegativeboost,
        '-clearallboost': clearallboost,
        '-singleturn': singleturn,
        'upkeep': upkeep,
        'turn': turn,
    }

    function_to_call = battle_modifiers_lookup.get(action)
    if function_to_call is not None:
        function_to_call(battle, split_msg)

    if action == 'turn' or action == 'upkeep':
        battle.p2.check_if_trapped(battle)
        battle.p2.lock_moves()
        battle.p1.check_if_trapped(battle)
        battle.p1.lock_moves()

    return True

