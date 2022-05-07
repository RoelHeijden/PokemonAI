from copy import deepcopy
import logging
import math

from Showdown_Pmariglia import constants
from Showdown_Pmariglia.data import all_move_json
from Showdown_Pmariglia.data import pokedex
from battle import Pokemon, LastUsedMove
from Showdown_Pmariglia.showdown.engine.helpers import normalize_name
from Showdown_Pmariglia.showdown.engine.helpers import calculate_stats


logger = logging.getLogger(__name__)


MOVE_END_STRINGS = {'move', 'switch', 'upkeep', ''}


def find_pokemon_in_reserves(pkmn_name, reserves):
    for reserve_pkmn in reserves:
        if pkmn_name.startswith(reserve_pkmn.name) or reserve_pkmn.name.startswith(pkmn_name) or reserve_pkmn.base_name == pkmn_name:
            return reserve_pkmn
    return None


def is_opponent(battle,  split_msg):
    return not split_msg[2].startswith(battle.user.name)


def switch_or_drag(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    if side.active is not None:
        # set the pkmn's types back to their original value if the types were changed
        if constants.TYPECHANGE in side.active.volatile_statuses:
            original_types = pokedex[side.active.name][constants.TYPES]
            logger.debug("{} had it's type changed - changing its types back to {}".format(side.active.name, original_types))
            side.active.types = original_types

        # if the target was transformed, reset its transformed attributes
        if constants.TRANSFORM in side.active.volatile_statuses:
            logger.debug("{} was transformed. Resetting its transformed attributes".format(side.active.name))
            side.active.stats = calculate_stats(side.active.base_stats, side.active.level)
            side.active.ability = None
            side.active.moves = []
            side.active.types = pokedex[side.active.name][constants.TYPES]

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


def heal_or_damage(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    pkmn = side.active

    if constants.FNT in split_msg[3]:
        pkmn.hp = 0
    else:
        pkmn.hp = float(split_msg[3].split('/')[0])

    # increase the amount of turns toxic has been active
    if len(split_msg) == 5 and constants.TOXIC in split_msg[3] and '[from] psn' in split_msg[4]:
        side.side_conditions[constants.TOXIC_COUNT] += 1


def faint(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    side.active.hp = 0
    side.active.fainted = True


def move(battle, split_msg):
    if '[from]' in split_msg[-1] and split_msg[-1] != "[from]lockedmove":
        return

    move_name = normalize_name(split_msg[3].strip().lower())

    if is_opponent(battle, split_msg):
        side = battle.opponent
        pkmn = battle.opponent.active
    else:
        side = battle.user
        pkmn = battle.user.active

    # remove volatile status if they have it
    # this is for preparation moves like Phantom Force
    if move_name in pkmn.volatile_statuses and not move_name == 'substitute':
        logger.debug("Removing volatile status {} from {}".format(move_name, pkmn.name))
        pkmn.volatile_statuses.remove(move_name)

    # add the move to it's moves if it hasn't been seen
    # decrement the PP by one
    # if the move is unknown, do nothing
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

    # if the opponent uses a boosting status move, they cannot have a choice item
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
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    amount = int(split_msg[4].strip())

    pkmn.boosts[stat] = min(pkmn.boosts[stat] + amount, constants.MAX_BOOSTS)


def unboost(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    stat = constants.STAT_ABBREVIATION_LOOKUPS[split_msg[3].strip()]
    amount = int(split_msg[4].strip())

    pkmn.boosts[stat] = max(pkmn.boosts[stat] - amount, -1*constants.MAX_BOOSTS)


def status(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    status_name = split_msg[3].strip()
    logger.debug("{} got status: {}".format(pkmn.name, status_name))
    pkmn.status = status_name


def activate(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
        other_pkmn = battle.user.active
    else:
        pkmn = battle.user.active
        other_pkmn = battle.opponent.active

    if split_msg[3].lower() == 'move: skill swap':
        pkmn.ability = normalize_name(split_msg[4])
        other_pkmn.ability = normalize_name(split_msg[5])

    # check if pokemon is trapped
    if 'move: ' in split_msg[3]:
        move = normalize_name(split_msg[3].strip('move: '))
        if move in constants.BINDING_MOVES:
            pkmn.volatile_statuses.append(constants.PARTIALLY_TRAPPED)


def prepare(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    being_prepared = normalize_name(split_msg[3])
    if being_prepared in pkmn.volatile_statuses:
        logger.warning("{} already has the volatile status {}".format(pkmn.name, being_prepared))
    else:
        pkmn.volatile_statuses.append(being_prepared)


def start_volatile_status(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
        side = battle.opponent
    else:
        pkmn = battle.user.active
        side = battle.user

    volatile_status = normalize_name(split_msg[3].split(":")[-1])

    if volatile_status == 'ingrain' or volatile_status == 'noretreat' :
        pkmn.volatile_statuses.append(constants.TRAPPED)
        return

    # for some reason futuresight is sent with the `-start` message
    # `-start` is typically reserved for volatile statuses
    if volatile_status == "futuresight":
        side.future_sight = (3, pkmn.name)
        return

    if volatile_status not in pkmn.volatile_statuses:
        logger.debug("Starting the volatile status {} on {}".format(volatile_status, pkmn.name))
        pkmn.volatile_statuses.append(volatile_status)

    if volatile_status == constants.DYNAMAX:
        pkmn.hp *= 2
        pkmn.max_hp *= 2
        logger.debug("{} started dynamax - doubling their HP to {}/{}".format(pkmn.name, pkmn.hp, pkmn.max_hp))

    if constants.ABILITY in split_msg[3]:
        pkmn.ability = volatile_status

    if len(split_msg) == 6 and constants.ABILITY in normalize_name(split_msg[5]):
        pkmn.ability = normalize_name(split_msg[5].split('ability:')[-1])

    if volatile_status == constants.TYPECHANGE:
        if split_msg[4] == "[from] move: Reflect Type":
            pkmn_name = normalize_name(split_msg[5].split(":")[-1])
            new_types = deepcopy(pokedex[pkmn_name][constants.TYPES])
        else:
            new_types = [normalize_name(t) for t in split_msg[4].split("/")]

        logger.debug("Setting {}'s types to {}".format(pkmn.name, new_types))
        pkmn.types = new_types


def end_volatile_status(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    volatile_status = normalize_name(split_msg[3].split(":")[-1])
    if volatile_status in constants.BINDING_MOVES:
        volatile_status = constants.PARTIALLY_TRAPPED

    if volatile_status not in pkmn.volatile_statuses:
        logger.warning("Pokemon '{}' does not have the volatile status '{}'".format(pkmn.to_dict(), volatile_status))
    else:
        logger.debug("Removing the volatile status {} from {}".format(volatile_status, pkmn.name))
        pkmn.volatile_statuses.remove(volatile_status)
        if volatile_status == constants.DYNAMAX:
            pkmn.hp /= 2
            pkmn.max_hp /= 2
            logger.debug("{} ended dynamax - halving their HP to {}/{}".format(pkmn.name, pkmn.hp, pkmn.max_hp))


def curestatus(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    pkmn_name = split_msg[2].split(':')[-1].strip()

    if normalize_name(pkmn_name) == side.active.name:
        pkmn = side.active
    else:
        try:
            pkmn = next(filter(lambda x: x.name == normalize_name(pkmn_name), side.reserve))
        except StopIteration:
            logger.warning(
                "The pokemon {} does not exist in the party, defaulting to the active pokemon".format(normalize_name(pkmn_name))
            )
            pkmn = side.active

    pkmn.status = None


def cureteam(battle, split_msg):
    """Cure every pokemon on the opponent's team of it's status"""
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    side.active.status = None
    for pkmn in filter(lambda p: isinstance(p, Pokemon), side.reserve):
        pkmn.status = None


def weather(battle, split_msg):
    weather_name = normalize_name(split_msg[2].split(':')[-1].strip())
    logger.debug("Weather {} started".format(weather_name))
    battle.weather = weather_name


def fieldstart(battle, split_msg):
    """Set the battle's field condition"""
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())

    # trick room shows up as a `-fieldstart` item but is separate from the other fields
    if field_name == constants.TRICK_ROOM:
        logger.debug("Setting trickroom")
        battle.trick_room = True
    else:
        logger.debug("Setting the field to {}".format(field_name))
        battle.field = field_name


def fieldend(battle, split_msg):
    """Remove the battle's field condition"""
    field_name = normalize_name(split_msg[2].split(':')[-1].strip())

    # trick room shows up as a `-fieldend` item but is separate from the other fields
    if field_name == constants.TRICK_ROOM:
        logger.debug("Removing trick room")
        battle.trick_room = False
    else:
        logger.debug("Setting the field to None")
        battle.field = None


def sidestart(battle, split_msg):
    """Set a side effect such as stealth rock or sticky web"""
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)

    if is_opponent(battle, split_msg):
        logger.debug("Side condition {} starting for opponent".format(condition))
        battle.opponent.side_conditions[condition] += 1
    else:
        logger.debug("Side condition {} starting for bot".format(condition))
        battle.user.side_conditions[condition] += 1


def sideend(battle, split_msg):
    """Remove a side effect such as stealth rock or sticky web"""
    condition = split_msg[3].split(':')[-1].strip()
    condition = normalize_name(condition)

    if is_opponent(battle, split_msg):
        logger.debug("Side condition {} ending for opponent".format(condition))
        battle.opponent.side_conditions[condition] = 0
    else:
        logger.debug("Side condition {} ending for bot".format(condition))
        battle.user.side_conditions[condition] = 0


def swapsideconditions(battle, _):
    user_sc = battle.user.side_conditions
    opponent_sc = battle.opponent.side_conditions
    for side_condition in constants.COURT_CHANGE_SWAPS:
        user_sc[side_condition], opponent_sc[side_condition] = opponent_sc[side_condition], user_sc[side_condition]


def set_item(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    # if opponent is being given a choice scarf (via Trick or Switcheroo)
    item = normalize_name(split_msg[3].strip())
    logger.debug("Setting {}'s item to {}".format(side.active.name, item))
    side.active.item = item


def remove_item(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    logger.debug("Removing {}'s item".format(side.active.name))
    side.active.item = None


def set_ability(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    for msg in split_msg:
        if constants.ABILITY in normalize_name(msg):
            ability = normalize_name(msg.split(':')[-1])
            logger.debug("Setting {}'s ability to {}".format(side.active.name, ability))
            side.active.ability = ability


def form_change(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    prev_pkmn_state = side.active.to_dict()
    previous_boosts = side.active.boosts
    name = split_msg[3].split(',')[0]

    if normalize_name(name) == "zoroark":
        prev_active = deepcopy(side.active)
        side.active = find_pokemon_in_reserves(normalize_name(name), side.reserve)

        side.active.hp = math.ceil((prev_active.hp / prev_active.max_hp) * side.active.max_hp)

    else:
        new_pokemon = Pokemon(
            name=name,
            level=prev_pkmn_state['level'],
            gender=prev_pkmn_state['gender'],
            nature=prev_pkmn_state['nature'],
            evs=prev_pkmn_state['evs'],
            ivs=prev_pkmn_state['ivs'],
            ability=prev_pkmn_state['ability'],
            moves=prev_pkmn_state['moves'],
            item=prev_pkmn_state['item']
        )
        side.active = new_pokemon
        side.active.hp = prev_pkmn_state['hp']
        side.active.base_name = prev_pkmn_state['base_name']

    side.active.boosts = previous_boosts
    side.active.status = prev_pkmn_state['status']
    side.active.volatile_statuses = prev_pkmn_state['volatile_status']


def clearnegativeboost(battle, split_msg):
    if is_opponent(battle, split_msg):
        pkmn = battle.opponent.active
    else:
        pkmn = battle.user.active

    for stat, value in pkmn.boosts.items():
        if value < 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0


def clearallboost(battle, _):
    pkmn = battle.user.active
    for stat, value in pkmn.boosts.items():
        if value != 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0

    pkmn = battle.opponent.active
    for stat, value in pkmn.boosts.items():
        if value != 0:
            logger.debug("Setting {}'s {} stat to 0".format(pkmn.name, stat))
            pkmn.boosts[stat] = 0


def singleturn(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    move_name = normalize_name(split_msg[3].split(':')[-1])
    if move_name in constants.PROTECT_VOLATILE_STATUSES:
        # set to 2 because the `upkeep` function will decrement by 1 on every end-of-turn
        side.side_conditions[constants.PROTECT] = 2
        logger.debug("{} used protect".format(side.active.name))


def upkeep(battle, _):
    if battle.user.side_conditions[constants.PROTECT] > 0:
        battle.user.side_conditions[constants.PROTECT] -= 1
        logger.debug("Setting protect to {} for the bot".format(battle.user.side_conditions[constants.PROTECT]))

    if battle.opponent.side_conditions[constants.PROTECT] > 0:
        battle.opponent.side_conditions[constants.PROTECT] -= 1
        logger.debug("Setting protect to {} for the opponent".format(battle.opponent.side_conditions[constants.PROTECT]))

    if battle.user.wish[0] > 0:
        battle.user.wish = (battle.user.wish[0] - 1, battle.user.wish[1])
        logger.debug("Decrementing wish to {} for the bot".format(battle.user.wish[0]))

    if battle.opponent.wish[0] > 0:
        battle.opponent.wish = (battle.opponent.wish[0] - 1, battle.opponent.wish[1])
        logger.debug("Decrementing wish to {} for the opponent".format(battle.opponent.wish[0]))

    if battle.user.future_sight[0] > 0:
        battle.user.future_sight = (battle.user.future_sight[0] - 1, battle.user.future_sight[1])
        logger.debug("Decrementing future_sight to {} for the bot".format(battle.user.future_sight[0]))

    if battle.opponent.future_sight[0] > 0:
        battle.opponent.future_sight = (battle.opponent.future_sight[0] - 1, battle.opponent.future_sight[1])
        logger.debug("Decrementing future_sight to {} for the opponent".format(battle.opponent.future_sight[0]))


def mega(battle, split_msg):
    if is_opponent(battle, split_msg):
        side = battle.opponent
    else:
        side = battle.user

    side.active.is_mega = True
    logger.debug("Mega-Pokemon: {}".format(side.active.name))


def zpower(battle, split_msg):
    pass


def transform(battle, split_msg):
    if is_opponent(battle, split_msg):
        transformed_into_name = battle.user.active.name

        battle_copy = deepcopy(battle)
        battle.opponent.active.boosts = deepcopy(battle.user.active.boosts)

        battle_copy.user.from_json(battle_copy.request_json)

        if battle_copy.user.active.name == transformed_into_name or battle_copy.user.active.name.startswith(transformed_into_name):
            transformed_into = battle_copy.user.active
        else:
            transformed_into = find_pokemon_in_reserves(transformed_into_name, battle_copy.user.reserve)

        logger.debug("Opponent {} transformed into {}".format(battle.opponent.active.name, battle.user.active.name))
        battle.opponent.active.stats = deepcopy(transformed_into.stats)
        battle.opponent.active.ability = deepcopy(transformed_into.ability)
        battle.opponent.active.moves = deepcopy(transformed_into.moves)
        battle.opponent.active.types = deepcopy(transformed_into.types)

        if constants.TRANSFORM not in battle.opponent.active.volatile_statuses:
            battle.opponent.active.volatile_statuses.append(constants.TRANSFORM)


def turn(battle, split_msg):
    battle.turn = int(split_msg[2])


def update_battle(battle, split_msg):
    action = split_msg[1].strip()

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
        '-mega': mega,
        '-zpower': zpower,
        '-clearnegativeboost': clearnegativeboost,
        '-clearallboost': clearallboost,
        '-singleturn': singleturn,
        'upkeep': upkeep,
        'turn': turn,
    }

    function_to_call = battle_modifiers_lookup.get(action)
    if function_to_call is not None:
        function_to_call(battle, split_msg)

    if action == 'turn':
        battle.opponent.check_if_trapped(battle)
        battle.opponent.lock_moves()
        battle.user.check_if_trapped(battle)
        battle.opponent.lock_moves()

