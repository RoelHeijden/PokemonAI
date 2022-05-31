import re
import math


natures = {
    'lonely': {
        'plus': 'atk',
        'minus': 'def'
    },
    'adamant': {
        'plus': 'atk',
        'minus': 'spa'
    },
    'naughty': {
        'plus': 'atk',
        'minus': 'spd'
    },
    'brave': {
        'plus': 'atk',
        'minus': 'spe'
    },
    'bold': {
        'plus': 'def',
        'minus': 'atk'
    },
    'impish': {
        'plus': 'def',
        'minus': 'spa'
    },
    'lax': {
        'plus': 'def',
        'minus': 'spd'
    },
    'relaxed': {
        'plus': 'def',
        'minus': 'spe'
    },
    'modest': {
        'plus': 'spa',
        'minus': 'atk'
    },
    'mild': {
        'plus': 'spa',
        'minus': 'def'
    },
    'rash': {
        'plus': 'spa',
        'minus': 'spd'
    },
    'quiet': {
        'plus': 'spa',
        'minus': 'spe'
    },
    'calm': {
        'plus': 'spd',
        'minus': 'atk'
    },
    'gentle': {
        'plus': 'spd',
        'minus': 'def'
    },
    'careful': {
        'plus': 'spd',
        'minus': 'spa'
    },
    'sassy': {
        'plus': 'spd',
        'minus': 'spe'
    },
    'timid': {
        'plus': 'spe',
        'minus': 'atk'
    },
    'hasty': {
        'plus': 'spe',
        'minus': 'def'
    },
    'jolly': {
        'plus': 'spe',
        'minus': 'spa'
    },
    'naive': {
        'plus': 'spe',
        'minus': 'spd'
    },
}


# def normalize_name(name: str) -> str:
#     return "".join(char for char in name if char.isalnum()).lower()

def normalize_name(name):
    return "".join(re.findall("[a-zA-Z0-9]+", name)).replace(" ", "").lower()


class StatCalc:
    def __init__(self):
        pass

    @staticmethod
    def _stat_calc(base_stat: int, iv: int, ev: int, level: int):
        return math.floor(((2 * base_stat + iv + math.floor(ev / 4)) * level) / 100)

    @staticmethod
    def _apply_nature(stats, nature):
        new_stats = stats.copy()
        new_stats[natures[nature]['plus']] *= 1.1
        new_stats[natures[nature]['minus']] *= 0.9

        return new_stats

    @staticmethod
    def calculate_stats(base_stats, ivs, evs, nature, level):
        stats = {}

        stats['hp'] = StatCalc._stat_calc(
            base_stats['hp'],
            ivs['hp'],
            evs['hp'],
            level
        ) + level + 10

        stats['atk'] = StatCalc._stat_calc(
            base_stats['atk'],
            ivs['atk'],
            evs['atk'],
            level
        ) + 5

        stats['def'] = StatCalc._stat_calc(
            base_stats['def'],
            ivs['def'],
            evs['def'],
            level
        ) + 5

        stats['spa'] = StatCalc._stat_calc(
            base_stats['spa'],
            ivs['spa'],
            evs['spa'],
            level
        ) + 5

        stats['spd'] = StatCalc._stat_calc(
            base_stats['spd'],
            ivs['spd'],
            evs['spd'],
            level
        ) + 5

        stats['spe'] = StatCalc._stat_calc(
            base_stats['spe'],
            ivs['spe'],
            evs['spe'],
            level
        ) + 5

        stats = StatCalc._apply_nature(stats, nature)
        stats = {k: int(v) for k, v in stats.items()}
        return stats