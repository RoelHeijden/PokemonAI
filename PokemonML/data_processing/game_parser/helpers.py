import re


def normalize_name(name):
    return "".join(re.findall("[a-zA-Z0-9]+", name)).replace(" ", "").lower()

