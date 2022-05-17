import re


def normalize_name(name):
    return "".join(re.findall("[a-zA-Z]+", name)).replace(" ", "").lower()

