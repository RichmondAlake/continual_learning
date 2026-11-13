import re


def normalise(text):
    text = text.lower().replace("’", "'")
    return re.sub(r"(?<=\d),(?=\d)", "", text)  # "£2,500" -> "£2500"


def _pattern(alt):
    p = re.escape(alt)
    if alt[0].isdigit():
        p = r"(?<!\d)" + p
    if alt[-1].isdigit():
        p = p + r"(?!\d)"  # so "£250" does not match "£2500"
    return p


def score(answer, must_include):
    """Pass if every group of key terms has at least one alternative in the answer."""
    a = normalise(answer)
    return all(any(re.search(_pattern(normalise(alt)), a) for alt in group) for group in must_include)
