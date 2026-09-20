"""Boy subject filter: a tag that needs a man, in a prompt that says "solo".

"sex" says two people where "1girl, solo" says one. The tag stays and the
subject tags give way: "solo" goes, and the man is counted in if he is not
there yet. The node does the rewriting; what is here only reads the tags.
"""

import re

# How a prompt names its subjects: the count tags and the "multiple" forms.
_SUBJECT_RE = re.compile(r"\d+\+?(girl|boy|other)s?|multiple (girl|boy|other)s")

# Tags that only hold with a man in the picture. Matched against the whole
# tag, never a substring: "sex" as a substring would take "sex toy",
# "after sex", "unisex" and "sexy" with it.
BOY_TAGS = frozenset(
    {
        "sex",
        "hetero",
        "vaginal",
        "anal",
        "oral",
        "fellatio",
        "irrumatio",
        "deepthroat",
        "cunnilingus",
        "paizuri",
        "handjob",
        "footjob",
        "sex from behind",
        "standing sex",
        "clothed sex",
        "happy sex",
        "doggystyle",
        "missionary",
        "prone bone",
        "mating press",
        "cowgirl position",
        "girl on top",
        "straddling",
        "spitroast",
        "male penetrated",
        "group sex",
        "gangbang",
        "clothed female nude male",
    }
)

# Families too long to spell out; fullmatch, so no anchors. The cum family
# is absent on purpose: those tags hold for one person after the fact, and
# reading a man into them would undo what the solo guard let through.
BOY_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\w+ handjob",
        r"\w+ footjob",
        r"\w+ fellatio",
        r"\w+ paizuri",
        r"(double|triple) penetration",
        r"\w+ threesome",
        r"bisexual female",
        r"(reverse |squatting )?cowgirl position",
        r"(reverse )?(upright straddle|suspended congress|spitroast)",
    )
)

# A tag spelled with "another" names a second person by definition:
# "grabbing another's hand", "undressing another".
_ANOTHER = "another"

# A male body in the picture is read as a man in the picture.
MALE_BODY = frozenset({"penis", "erection", "precum", "testicles", "male focus"})

DEFAULT_ADD_TAGS = "(hetero:1.1), (couple:1.1), (deep skin:1.1)"


def _key(tag):
    return tag.replace("_", " ").strip().lower()


def _boy_tag(key):
    # "after X" is the aftermath, which one person can hold alone
    if key.startswith("after "):
        return False
    return (
        key in BOY_TAGS
        or key in MALE_BODY
        or _ANOTHER in key
        or any(p.fullmatch(key) for p in BOY_PATTERNS)
    )


def needs_boy(tags):
    """True when one of `tags` only holds with a man in the picture."""
    return any(_boy_tag(_key(t)) for t in tags)


def counts_boy(tags):
    """True when one of `tags` is a boy subject: 1boy, 2boys, multiple boys."""
    return any(_SUBJECT_RE.fullmatch(_key(t)) and "boy" in _key(t) for t in tags)


def is_solo(tag):
    return _key(tag) == "solo"
