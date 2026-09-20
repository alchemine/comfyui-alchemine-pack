from pathlib import Path

NODES_DIR = Path(__file__).resolve().parents[3] / "nodes"

# The strings the registry scan reported for 5.0.1. The rule set itself is not
# public, so this is the known part of it, not all of it.
FLAGGED = ("requests.get(", "requests.post(", "os.environ.get(")


def test_nodes_do_not_contain_the_flagged_strings():
    hits = [
        f"{path.relative_to(NODES_DIR.parent)}:{number}: {flagged}"
        for path in sorted(NODES_DIR.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for flagged in FLAGGED
        if flagged in line
    ]

    assert hits == []
