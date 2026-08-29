# Tag group data

What TagGenerator consults to answer "which knob does this tag turn,
and how explicit is it". Vocab: general tags with count >= 100 in
danbooru-2026-clean-metadata (20,811 tags); keys use spaces, npz vocabs
use underscores. The layers are versioned independently because they
change for unrelated reasons — a wiki refresh moves tags, a UI decision
moves categories — so record which combination a build used.

The statistics themselves live in `../suggest_v1.0.npz`, built by
playground/tag_grouping/build_suggest.py; `ratings_v*.json` is derived
from it, the rest are derived from the Danbooru wiki.

## tags_v*.json — tag -> [group, ...]

Cumulative versions (each includes everything before it; higher-trust
sources win on conflict):

| file | adds | source | tags |
|---|---|---|---|
| tags_v0.1.json | wiki membership | kierarkia/danbooru-wiki-2026 dump, parsed by playground/tag_grouping/extract_tag_groups.py | 13,990 |
| tags_v0.2.json | + modifier inheritance (white shirt -> shirt) | playground/tag_grouping/apply_rules.py | 15,592 |
| tags_v1.0.json | + LLM classification of the remainder | Claude subagents; merged result kept as playground/tag_grouping/tag_groups_llm.json | 25,984 — covers 100% of the vocab |

## hierarchy_v1.0.json — group tree

The group hierarchy from the Danbooru wiki "tag groups" page
(playground/tag_grouping/parse_groups.py), plus additions the wiki's
ToC lacks: 5 custom groups (furniture and household, animal girls,
relative traits, effects and magic, franchise concepts) and 12 orphan
group pages attached to their natural parents (dress/panties/bra/
attire#shoes/embellishment under attire, tail/penis/skin folds under
body parts, meme under more, ...).

## ratings_v*.json — tag -> rating level (g/s/q/e)

Built by playground/tag_grouping/build_ratings.py from the cumulative
rating tiers in suggest_v1.0.npz. One rule, one parameter, no curated
lists: a tag's level is the first cut (g, s, q) whose cumulative odds
ratio reaches THETA=0.2, else e. Since rating filters are cumulative
("rating=s" allows g and s), the level answers "how mild a tier can
still host this tag" rather than "which tier is it most typical of".
Covers the npz vocabulary (8,320); tags outside it are unrated.

Levels: g 6,979 / s 583 / q 200 / e 558.

## categories_v*.json — user-facing primary categories, and how tags reach them

Alias layer folding hierarchy nodes into the 10 categories the
TagGenerator node exposes: characters, expressions, pose, clothes,
background, compositions, body, objects, creatures, etc. These follow
prompt-writing axes rather than the wiki's own sections, so expressions
and pose are lifted out of `body` and backgrounds out of
`image composition`.

A tag's category is the one most of its groups point at, ties broken by
`priority` order. Strict priority was tried first and lost to wiki
noise — a single stray group ("beach" is listed under `swimsuit`,
"building" under the gerund list) was enough to move a tag to the
wrong knob.

Distribution: clothes 6,263 / objects 3,529 / pose 3,169 /
compositions 1,863 / background 1,807 / body 1,472 / expressions 966 /
etc 644 / creatures 575 / characters 523.
