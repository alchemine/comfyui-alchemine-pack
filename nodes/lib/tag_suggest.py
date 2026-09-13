"""TagSuggest: recommend tags that usually accompany the input tags.

The other direction of the same statistic TagVeto uses: lift far below 1
means two tags avoid each other, lift far above 1 means they attract.
suggest_v1.0.npz (20,811 tags over 9.23M 2026 posts) stores both ends --
each tag's strongest attraction neighbours per rating tier, and one
shared table of its strongest repulsions. Repulsion matters because an
unstored pair reads as neutral: without it the sampler can only ever be
pulled toward a tag, never pushed away from an awkward one.

score(t | inputs) = log P(t) + sum over inputs of log lift(t, input),
attraction and repulsion included. Attraction alone decides candidacy,
so a tag nothing in the prompt calls for stays out; repulsion only
discounts. Each pick joins the context, must pass the TagVeto gate, and
may be limited by category quota and rating level.
"""
import math
import re

from . import (artifact, tag_alias, tag_avoid, tag_category, tag_solo,
               tag_subject, tag_veto)
from .tag_category import RATING_ORDER
from .tag_veto import normalize, split_prompt_tags, DEFAULT_LIFT_TH
from .utils import get_logger

DEFAULT_MIN_COUNT = 5000
_MIN_REPEL_LIFT = 1e-6               # keeps log() finite on stored zeros

# must match build_suggest.py: stored repulsion lift is smoothed as
# (observed + ALPHA) / (expected + ALPHA), which is invertible given the
# tag counts, so the raw ratio TagVeto thresholds on can be recovered.
_SMOOTHING = 5.0
_MIN_EXPECTED = 15.0

# How much a pick conditions the picks after it, relative to a prompt
# tag. At 0 every tag answers to the prompt alone and they end up with
# nothing to do with each other -- mechanical arms next to oversized
# wings next to a leg tattoo. At 1.0 they cohere into one scene, at the
# price of running away with it: "chair" pulls "office chair" pulls
# "computer keyboard" and the prompt stops mattering. The category
# quotas now cap any single axis, which is what used to make the high
# end dangerous, so this sits in the middle rather than low.
DEFAULT_MOMENTUM = 0.5

# How much the odds of a tag are divided by for each tag already picked
# that shares one of its slots (see TagSuggest.slots). 2.0 halves them
# each time: the second "<colour> skin" needs twice the evidence the
# first one did, the third four times. 1.0 disables it, and the
# direction follows the LLM convention -- raise it to push harder.
#
# It exists because momentum pulls hardest along the axis it just moved
# on: one "blue skin" makes every other skin colour a top neighbour,
# one "hands on own face" makes every other "hands on own ..." one, and
# a draw can spend half its budget enumerating one noun. The category
# quotas cannot see this -- all of those tags live in the same category.
#
# Exact repeats are not its business: a tag already in the context is
# banned outright, whatever this is set to.
DEFAULT_REPETITION_PENALTY = 2.0

# auto-length EOS: stop when no candidate is at least this much more
# likely than chance given the context (combined lift >= 2)
_EOS_LOG_LIFT = 0.6931471805599453   # ln(2)
_FALLBACK_TARGET_LEN = 31            # solo-post median, if len_hist absent

# rating is a ceiling, so the milder tiers below it keep dominating the
# pool by sheer count. Tags labelled at exactly the requested level get
# their odds multiplied by this, so asking for "explicit" leans explicit
# instead of merely permitting it. 1.0 disables the tilt.
_RATING_BIAS = 2.0

# "all" is not a fifth rating but the absence of one: the tables come
# from the whole corpus and no level is capped or favoured, so what
# comes out is whatever the prompt calls for -- explicit tags for a
# nude prompt, none at all for a school uniform one.
#
# Per-level multipliers, mildest first, in case that should ever be
# nudged. Measured on three prompts x 40 seeds, (1, 1, 2, 2) moves the
# signal-less "1girl, solo" from 11% explicit to 18%, but pushes
# "1girl, solo, nude, bed" from 83% to 96% and leaves a school-uniform
# prompt where it was: a constant factor cannot compete with lift,
# which swings by orders of magnitude, so it bites hardest exactly
# where it is least wanted. Hence 1.0 across the board.
_RATING_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Words that link the two halves of a tag, or trail a verb as a
# particle. The list only has to name them; which role a word plays in
# a given tag falls out of whether anything follows it, so "up" needs
# no entry of its own in "hand_up" versus "looking_up".
_PARTICLES = frozenset("""
    on in at to of by with from into onto under over above below behind
    between around against across through beside near up down out off
    back together apart forward aside away
""".split())


def _depluralize(word):
    """Crude singular, so "hands_on_own_hip" and "hand_on_own_face"
    land in one slot. Short words are left alone: "ass" and "grass" are
    not plurals, and nothing is gained by splitting hairs over the few
    that slip through -- a wrong merge costs one discount, not a ban.
    """
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _slot_keys(tag):
    """The one or two slots a tag occupies. See TagSuggest.slots."""
    words = tag.split("_")
    at = next((i for i, w in enumerate(words)
               if i > 0 and w in _PARTICLES), None)
    if at is None:
        return (("W", _depluralize(words[-1])),)
    left = ("L", words[at], "_".join(_depluralize(w) for w in words[:at]))
    if at == len(words) - 1:                  # particle, nothing follows
        return (left,)
    return (left, ("R", words[at], "_".join(words[at + 1:])))


_SUGGEST_PATH = artifact.resource("suggest_v1.1.npz")
# not committed (106MB); fetched from the data release on first use.
# v1.1 keeps 384 attraction neighbours instead of 256, at least 24 of
# them per category: a plain top-K row goes to whichever axis the tag
# pulls hardest, and since a tag has to appear in some context tag's row
# to be sampled at all, the axes it starves become unreachable.
_SUGGEST_URL = artifact.url_for("data-v4", "suggest_v1.1.npz")
_SUGGEST_SHA256 = ("90248ad9142e28b76d008071cbebfa92c7162c1a"
                   "b75002b7011423266d69248f")


logger = get_logger()


class TagSuggest:
    """Suggest companion tags from the attraction-neighbor table."""

    def __init__(self, path=_SUGGEST_PATH):
        import numpy as np
        self._np = np
        if path == _SUGGEST_PATH:
            artifact.ensure(path, _SUGGEST_URL, _SUGGEST_SHA256,
                            "TagSuggest", "106MB")
        data = np.load(path)
        self.vocab = [str(t) for t in data["tags"]]
        # same alias folding as TagVeto: old and new spellings share a row
        self.index = tag_alias.expand_index(
            {t: i for i, t in enumerate(self.vocab)})
        # per-rating-tier tables; tiers are cumulative (g < s < q < e).
        # legacy single-table artifacts load as tier "e" only.
        self._tiers = {}
        # repulsion is one table over every post, shared by all tiers;
        # pre-v2 artifacts have none
        neg_ids = data.get("neg_ids", None)
        neg_lift = (data["neg_lift"].astype(np.float32)
                    if "neg_lift" in data else None)
        for r in ("g", "s", "q", "e"):
            if f"nbr_ids_{r}" in data:
                self._tiers[r] = {
                    "counts": data[f"counts_{r}"],
                    "ids": data[f"nbr_ids_{r}"],
                    "lift": data[f"nbr_lift_{r}"].astype(np.float32),
                    "len_hist": data[f"len_hist_{r}"].astype(np.float64),
                    "posts": (float(data[f"posts_{r}"])
                              if f"posts_{r}" in data else 0.0),
                    "neg_ids": neg_ids,
                    "neg_lift": neg_lift,
                }
        if not self._tiers:
            self._tiers["e"] = {
                "counts": data["counts"],
                "ids": data["nbr_ids"],
                "lift": data["nbr_lift"].astype(np.float32),
                "len_hist": (data["len_hist"].astype(np.float64)
                             if "len_hist" in data else None),
                "neg_ids": None,
                "neg_lift": None,
            }
        self._labels = None          # lazy (category, rating) arrays
        self._avoid = None           # lazy avoidance table
        self._blacklist = None       # (pattern, mask) of the last regex
        self._slots = None           # lazy (slot ids per tag, count)
        self._solo = None            # lazy (multi-person, male) vetoes
        self._subject = None         # lazy subject-conjunction table
        self._veto_ids = None        # lazy vocab mapped onto TagVeto's

    def labels(self):
        """(category index, rating level) per vocab entry, or None."""
        if self._labels is None:
            source = tag_category.load_labels()
            self._labels = (source, source.arrays(self.vocab)) if source \
                else (None, (None, None))
        return self._labels

    def slots(self):
        """(two slot ids per vocab entry, number of distinct slots).

        A slot is "the same place in the picture", and
        repetition_penalty discounts a candidate once per tag already
        picked that shares one. Two spellings feed it:

        A linking word with material on both sides splits the tag in
        two -- "hands_on_own_face" is both a hands tag and an own_face
        tag, so it collides with "hands_on_own_head" on the left and
        with "blood_on_face" on the right. Both are real runaways: one
        pair of hands in three places, three things on one face.

        With nothing to its right the word is a particle rather than a
        preposition ("looking_up", "tongue_out") and only the left side
        is a slot. That falls out of the position, so the word list
        never has to say which role a word is playing.

        Everything else keeps the old rule, the last word: "blue_skin"
        and "two-tone_skin" share a slot, "blue_skin" and "blue_eyes"
        do not.

        The three kinds are kept in separate namespaces, so the slot a
        prepositional tag takes from its right side never meets a plain
        tag's last word. "hands_on_own_face" and "covering_face" are
        related, but relating them is the co-occurrence data's job --
        this is only meant to catch the lexical families, where the
        same words are being re-spelled around one axis.
        """
        if self._slots is None:
            np = self._np
            index = {}
            ids = np.zeros((len(self.vocab), 2), dtype=np.int32)
            index[None] = 0                           # the empty slot
            for i, tag in enumerate(self.vocab):
                for k, key in enumerate(_slot_keys(tag)):
                    ids[i, k] = index.setdefault(key, len(index))
            self._slots = (ids, len(index))
        return self._slots

    def solo_masks(self):
        """(multi-person veto, male-anatomy veto) over the vocabulary.

        Neither depends on the prompt, only on which tags exist, so they
        are built once and indexed per draw (see tag_solo).
        """
        if self._solo is None:
            self._solo = tag_solo.masks(self.vocab)
        return self._solo

    def _tier(self, rating):
        return self._tiers.get(rating, self._tiers["e"])

    def _log_lift_sum(self, ids, tier):
        """Sum of log lift(t, c) over context tags c, split by direction.

        Attraction and repulsion are kept apart because they answer
        different questions: attraction decides whether a tag is a
        candidate at all, while repulsion only discounts one. Pairs in
        neither table are neutral (log-lift 0).
        """
        np = self._np
        attract = np.zeros(len(self.vocab))
        repel = np.zeros(len(self.vocab))
        for i in ids:
            row_lift = tier["lift"][i]
            real = row_lift > 0                      # drop padding
            attract[tier["ids"][i][real]] += np.log(row_lift[real])
            if tier["neg_ids"] is None:
                continue
            neg_ids = tier["neg_ids"][i]
            real = neg_ids >= 0                      # -1 is padding
            repel[neg_ids[real]] += np.log(
                np.maximum(tier["neg_lift"][i][real], _MIN_REPEL_LIFT))
        return attract, repel

    def _blacklist_mask(self, pattern):
        """Vocabulary entries a user regex rejects, or None for no filter.

        Matched with search() against the spaced form the node reads and
        writes, so "hair" rejects every hair tag and "^black " only the
        ones starting that way. An unparseable pattern is reported and
        ignored rather than raised: a typo should not stop generation.
        """
        np = self._np
        if not pattern or not pattern.strip():
            return None
        if self._blacklist and self._blacklist[0] == pattern:
            return self._blacklist[1]
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            logger.error("[TagSuggest] ignoring invalid blacklist regex %r (%s)"
                  % (pattern, exc), exc_info=True)
            return None
        mask = np.fromiter(
            (rx.search(t.replace("_", " ")) is not None for t in self.vocab),
            dtype=bool, count=len(self.vocab))
        self._blacklist = (pattern, mask)
        return mask

    def _veto_vocab_ids(self, veto):
        """This vocabulary in TagVeto's numbering, built once."""
        if self._veto_ids is None:
            self._veto_ids = veto.vocab_ids(self.vocab)
        return self._veto_ids

    def _subject_joint(self):
        """Lazy handle on the subject-conjunction table, or None."""
        if self._subject is None:
            self._subject = tag_subject.load_subject_joint(self.vocab)
        return self._subject or None

    def _avoidance(self):
        """Lazy handle on the avoidance table, or None when absent."""
        if self._avoid is None:
            self._avoid = tag_avoid.load_avoidance(self.vocab)
        return self._avoid or None

    def _repel_veto(self, ids, tier, lift_th, avoid_alpha):
        """Tags that avoid the given context strongly enough to ban.

        TagVeto only knows 8,320 of the 20,811 tags, so on its own it
        leaves most of the vocabulary unchecked. The repulsion table
        covers all of it: undo the smoothing to recover the raw
        observed/expected ratio and apply the same rule TagVeto does --
        ban when the corpus expected the pair often (>= 15) and it still
        barely happened.
        """
        np = self._np
        banned = np.zeros(len(self.vocab), dtype=bool)
        if tier["neg_ids"] is None or not tier["posts"]:
            return banned
        counts = tier["counts"].astype(np.float64)
        for i in ids:
            neighbours = tier["neg_ids"][i]
            real = neighbours >= 0
            j = neighbours[real]
            if not len(j):
                continue
            smoothed = tier["neg_lift"][i][real].astype(np.float64)
            expected = counts[i] * counts[j] / tier["posts"]
            observed = smoothed * (expected + _SMOOTHING) - _SMOOTHING
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(expected > 0, observed / expected, 1.0)
            banned[j[(expected >= _MIN_EXPECTED) & (raw < lift_th)]] = True
        avoid = self._avoidance()
        if avoid is not None:
            banned |= avoid.mask(ids, avoid_alpha)
        return banned

    def _eligible(self, counts, min_count, rating, level_of, cat_of,
                  allowed, blacklist):
        """Tags allowed to be drawn at all, before any context is read.

        Everything here is a property of the request rather than of the
        picks, so it is computed once and never revisited: how common a
        tag is, how explicit, which category it belongs to, and whether
        the user's regex rejects it.
        """
        eligible = counts >= min_count
        if level_of is not None and rating != "all":
            eligible &= level_of <= RATING_ORDER.index(rating)
        if allowed is not None and cat_of is not None:
            eligible &= self._np.isin(cat_of, list(allowed))
        rejected = self._blacklist_mask(blacklist)
        if rejected is not None:
            eligible &= ~rejected
        return eligible

    def _rating_log_weights(self, rating):
        """Per-level log multiplier on the prior, or None for a flat one.

        A named rating tilts toward itself by _RATING_BIAS; "all" spends
        _RATING_WEIGHTS, which is flat unless someone edits it.
        """
        np = self._np
        if rating == "all":
            w = np.log(np.asarray(_RATING_WEIGHTS, dtype=np.float64))
        else:
            w = np.zeros(len(RATING_ORDER))
            w[RATING_ORDER.index(rating)] = math.log(_RATING_BIAS)
        return w if w.any() else None

    def _log_prior(self, counts, level_of, rating):
        """log P(t), tilted toward the requested rating tier."""
        np = self._np
        # a few composition tags have ~0 solo-corpus count; floor at 1
        prior = np.log(np.maximum(counts.astype(np.float64), 1.0)
                       / counts.sum())
        w = self._rating_log_weights(rating)
        if level_of is not None and w is not None:
            prior = prior + w[np.clip(level_of, 0, len(w) - 1)]
        return prior

    def suggest(self, inputs, m=10, min_count=DEFAULT_MIN_COUNT,
                lift_th=DEFAULT_LIFT_TH, temperature=0.0,
                top_k=0, top_p=1.0, min_p=0.0, seed=0, rating="e",
                categories="", blacklist="", quota_total=None,
                avoid_alpha=tag_avoid.DEFAULT_ALPHA,
                momentum=DEFAULT_MOMENTUM,
                repetition_penalty=DEFAULT_REPETITION_PENALTY):
        """Return up to m tags (Danbooru form) that go with the inputs.

        One tag per step, LM-style. The step distribution is naive Bayes:
        log P(t | context) = log P(t) + sum_i log lift(t, context_i),
        and every pick joins the context, re-conditioning the next step.
        temperature 0 = greedy argmax; above 0 the usual sampling filters
        (top_k, top_p, min_p) apply. Vetoed candidates are masked, so the
        output cannot contradict the inputs or itself.

        Each knob is documented once, on the TagGenerator widget that
        turns it (nodes/prompt.py); only what the widgets cannot say is
        repeated here:

        m <= 0 selects auto length: a target count is drawn from the
        solo-post length distribution (median 31) and generation also
        stops at the EOS analog -- no candidate at least twice as likely
        as chance, i.e. the data has nothing left to say.

        rating works on both halves of the statistic: it selects the
        cumulative corpus subset the lift tables come from, and it caps
        the rating level of the tags themselves. Since the cap admits
        every milder tier too, tags at exactly the requested level are
        multiplied by _RATING_BIAS, so the request reads as a leaning
        rather than only a ceiling. "all" is the absence of a request:
        the whole corpus, no cap, no tilt.

        quota_total overrides m as the base the category shares are
        fractions of, for a caller that asks for more than it keeps.

        inputs: tags in any prompt form; out-of-vocabulary ones are
        ignored for scoring but still block duplicates.
        """
        np = self._np
        tier = self._tier(rating)
        counts = tier["counts"]
        tags = [normalize(t) for t in inputs]
        ids = [self.index[t] for t in tags if t in self.index]
        if not ids:
            return []
        rng = np.random.default_rng(seed)

        source, (cat_of, level_of) = self.labels()
        allowed, quota = (tag_category.parse_categories(categories,
                                                        source.names)
                          if source else (None, None))
        used = {}

        if m <= 0:
            m = max(0, self._draw_length(rng, tier) - len(tags))
            if m == 0:
                return []
            auto = True
        else:
            auto = False
        # shares are fractions of the final count, so they can only be
        # resolved once m is known -- including the auto-length case,
        # where the corpus draw above supplies it. quota_total overrides
        # it for a caller that asks for more than it intends to keep:
        # scaling the caps to an inflated m would loosen them.
        quota = tag_category.resolve_quota(quota, quota_total or m)
        veto = tag_veto.load_veto()
        log_prior = self._log_prior(counts, level_of, rating)
        eligible = self._eligible(counts, min_count, rating, level_of,
                                  cat_of, allowed, blacklist)
        log_lift, log_repel = self._log_lift_sum(ids, tier)

        # One mask for everything a candidate can be ruled out by, since
        # the loop only ever asks whether it is ruled out: tags already
        # in the prompt, tags the corpus shows the context avoiding,
        # tags the subject tags rule out together, tags TagVeto judges
        # against a reference. All four only ever grow.
        chosen, refs = [], [t for t in tags if t]
        banned = self._repel_veto(ids, tier, lift_th, avoid_alpha)
        for t in refs:
            if t in self.index:
                banned[self.index[t]] = True

        # what two subject tags rule out between them -- the one thing
        # the pairwise tables cannot say (see tag_subject). Recomputed
        # when a pick adds a subject tag, since that makes new pairs.
        joint = self._subject_joint()
        subject_ids = list(ids)
        if joint is not None:
            banned |= joint.mask(subject_ids)

        # ...and what one person cannot do at all, which is a rule
        # rather than a measurement: the corpus is not surprised by
        # "solo" beside "rape" or "hug", so no table here can object to
        # it (see tag_solo). Read off the prompt alone -- every tag that
        # could change the count is on the list it applies.
        alone, female = tag_solo.context(tags)
        if alone:
            multi_veto, male_veto = self.solo_masks()
            banned |= multi_veto
            if female:
                banned |= male_veto

        # The veto verdict for a candidate only depends on the reference
        # tags, and refs only ever grows by the tag just picked, so a
        # candidate cleared against refs[:k] never has to be re-judged
        # against them. refs_judged is how far the vocabulary has been
        # weighed; each step pays for the new reference only.
        refs_judged = 0
        veto_ids = self._veto_vocab_ids(veto) if veto else None

        # Repetition penalty, in log space: dividing the odds by
        # repetition_penalty per repeat is subtracting its log per
        # repeat, so the geometric decay and the log-linear penalty are
        # the same statement. A tag sits in up to two slots, and hits
        # add across both. The prompt's own tags seed the counts --
        # asking to extend "pale skin" should already discount the next
        # skin tag, not wait for the sampler to pick one itself.
        slot_of, n_slots = self.slots()
        log_penalty = (math.log(repetition_penalty)
                       if repetition_penalty > 1 else 0.0)
        slot_used = np.zeros(n_slots, dtype=np.int32)
        if log_penalty:
            np.add.at(slot_used, slot_of[ids].ravel(), 1)
            slot_used[0] = 0                      # the empty slot never hits

        for _ in range(m):
            if veto:
                # one vectorised pass per new reference beats a Python
                # call per candidate by ~40x, so the whole vocabulary is
                # judged at once rather than only the live candidates
                while refs_judged < len(refs):
                    banned |= veto.conflict_mask(
                        veto_ids, refs[refs_judged], lift_th)
                    refs_judged += 1
            # candidates: some attraction, allowed, not ruled out
            ok = (log_lift > 0) & eligible & ~banned
            if quota and cat_of is not None:
                spent = [r for r, (key, cap) in quota.items()
                         if used.get(key, 0) >= cap]
                if spent:
                    ok &= ~np.isin(cat_of, spent)
            cand = np.nonzero(ok)[0]
            if not len(cand):
                break
            if auto and log_lift[cand].max() < _EOS_LOG_LIFT:
                break                                 # nothing left to say
            # only the candidates can be picked, so the score and the
            # sampling filters run over them alone rather than over a
            # 20k vector that is masked off almost everywhere
            logits = (log_prior[cand] + log_lift[cand] + log_repel[cand])
            if log_penalty:
                hits = slot_used[slot_of[cand]].sum(axis=1)
                logits = logits - log_penalty * hits
            j = int(cand[self._pick(logits, rng, temperature,
                                    top_k, top_p, min_p)])
            tag = self.vocab[j]
            chosen.append(tag)
            refs.append(tag)                          # picks must cohere
            banned[j] = True                          # and cannot repeat
            if log_penalty:
                np.add.at(slot_used, slot_of[j], 1)
                slot_used[0] = 0
            if quota and cat_of is not None:
                budget = quota.get(int(cat_of[j]))
                if budget is not None:
                    used[budget[0]] = used.get(budget[0], 0) + 1
            gain, loss = self._log_lift_sum([j], tier)   # re-condition
            log_lift += momentum * gain
            log_repel += momentum * loss
            banned |= self._repel_veto([j], tier, lift_th, avoid_alpha)
            if joint is not None and j in joint.subject_ids:
                subject_ids.append(j)
                banned |= joint.mask(subject_ids)
        return chosen

    def _draw_length(self, rng, tier):
        """Target total tag count, drawn from the tier's distribution."""
        if tier["len_hist"] is None:
            return _FALLBACK_TARGET_LEN
        p = tier["len_hist"] / tier["len_hist"].sum()
        return int(rng.choice(len(p), p=p))

    def _pick(self, logits, rng, temperature, top_k, top_p, min_p):
        """One sampling step: temperature -> top_k -> min_p -> top_p.

        Returns a position in `logits`, which holds the candidates only.
        """
        np = self._np
        if temperature <= 0:
            return int(logits.argmax())
        if 0 < top_k < len(logits):
            # only the top_k survive, so partition them out first
            # instead of ranking every candidate to keep 50
            order = np.argpartition(logits, -top_k)[-top_k:]
            order = order[np.argsort(logits[order])[::-1]]
        else:
            order = np.argsort(logits)[::-1]
        p = np.exp((logits[order] - logits[order[0]]) / temperature)
        p /= p.sum()
        if min_p > 0:
            keep = p >= min_p * p[0]
            order, p = order[keep], p[keep] / p[keep].sum()
        if top_p < 1.0:
            cut = int(np.searchsorted(np.cumsum(p), top_p)) + 1
            order, p = order[:cut], p[:cut] / p[:cut].sum()
        return int(rng.choice(order, p=p))


@artifact.lazy
def load_suggest():
    """The suggest table, or False when the artifact is missing."""
    return TagSuggest()


def suggest_available():
    return bool(load_suggest())


def category_names():
    """User-facing category names, or () when the labels are missing."""
    labels = tag_category.load_labels()
    return tuple(labels.names) if labels else ()


def suggest_tags(prompt, n=10, min_count=DEFAULT_MIN_COUNT,
                 temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, seed=0,
                 rating="e", categories="", blacklist="",
                 lift_th=DEFAULT_LIFT_TH, quota_total=None,
                 avoid_alpha=tag_avoid.DEFAULT_ALPHA,
                 momentum=DEFAULT_MOMENTUM,
                 repetition_penalty=DEFAULT_REPETITION_PENALTY):
    """Comma-separated prompt in, list of suggested tags (space form) out."""
    engine = load_suggest()
    inputs = split_prompt_tags(prompt)
    tags = engine.suggest(inputs, m=n, min_count=min_count,
                          lift_th=lift_th,
                          temperature=temperature, top_k=top_k,
                          top_p=top_p, min_p=min_p, seed=seed, rating=rating,
                          categories=categories, blacklist=blacklist,
                          quota_total=quota_total,
                          avoid_alpha=avoid_alpha, momentum=momentum,
                          repetition_penalty=repetition_penalty)
    # keep emoticon tags (^_^, o_o) intact: only wordlike tags get spaces
    return [t.replace("_", " ") if re.search(r"[a-z]", t) else t
            for t in tags]
