# -*- coding: utf-8 -*-
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
import os
import re

from . import artifact, tag_category, tag_veto
from .tag_category import RATING_ORDER
from .tag_veto import normalize, DEFAULT_LIFT_TH

DEFAULT_MIN_COUNT = 5000
_MIN_REPEL_LIFT = 1e-6               # keeps log() finite on stored zeros

# must match build_suggest.py: stored repulsion lift is smoothed as
# (observed + ALPHA) / (expected + ALPHA), which is invertible given the
# tag counts, so the raw ratio TagVeto thresholds on can be recovered.
_SMOOTHING = 5.0
_MIN_EXPECTED = 15.0

# auto-length EOS: stop when no candidate is at least this much more
# likely than chance given the context (combined lift >= 2)
_EOS_LOG_LIFT = 0.6931471805599453   # ln(2)
_FALLBACK_TARGET_LEN = 31            # solo-post median, if len_hist absent

_SUGGEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "resources", "suggest_v1.0.npz")
# not committed (72MB); fetched from the data release on first use
_SUGGEST_URL = artifact.url_for("data-v3", "suggest_v1.0.npz")
_SUGGEST_SHA256 = ("aa6527c1c011dd203234ae6632094d65fa94f4eb"
                   "e1a9cda808de1b89280c8e07")

_SUGGEST = None  # lazy singleton: TagSuggest, or False if unavailable


class TagSuggest:
    """Suggest companion tags from the attraction-neighbor table."""

    def __init__(self, path=_SUGGEST_PATH):
        import numpy as np
        self._np = np
        if path == _SUGGEST_PATH:
            artifact.ensure(path, _SUGGEST_URL, _SUGGEST_SHA256,
                            "TagSuggest", "72MB")
        data = np.load(path)
        self.vocab = [str(t) for t in data["tags"]]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        # per-rating-tier tables; tiers are cumulative (g < s < q < e).
        # legacy single-table artifacts load as tier "e" only.
        self._tiers = {}
        # repulsion is one table over every post, shared by all tiers;
        # pre-v2 artifacts have none
        neg_ids = data["neg_ids"] if "neg_ids" in data else None
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
        self._blacklist = None       # (pattern, mask) of the last regex

    def labels(self):
        """(category index, rating level) per vocab entry, or None."""
        if self._labels is None:
            source = tag_category.load_labels()
            self._labels = (source, source.arrays(self.vocab)) if source \
                else (None, (None, None))
        return self._labels

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
            print("[TagSuggest] ignoring invalid blacklist regex %r (%s)"
                  % (pattern, exc))
            return None
        mask = np.fromiter(
            (rx.search(t.replace("_", " ")) is not None for t in self.vocab),
            dtype=bool, count=len(self.vocab))
        self._blacklist = (pattern, mask)
        return mask

    def _repel_veto(self, ids, tier, lift_th):
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
        return banned

    def suggest(self, inputs, m=10, min_count=DEFAULT_MIN_COUNT,
                lift_th=DEFAULT_LIFT_TH, temperature=0.0,
                top_k=0, top_p=1.0, min_p=0.0, seed=0, rating="e",
                categories="", blacklist=""):
        """Return up to m tags (Danbooru form) that go with the inputs.

        One tag per step, LM-style. The step distribution is naive Bayes:
        log P(t | context) = log P(t) + sum_i log lift(t, context_i),
        and every pick joins the context, re-conditioning the next step.
        temperature 0 = greedy argmax; above 0 the usual sampling filters
        (top_k, top_p, min_p) apply. Vetoed candidates are masked, so the
        output cannot contradict the inputs or itself.

        m <= 0 selects auto length: a target tag count is drawn from the
        solo-post length distribution (median 31) and generation also
        stops early at the EOS analog -- when no candidate is at least
        twice as likely as chance given the context, the data has nothing
        left to say.

        rating works on both halves of the statistic: it selects the
        cumulative corpus subset the lift tables come from ("g" = general
        only, "s" = g+s, "q" = g+s+q, "e" = every post), and it caps the
        rating level of the tags themselves, so a tag Danbooru only ever
        applies to racier art cannot surface in a milder request.

        categories restricts which knobs the output may come from:
        "pose, clothes" allows only those, and "pose:3, clothes:2" also
        caps how many tags each contributes -- with m unset, the caps
        become the target length. A category whose quota runs out is
        masked for the remaining steps.

        blacklist is a regex; tags it matches are removed from the
        candidates before sampling, not from the result afterwards, so
        the requested count still comes back filled.

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

        auto = m <= 0
        if auto and quota:
            m = sum(quota.values())          # the quotas are the request
            auto = False
        if auto:
            m = max(0, self._draw_length(rng, tier) - len(tags))
            if m == 0:
                return []
        veto = tag_veto.load_veto()
        # a few composition tags have ~0 solo-corpus count; floor at 1
        log_prior = np.log(np.maximum(counts.astype(np.float64), 1.0)
                           / counts.sum())

        log_lift, log_repel = self._log_lift_sum(ids, tier)
        eligible = counts >= min_count
        if level_of is not None:
            eligible &= level_of <= RATING_ORDER.index(rating)
        if allowed is not None and cat_of is not None:
            eligible &= np.isin(cat_of, list(allowed))
        banned = self._blacklist_mask(blacklist)
        if banned is not None:
            eligible &= ~banned

        chosen, refs = [], [t for t in tags if t]
        blocked = set(refs)
        vetoed = self._repel_veto(ids, tier, lift_th)
        for _ in range(m):
            logits = log_prior + log_lift + log_repel
            # candidates: some attraction, allowed, not used, not vetoed
            ok = (log_lift > 0) & eligible & ~vetoed
            if quota and cat_of is not None:
                spent = [r for r, cap in quota.items()
                         if used.get(r, 0) >= cap]
                if spent:
                    ok &= ~np.isin(cat_of, spent)
            for j in np.nonzero(ok)[0]:
                tag = self.vocab[j]
                if tag in blocked or (
                        veto and veto.conflict(tag, refs, lift_th)):
                    ok[j] = False
            if not ok.any():
                break
            if auto and log_lift[ok].max() < _EOS_LOG_LIFT:
                break                                 # nothing left to say
            logits[~ok] = -np.inf

            j = self._pick(logits, rng, temperature, top_k, top_p, min_p)
            tag = self.vocab[j]
            chosen.append(tag)
            refs.append(tag)                          # picks must cohere
            blocked.add(tag)
            if cat_of is not None:
                rank = int(cat_of[j])
                used[rank] = used.get(rank, 0) + 1
            gain, loss = self._log_lift_sum([j], tier)   # re-condition
            log_lift += gain
            log_repel += loss
            vetoed |= self._repel_veto([j], tier, lift_th)
        return chosen

    def _draw_length(self, rng, tier):
        """Target total tag count, drawn from the tier's distribution."""
        if tier["len_hist"] is None:
            return _FALLBACK_TARGET_LEN
        p = tier["len_hist"] / tier["len_hist"].sum()
        return int(rng.choice(len(p), p=p))

    def _pick(self, logits, rng, temperature, top_k, top_p, min_p):
        """One sampling step: temperature -> top_k -> min_p -> top_p."""
        np = self._np
        if temperature <= 0:
            return int(logits.argmax())
        order = np.argsort(logits)[::-1]
        order = order[np.isfinite(logits[order])]
        if top_k > 0:
            order = order[:top_k]
        p = np.exp((logits[order] - logits[order[0]]) / temperature)
        p /= p.sum()
        if min_p > 0:
            keep = p >= min_p * p[0]
            order, p = order[keep], p[keep] / p[keep].sum()
        if top_p < 1.0:
            cut = int(np.searchsorted(np.cumsum(p), top_p)) + 1
            order, p = order[:cut], p[:cut] / p[:cut].sum()
        return int(rng.choice(order, p=p))


def load_suggest():
    global _SUGGEST
    if _SUGGEST is None:
        try:
            _SUGGEST = TagSuggest()
        except Exception:
            _SUGGEST = False
    return _SUGGEST


def suggest_available():
    return bool(load_suggest())


def category_names():
    """User-facing category names, or () when the labels are missing."""
    labels = tag_category.load_labels()
    return tuple(labels.names) if labels else ()


def suggest_tags(prompt, n=10, min_count=DEFAULT_MIN_COUNT,
                 temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, seed=0,
                 rating="e", categories="", blacklist=""):
    """Comma-separated prompt in, list of suggested tags (space form) out."""
    engine = load_suggest()
    inputs = [t for t in prompt.split(",") if t.strip()]
    tags = engine.suggest(inputs, m=n, min_count=min_count,
                          temperature=temperature, top_k=top_k,
                          top_p=top_p, min_p=min_p, seed=seed, rating=rating,
                          categories=categories, blacklist=blacklist)
    # keep emoticon tags (^_^, o_o) intact: only wordlike tags get spaces
    return [t.replace("_", " ") if re.search(r"[a-z]", t) else t
            for t in tags]
