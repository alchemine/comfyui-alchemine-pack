# -*- coding: utf-8 -*-
"""TagVeto: drop suggested tags that contradict the reference tags.

Runtime port of playground/experiments/veto.ipynb. Built on
danbooru2026_clean (5.48M solo posts, 8,320 general tags).

Which suggestion is *best* is subjective; which suggestion is
*impossible* is not. Two frequent tags that almost never share a post
are mutually exclusive in practice, which shows up as
lift = observed / expected co-occurrence far below 1.

Judgment rules (see the notebook for the measurements behind each):
  - veto when lift < lift_th. The E gate (expected co-occurrence >= 15,
    below which an observed 0 proves nothing) is baked into the artifact.
  - composition tags (2girls, yuri, ...) are judged on the unfiltered
    corpus: the solo corpus starves them and makes their lift
    arithmetically 1.0.
  - character-count/gender tags are only compared with each other
    (1boy vs large_breasts is avoidance, not substitution).
  - fixed tags are assumed consistent and always kept; surviving
    suggestions immediately become references.

lift_th is the only tuned parameter.
"""
import os
import re
from collections import namedtuple

from . import artifact

E_MIN = 15.0
DEFAULT_LIFT_TH = 0.1

_VETO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "resources", "tag_veto.npz")
# not committed (13MB); fetched from the data release on first use
_VETO_URL = artifact.url_for("data-v3", "tag_veto.npz")
_VETO_SHA256 = ("3fb3603bcadef8e8add34eb742836215e3c98264"
                "cbcdec7007b16c2215b4bb37")


# --- tag normalization -----------------------------------------------------

_WEIGHT_RE = re.compile(r":\s*[0-9.]+\s*$")
_SUBJECT_RE = re.compile(r"^\d+\+?(boy|girl|other)s?$")
_SUBJECT_TAGS = {"solo", "solo_focus", "male_focus", "female_focus",
                 "multiple_boys", "multiple_girls", "multiple_others",
                 "no_humans", "everyone"}


def normalize(tag):
    """Prompt token -> Danbooru form: '(blonde hair:1.2)' -> 'blonde_hair'."""
    t = tag.strip().lower().replace("\\(", "(").replace("\\)", ")")
    t = _WEIGHT_RE.sub("", t).strip()
    while t.startswith("(") and t.endswith(")"):
        t = t[1:-1].strip()
    return re.sub(r"\s+", "_", t.replace("_", " ").strip())


def is_subject(tag):
    """Character-count/gender tags (1boy, 2girls, solo, ...)."""
    return tag in _SUBJECT_TAGS or bool(_SUBJECT_RE.match(tag))


# --- the filter ------------------------------------------------------------

Verdict = namedtuple("Verdict", "raw tag keep source ref lift")


class TagVeto:
    """Judge suggestions against reference tags using tag_veto.npz."""

    def __init__(self, path=_VETO_PATH):
        import numpy as np
        self._np = np
        if path == _VETO_PATH:
            artifact.ensure(path, _VETO_URL, _VETO_SHA256, "TagVeto", "13MB")
        data = np.load(path)

        self.vocab = [str(t) for t in data["tags"]]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self._n = len(self.vocab)

        # exclusion pairs (lift < 0.5), stored as a sorted array of
        # i*n+j keys (i < j) for binary-search lookup: a dict would cost
        # ~10x the memory for 1.5M pairs.
        keys = data["pair_a"].astype(np.int64) * self._n + data["pair_b"]
        order = np.argsort(keys)
        self._pair_keys = keys[order]
        self._pair_lift = data["pair_lift"].astype(np.float32)[order]

        # composition tags: co-occurrence rows over the unfiltered corpus
        self._starved = {str(t): row for row, t in enumerate(data["starved"])}
        self._cooc_all = data["starved_cooc"]
        self._counts_all = data["counts_all"].astype(np.float64)
        self._n_all = float(data["n_all"])

        # bridge pairs (blue_eyes + red_eyes -> heterochromia): the
        # overlap is a named situation. Informational only -- gating the
        # filter on them buys 0.06pp for a second parameter.
        self.bridges = {
            frozenset((self.vocab[i], self.vocab[j])): self.vocab[t]
            for i, j, t in zip(data["bridge_a"], data["bridge_b"],
                               data["bridge_t"])
        }

    # --- pair level -------------------------------------------------------

    def pair_lift(self, a, b):
        """lift for a pair, or None when the data says nothing:
        out of vocabulary, not stored (lift >= 0.5), or below the E gate.
        None always means "no veto"."""
        if a not in self.index or b not in self.index:
            return None
        if a in self._starved or b in self._starved:
            return self._lift_unfiltered(a, b)
        return self._lift_stored(a, b)

    def _lift_stored(self, a, b):
        i, j = sorted((self.index[a], self.index[b]))
        key = i * self._n + j
        pos = int(self._np.searchsorted(self._pair_keys, key))
        if pos < len(self._pair_keys) and self._pair_keys[pos] == key:
            return float(self._pair_lift[pos])
        return None

    def _lift_unfiltered(self, a, b):
        """Lift over all posts, for pairs involving a composition tag."""
        if a not in self._starved:
            a, b = b, a
        row, j = self._starved[a], self.index[b]
        observed = float(self._cooc_all[row, j])
        expected = (self._counts_all[self.index[a]] * self._counts_all[j]
                    / self._n_all)
        if expected < E_MIN:
            return None
        return observed / expected

    def conflict(self, cand, refs, lift_th=DEFAULT_LIFT_TH):
        """First reference tag that cand contradicts, or None."""
        for ref in refs:
            if ref == cand or is_subject(ref) != is_subject(cand):
                continue
            lift = self.pair_lift(cand, ref)
            if lift is not None and lift < lift_th:
                return ref
        return None

    def bridge_for(self, a, b):
        """The tag naming this pair's situation, if the data has one."""
        return self.bridges.get(frozenset((a, b)))

    # --- prompt level -----------------------------------------------------

    def judge(self, inputs, suggestions, lift_th=DEFAULT_LIFT_TH):
        """Judge (raw, tag) suggestions against (raw, tag) inputs.

        Inputs are always kept. A surviving suggestion joins the
        references, so two mutually contradictory suggestions cannot
        both pass; suggestion order (the generator's ranking) decides
        which one wins. Returns a list of Verdict rows.
        """
        rows, seen, refs = [], set(), []

        for raw, tag in inputs:
            if tag and tag not in seen:
                seen.add(tag)
                refs.append(tag)
                rows.append(Verdict(raw, tag, True, "input", None, None))

        for raw, tag in suggestions:
            if not tag or tag in seen:
                continue
            seen.add(tag)
            ref = self.conflict(tag, refs, lift_th)
            if ref is None:
                refs.append(tag)
                rows.append(Verdict(raw, tag, True, "suggestion", None, None))
            else:
                rows.append(Verdict(raw, tag, False, "suggestion", ref,
                                    self.pair_lift(tag, ref)))
        return rows


# --- module-level API ------------------------------------------------------

_VETO = None  # lazy singleton: TagVeto, or False if the artifact is missing


def load_veto():
    global _VETO
    if _VETO is None:
        try:
            _VETO = TagVeto()
        except Exception:
            _VETO = False
    return _VETO


def veto_available():
    return bool(load_veto())


def filter_by_veto(generated_prompt, fixed_prompt="",
                   lift_th=DEFAULT_LIFT_TH):
    """Drop generated tags that contradict the fixed tags (or earlier
    surviving generated tags). Returns (filtered_prompt, report_table).
    """
    veto = load_veto()
    rows = veto.judge(_split(fixed_prompt), _split(generated_prompt), lift_th)
    kept = [r.raw for r in rows if r.keep]
    return ", ".join(kept), _format_table(rows, veto, lift_th)


def _split(prompt):
    pairs = [(raw.strip(), normalize(raw)) for raw in prompt.split(",")]
    return [(raw, tag) for raw, tag in pairs if tag]


# --- report table ----------------------------------------------------------

_TABLE_MAX_ROWS = 20


def _table_cell(row, veto):
    if not row.keep:
        return (row.tag, row.ref, "%.3f" % row.lift, "VETOED",
                veto.bridge_for(row.tag, row.ref) or "-")
    verdict = "kept" if row.tag in veto.index else "kept (OOV)"
    return (row.tag, "-", "-", verdict, "-")


def _format_table(rows, veto, lift_th):
    """One line per suggestion. Every VETOED row is shown; kept rows are
    sampled to stay under _TABLE_MAX_ROWS."""
    cells = [_table_cell(r, veto) for r in rows if r.source == "suggestion"]
    if not cells:
        return ""

    if len(cells) > _TABLE_MAX_ROWS:
        import random
        vetoed = [c for c in cells if c[3] == "VETOED"]
        kept = [c for c in cells if c[3] != "VETOED"]
        cells = vetoed + random.sample(
            kept, min(max(0, _TABLE_MAX_ROWS - len(vetoed)), len(kept)))

    header = ("tag", "vs", "lift", "verdict", "bridge")
    widths = [max(len(row[col]) for row in [header] + cells)
              for col in range(len(header))]

    def fmt(row):
        return "| %s |" % " | ".join(
            cell.rjust(w) if col == 2 else cell.ljust(w)
            for col, (cell, w) in enumerate(zip(row, widths)))

    lines = [fmt(header), "|%s|" % "|".join("-" * (w + 2) for w in widths)]
    lines += [fmt(c) for c in cells]
    lines.append("(veto: lift < %.2f, E >= %.0f; bridge = the tag naming "
                 "the overlap, informational only)" % (lift_th, E_MIN))
    return "\n".join(lines)
