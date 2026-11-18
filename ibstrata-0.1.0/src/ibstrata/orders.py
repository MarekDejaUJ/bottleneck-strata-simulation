"""Person-level statistics of anchored tap-order block tasks.

In a tap-order block the respondent taps the cards from first to last; an anchor card in
every block marks a threshold. For one respondent and one task:

* the anchor share of an item is the share of its blocks in which the item was tapped
  before the anchor; with an odd number of anchor comparisons the majority never ties;
* the precedence share of an ordered pair (a, b) is the share of the blocks holding both
  in which a was tapped before b;
* the Copeland score of an item is the number of other items it precedes by majority
  minus the number that precede it; the person's order is the order of the scores;
* consistency is the share of pair observations that agree with the pair's majority;
* cyclic triads count the intransitive triples of the majority tournament.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import SurveyExport


@dataclass
class TapTask:
    """The block responses of one tap-order task."""

    key: str
    items: list[str]
    anchor: str | None
    rows: pd.DataFrame  # response_uuid, position, cards, order (list), gate, attention

    @property
    def respondents(self) -> list[str]:
        return sorted(self.rows["response_uuid"].unique())

    def _orders(self):
        for rid, grp in self.rows.groupby("response_uuid"):
            yield rid, [o for o, att in zip(grp["order"], grp["attention"]) if not att]

    def anchor_share(self) -> pd.DataFrame:
        """Share of an item's blocks in which the item precedes the anchor (respondents x items)."""
        if self.anchor is None:
            raise ValueError("the task has no anchor card")
        out = {}
        for rid, orders in self._orders():
            above = dict.fromkeys(self.items, 0)
            seen = dict.fromkeys(self.items, 0)
            for order in orders:
                pos = {c: i for i, c in enumerate(order)}
                for c in order:
                    if c in seen:
                        seen[c] += 1
                        above[c] += int(pos[c] < pos[self.anchor])
            out[rid] = {c: (above[c] / seen[c] if seen[c] else np.nan) for c in self.items}
        return pd.DataFrame.from_dict(out, orient="index")[self.items]

    def anchor_majority(self) -> pd.DataFrame:
        """+1 when an item precedes the anchor in most of its blocks, -1 when it follows it,
        0 on a tie."""
        return np.sign(self.anchor_share() - 0.5).astype(float)

    def precedence(self) -> pd.DataFrame:
        """Share of joint blocks in which the first item of each pair precedes the second
        (respondents x pairs in the order of ``itertools.combinations(items, 2)``)."""
        pairs = list(itertools.combinations(self.items, 2))
        out = {}
        for rid, orders in self._orders():
            wins = dict.fromkeys(pairs, 0)
            meets = dict.fromkeys(pairs, 0)
            for order in orders:
                pos = {c: i for i, c in enumerate(order)}
                for a, b in pairs:
                    if a in pos and b in pos:
                        meets[(a, b)] += 1
                        wins[(a, b)] += int(pos[a] < pos[b])
            out[rid] = {f"{a}>{b}": (wins[(a, b)] / meets[(a, b)] if meets[(a, b)] else np.nan)
                        for a, b in pairs}
        return pd.DataFrame.from_dict(out, orient="index")

    def copeland(self) -> pd.DataFrame:
        """Copeland scores from the pairwise majorities (respondents x items)."""
        prec = self.precedence()
        scores = pd.DataFrame(0.0, index=prec.index, columns=self.items)
        for col in prec.columns:
            a, b = col.split(">")
            s = np.sign(prec[col] - 0.5).fillna(0.0)
            scores[a] += s
            scores[b] -= s
        return scores

    def consistency(self) -> pd.Series:
        """Share of pair observations that agree with the respondent's pair majority."""
        pairs = list(itertools.combinations(self.items, 2))
        out = {}
        for rid, orders in self._orders():
            obs: dict[tuple[str, str], list[int]] = {p: [] for p in pairs}
            for order in orders:
                pos = {c: i for i, c in enumerate(order)}
                for a, b in pairs:
                    if a in pos and b in pos:
                        obs[(a, b)].append(int(pos[a] < pos[b]))
            agree = total = 0
            for v in obs.values():
                if len(v) > 1:
                    maj = int(np.mean(v) > 0.5) if np.mean(v) != 0.5 else None
                    total += len(v)
                    agree += sum(int(x == maj) for x in v) if maj is not None else len(v) / 2
            out[rid] = agree / total if total else np.nan
        return pd.Series(out, name=f"{self.key}_consistency")

    def cyclic_triads(self) -> pd.Series:
        """Number of intransitive triples in the majority tournament."""
        prec = self.precedence()
        out = {}
        for rid, row in prec.iterrows():
            beats = {}
            for col, v in row.items():
                a, b = col.split(">")
                if v > 0.5:
                    beats[(a, b)] = True
                elif v < 0.5:
                    beats[(b, a)] = True
            cyc = 0
            for x, y, z in itertools.combinations(self.items, 3):
                if ((beats.get((x, y)) and beats.get((y, z)) and beats.get((z, x)))
                        or (beats.get((y, x)) and beats.get((z, y)) and beats.get((x, z)))):
                    cyc += 1
            out[rid] = cyc
        return pd.Series(out, name=f"{self.key}_cyclic_triads")

    def tau_b(self, reference: list[list[str]]) -> pd.Series:
        """Kendall's tau-b between each respondent's Copeland order and a reference order
        given as tiers (first tier best)."""
        tier = {c: -t for t, codes in enumerate(reference) for c in codes}
        ref = np.array([tier[c] for c in self.items], dtype=float)
        scores = self.copeland()
        values = {rid: kendall_tau_b(row.to_numpy(float), ref) for rid, row in scores.iterrows()}
        return pd.Series(values, name=f"{self.key}_tau_b")

    def gate_calibration(self, reference: list[list[str]], sure: str = "yes") -> pd.DataFrame:
        """Per respondent: block accuracy (share of block pairs ordered as in the reference)
        in blocks marked sure and unsure, and their difference."""
        tier = {c: t for t, codes in enumerate(reference) for c in codes}
        out = {}
        for rid, grp in self.rows.groupby("response_uuid"):
            acc = {True: [], False: []}
            for order, gate, att in zip(grp["order"], grp["gate"], grp["attention"]):
                if att:
                    continue
                cards = [c for c in order if c in tier]
                good = total = 0
                for i, j in itertools.combinations(range(len(cards)), 2):
                    if tier[cards[i]] != tier[cards[j]]:
                        total += 1
                        good += int(tier[cards[i]] < tier[cards[j]])
                if total:
                    acc[gate == sure].append(good / total)
            s = np.mean(acc[True]) if acc[True] else np.nan
            u = np.mean(acc[False]) if acc[False] else np.nan
            out[rid] = {"accuracy_sure": s, "accuracy_unsure": u, "calibration": s - u,
                        "share_sure": len(acc[True]) / max(1, len(acc[True]) + len(acc[False]))}
        return pd.DataFrame.from_dict(out, orient="index")

    def features(self) -> pd.DataFrame:
        """Anchor shares (when the task has an anchor) and precedence shares, prefixed with
        the task key."""
        parts = []
        if self.anchor is not None:
            parts.append(self.anchor_share().add_prefix(f"{self.key}:anchor:"))
        parts.append(self.precedence().add_prefix(f"{self.key}:prec:"))
        return pd.concat(parts, axis=1)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b of two score vectors (higher is better in both), with ties."""
    n = len(x)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / denom) if denom > 0 else 0.0


def tap_task(export: SurveyExport, key: str) -> TapTask:
    """The tap-order task ``key`` of an export, with the items and the anchor of its bundle."""
    module = export.module(key)
    items = [c["code"] for c in module.get("items", [])]
    anchor = module.get("anchor", {}).get("code") if module.get("anchor") else None
    b = export.blocks
    rows = b[b["module_key"] == key].copy()
    rows["order"] = rows["result"].map(json.loads)
    rows["position"] = rows["position"].astype(int)
    rows["attention"] = rows["attention"].isin(["1", "True", "true"])
    rows["gate"] = rows.get("gate_value", "")
    rows = rows.sort_values(["response_uuid", "position"])
    return TapTask(key, items, anchor, rows[["response_uuid", "position", "cards", "order", "gate",
                                             "attention"]])
