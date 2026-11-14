"""Tiny BM25 implementation (no extra dependency). Helps with rare local food names (kontomire, waakye,
kokonte...) that small English embedding models do not know well."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9ɛɔ]+")
STOP = set(("a an the and or of to in is are be for with on at by it this that what which how can i my me do does should "
             "good bad okay ok eat eating best better someone people diabetes diabetic have has if still will there from "
             "much many about please you your am was were get").split())


def tokenize(text: str) -> list[str]:
    toks = [t for t in _TOK.findall(text.lower()) if t not in STOP]
    # light normalisation of UK/US spelling and plurals
    out = []
    for t in toks:
        t = t.replace("glycaemic", "glycemic").replace("haemo", "hemo").replace("okro", "okra")
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        res = []
        for d, L in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avg))
            res.append(s)
        return res
