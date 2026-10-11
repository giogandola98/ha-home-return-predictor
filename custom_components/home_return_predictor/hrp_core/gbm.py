"""In-house histogram gradient boosting for binary log-loss (numpy only at fit time).

Training needs numpy, which is imported lazily inside :meth:`HistGBM.fit`, so this module
imports fine without it. Serving is pure Python: :meth:`HistGBM.export` returns a JSON-safe
model (bin edges per feature, flat per-tree node lists, baseline logit) that
:meth:`HistGBM.predict_proba_exported` evaluates without numpy.

Algorithm
---------
* Per feature, quantile bin edges over the non-missing values (at most ``max_bins`` bins,
  edges unique). Codes: ``0`` is the dedicated *missing* bin (NaN/None), ``1..len(edges)+1``
  are value bins, where code ``c`` means ``edges[c-2] < x <= edges[c-1]``.
* Init = weighted log-odds of the base rate. Each iteration computes the log-loss gradient
  ``g = (p - y) w`` and hessian ``h = p (1 - p) w`` and grows one depth-limited tree
  level-wise. Histograms of (G, H, count) per (feature, node, bin) come from a single
  ``np.bincount`` over a combined index, so the per-row work runs inside numpy.
* A split "code <= k" (k >= 1) maps to ``x <= edges[k-1]``. Gain is
  ``G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)``; both children need ``min_samples_leaf``
  rows and hessian > 1e-6. Leaf value is ``−G/(H+λ) × learning_rate``.
* **Missing values always go left** (the missing bin is code 0, below every split point),
  both in training and in prediction.

Deterministic (ties resolved by first index); no global state other than the numpy
availability cache, so separate instances can be trained concurrently in threads.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections.abc import Sequence
from typing import Any

_FORMAT = "hrp_hist_gbm_v1"
_MIN_HESSIAN = 1e-6
_PROB_CLIP = 1e-6

_numpy_ok: bool = False  # caches only a successful import


def numpy_available() -> bool:
    """Return True if numpy can be imported (a successful import is cached)."""
    global _numpy_ok
    if _numpy_ok:
        return True
    try:
        if importlib.util.find_spec("numpy") is None:
            return False
        importlib.import_module("numpy")
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    _numpy_ok = True
    return True


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class HistGBM:
    """Histogram gradient boosting classifier for binary targets."""

    def __init__(
        self,
        *,
        max_iter: int = 80,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        max_bins: int = 64,
    ) -> None:
        if max_iter < 0 or max_depth < 0 or min_samples_leaf < 1 or max_bins < 2:
            raise ValueError("invalid HistGBM hyper-parameters")
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.max_bins = int(max_bins)
        self._model: dict[str, Any] | None = None

    # ------------------------------------------------------------------ training
    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        sample_weight: Sequence[float] | None = None,
    ) -> HistGBM:
        import numpy as np

        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array")
        n, n_feat = x.shape
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        if yv.shape[0] != n:
            raise ValueError("X and y length mismatch")
        if sample_weight is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w.shape[0] != n or np.any(w < 0) or not np.all(np.isfinite(w)):
                raise ValueError("invalid sample_weight")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("sample weights sum to zero")

        # Binning.
        edges_list: list[Any] = []
        codes = np.empty((n, n_feat), dtype=np.int64)
        qs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for f in range(n_feat):
            col = x[:, f]
            missing = np.isnan(col)
            vals = col[~missing]
            if vals.size == 0:
                edges = np.empty(0, dtype=np.float64)
            else:
                uniq = np.unique(vals)
                if uniq.size <= self.max_bins:
                    edges = (uniq[:-1] + uniq[1:]) * 0.5
                else:
                    edges = np.unique(np.quantile(vals, qs))
                edges = edges[np.isfinite(edges)]
            edges_list.append(edges)
            c = np.searchsorted(edges, col, side="left") + 1
            c[missing] = 0
            codes[:, f] = c
        n_edges = np.array([e.size for e in edges_list], dtype=np.int64)
        n_codes = int(n_edges.max()) + 2 if n_feat else 2
        # Valid split points k (code <= k goes left): 1..n_edges[f].
        kk = np.arange(n_codes)[None, :]
        valid_k = (kk >= 1) & (kk <= n_edges[:, None])  # (F, B)

        base = float((w * yv).sum()) / w_sum
        base = min(max(base, _PROB_CLIP), 1.0 - _PROB_CLIP)
        baseline = math.log(base / (1.0 - base))
        raw = np.full(n, baseline, dtype=np.float64)

        lam = self.l2_regularization
        msl = self.min_samples_leaf
        trees: list[dict[str, list[Any]]] = []
        feat_offsets_unit = np.arange(n_feat, dtype=np.int64)

        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-raw))
            g = (p - yv) * w
            h = np.maximum(p * (1.0 - p), 1e-12) * w

            # Flat node arrays; f == -1 marks a leaf. Node 0 is the root.
            t_f: list[int] = [-1]
            t_t: list[float] = [0.0]
            t_l: list[int] = [-1]
            t_r: list[int] = [-1]
            t_v: list[float] = [0.0]

            frontier = [0]
            pos = np.zeros(n, dtype=np.int64)  # frontier position, -1 = settled
            leaf_of = np.zeros(n, dtype=np.int64)
            depth = 0
            while frontier:
                ia = np.flatnonzero(pos >= 0)
                m = len(frontier)
                pa = pos[ia]
                ga = g[ia]
                ha = h[ia]
                g_tot = np.bincount(pa, weights=ga, minlength=m)
                h_tot = np.bincount(pa, weights=ha, minlength=m)
                c_tot = np.bincount(pa, minlength=m)

                feat_j = np.full(m, -1, dtype=np.int64)
                k_j = np.zeros(m, dtype=np.int64)
                left_j = np.full(m, -1, dtype=np.int64)
                right_j = np.full(m, -1, dtype=np.int64)

                if depth < self.max_depth and n_feat and bool(np.any(c_tot >= 2 * msl)):
                    idx = (
                        codes[ia]
                        + (pa * n_codes)[:, None]
                        + (feat_offsets_unit * (m * n_codes))[None, :]
                    ).ravel()
                    size = n_feat * m * n_codes
                    shape = (n_feat, m, n_codes)
                    hg = np.bincount(idx, weights=np.repeat(ga, n_feat), minlength=size).reshape(
                        shape
                    )
                    hh = np.bincount(idx, weights=np.repeat(ha, n_feat), minlength=size).reshape(
                        shape
                    )
                    hc = np.bincount(idx, minlength=size).reshape(shape)
                    gl = np.cumsum(hg, axis=2)
                    hl = np.cumsum(hh, axis=2)
                    cl = np.cumsum(hc, axis=2)
                    gr = g_tot[None, :, None] - gl
                    hr = h_tot[None, :, None] - hl
                    cr = c_tot[None, :, None] - cl
                    ok = (
                        valid_k[:, None, :]
                        & (cl >= msl)
                        & (cr >= msl)
                        & (hl > _MIN_HESSIAN)
                        & (hr > _MIN_HESSIAN)
                    )
                    parent = (g_tot * g_tot / (h_tot + lam))[None, :, None]
                    gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
                    gain = np.where(ok, gain, -np.inf)
                    # (F, m, B) -> (m, F*B); argmax picks the first best (deterministic).
                    flat = gain.transpose(1, 0, 2).reshape(m, n_feat * n_codes)
                    best = np.argmax(flat, axis=1)
                    best_gain = flat[np.arange(m), best]
                    for j in range(m):
                        if best_gain[j] > 1e-12:
                            f_best, k_best = divmod(int(best[j]), n_codes)
                            node = frontier[j]
                            t_f[node] = f_best
                            t_t[node] = float(edges_list[f_best][k_best - 1])
                            feat_j[j] = f_best
                            k_j[j] = k_best

                next_frontier: list[int] = []
                for j in range(m):
                    node = frontier[j]
                    if feat_j[j] >= 0:
                        lc = len(t_f)
                        rc = lc + 1
                        t_f += [-1, -1]
                        t_t += [0.0, 0.0]
                        t_l += [-1, -1]
                        t_r += [-1, -1]
                        t_v += [0.0, 0.0]
                        t_l[node], t_r[node] = lc, rc
                        left_j[j] = len(next_frontier)
                        next_frontier.append(lc)
                        right_j[j] = len(next_frontier)
                        next_frontier.append(rc)
                    else:
                        t_v[node] = -float(g_tot[j]) / (float(h_tot[j]) + lam) * self.learning_rate

                splitting = feat_j[pa] >= 0
                settle = ia[~splitting]
                leaf_of[settle] = np.asarray(frontier, dtype=np.int64)[pa[~splitting]]
                pos[settle] = -1
                if next_frontier:
                    rows = ia[splitting]
                    pr = pa[splitting]
                    c = codes[rows, feat_j[pr]]
                    pos[rows] = np.where(c <= k_j[pr], left_j[pr], right_j[pr])
                frontier = next_frontier
                depth += 1

            raw += np.asarray(t_v, dtype=np.float64)[leaf_of]
            trees.append({"f": t_f, "t": t_t, "l": t_l, "r": t_r, "v": t_v})

        self._model = {
            "format": _FORMAT,
            "n_features": int(n_feat),
            "baseline": float(baseline),
            "bin_edges": [[float(v) for v in e] for e in edges_list],
            "missing": "left",
            "params": {
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "l2_regularization": self.l2_regularization,
                "max_bins": self.max_bins,
            },
            "trees": trees,
        }
        return self

    # ------------------------------------------------------------------ serving
    def export(self) -> dict[str, Any]:
        """Return the JSON-safe plain-Python model (deep copy)."""
        if self._model is None:
            raise RuntimeError("HistGBM is not fitted")
        m = self._model
        return {
            **m,
            "bin_edges": [list(e) for e in m["bin_edges"]],
            "params": dict(m["params"]),
            "trees": [{k: list(v) for k, v in t.items()} for t in m["trees"]],
        }

    @staticmethod
    def predict_proba_exported(model: dict[str, Any], row: Sequence[float]) -> float:
        """Probability of the positive class for one row (pure Python, no numpy)."""
        raw = float(model["baseline"])
        for tree in model["trees"]:
            f = tree["f"]
            t = tree["t"]
            left = tree["l"]
            right = tree["r"]
            node = 0
            while f[node] >= 0:
                x = row[f[node]]
                # Missing (None/NaN) goes left.
                node = left[node] if x is None or x != x or x <= t[node] else right[node]
            raw += tree["v"][node]
        return _sigmoid(raw)

    def predict_proba(self, row: Sequence[float]) -> float:
        if self._model is None:
            raise RuntimeError("HistGBM is not fitted")
        return HistGBM.predict_proba_exported(self._model, row)
