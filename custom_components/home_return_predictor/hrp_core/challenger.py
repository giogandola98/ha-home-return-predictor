"""Optional scikit-learn challenger (HistGradientBoostingClassifier).

The module is importable without scikit-learn: sklearn / numpy / threadpoolctl are
imported lazily inside the functions that need them. The fitted estimator is never
pickled or persisted; it is retrained from the persisted prediction log on demand.

After fitting, the trees are compiled into plain Python tuples so that
``predict_proba`` of a single row is fast (~0.1 ms), thread-free and does not
touch numpy; if the private tree layout of the installed sklearn is not
recognised, prediction falls back to the estimator's own ``predict_proba``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Any

from .fusion import P_MAX, P_MIN, PRIOR_CLAMP, clamp, logit, sigmoid

CHALLENGER_FEATURES: tuple[str, ...] = (
    "z_prior",
    "ln_lr",
    "away_horizons",
    "dep_hour_sin",
    "dep_hour_cos",
    "hour_sin",
    "hour_cos",
    "rest_day",
    "log1p_distance_km",
    "speed_kmh",
    "has_gps",
    "z_bayes",
)
N_CHALLENGER_FEATURES = len(CHALLENGER_FEATURES)
MAX_AWAY_HORIZONS = 16.0
MAX_ITER = 100
MAX_LEAF_NODES = 15
LEARNING_RATE = 0.1
L2_REGULARIZATION = 1.0
# Features are snapped to <= MAX_DISTINCT unweighted-quantile edges before fitting:
# HistGradientBoosting then needs no weighted percentiles to bin them (with sample
# weights those cost ~0.3 ms per call and made a 3000-row fit take ~1 s instead of
# ~0.15 s). The same snapping is applied at prediction time.
MAX_DISTINCT = 255

# Compiled tree node: (feature, threshold, left, right, is_leaf, value)
_Node = tuple[int, float, int, int, bool, float]


_AVAILABLE = False  # only a positive result is cached (sklearn may be installed later)


def sklearn_available() -> bool:
    """True if scikit-learn (with numpy and threadpoolctl) can be imported."""
    global _AVAILABLE
    if _AVAILABLE:
        return True
    try:
        import numpy  # noqa: F401
        import sklearn.ensemble  # noqa: F401
        import threadpoolctl  # noqa: F401
    except Exception:  # ImportError, or a broken binary install
        return False
    _AVAILABLE = True
    return True


def _hour_angle(hour: float) -> float:
    return 2.0 * math.pi * hour / 24.0


def build_challenger_features(
    *,
    prior: float,
    likelihood_ratio: float,
    minutes_away: float,
    horizon_minutes: float,
    departure_hour: float | None,
    local_hour: float,
    rest_day: bool,
    distance_m: float | None,
    speed_kmh: float | None,
    has_gps: bool,
    p_bayes: float,
) -> list[float]:
    """Feature vector of the challenger (see CHALLENGER_FEATURES)."""
    dep = _hour_angle(departure_hour) if departure_hour is not None else None
    now = _hour_angle(local_hour)
    return [
        logit(clamp(prior, PRIOR_CLAMP, 1 - PRIOR_CLAMP)),
        math.log(max(likelihood_ratio, 1e-9)),
        min(max(minutes_away, 0.0) / max(horizon_minutes, 1.0), MAX_AWAY_HORIZONS),
        math.sin(dep) if dep is not None else 0.0,
        math.cos(dep) if dep is not None else 0.0,
        math.sin(now),
        math.cos(now),
        1.0 if rest_day else 0.0,
        math.log1p(distance_m / 1000.0) if distance_m is not None else 0.0,
        float(speed_kmh) if speed_kmh is not None else 0.0,
        1.0 if has_gps else 0.0,
        logit(clamp(p_bayes, P_MIN, P_MAX)),
    ]


def bayes_from_challenger_features(row: Sequence[float]) -> float:
    """The kernel-Bayes probability stored in a challenger feature row."""
    return clamp(sigmoid(row[-1]), P_MIN, P_MAX)


class SklearnChallenger:
    """Gradient-boosted trees on the prediction-log features (not persisted)."""

    def __init__(self) -> None:
        self._estimator: Any = None
        self._trees: list[list[_Node]] | None = None
        self._baseline = 0.0
        self._grids: list[list[float] | None] = []
        self.trained_on = 0

    @property
    def trained(self) -> bool:
        return self._estimator is not None or self._trees is not None

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        weights: Sequence[float] | None = None,
    ) -> None:
        """Fit single-threaded. Raises RuntimeError if sklearn is missing."""
        if not sklearn_available():
            raise RuntimeError("scikit-learn is not available")
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits

        x = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=int)
        self._grids = []
        for j in range(x.shape[1] if x.ndim == 2 else 0):
            col = x[:, j]
            if len(np.unique(col)) <= MAX_DISTINCT:
                self._grids.append(None)
                continue
            edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, MAX_DISTINCT)))
            self._grids.append([float(e) for e in edges])
            idx = np.minimum(np.searchsorted(edges, col, side="left"), len(edges) - 1)
            x[:, j] = edges[idx]
        if len(set(y.tolist())) < 2:
            # Degenerate log (all returns or none): a constant predictor.
            rate = float(np.clip(y.mean() if len(y) else 0.5, P_MIN, P_MAX))
            self._estimator, self._trees = None, []
            self._baseline = logit(rate)
            self.trained_on = len(y)
            return
        sw = None if weights is None else np.asarray(weights, dtype=float)
        est = HistGradientBoostingClassifier(
            max_iter=MAX_ITER,
            max_leaf_nodes=MAX_LEAF_NODES,
            learning_rate=LEARNING_RATE,
            l2_regularization=L2_REGULARIZATION,
            early_stopping=False,
            random_state=0,
        )
        with threadpool_limits(limits=1):
            est.fit(x, y, sample_weight=sw)
        self._estimator = est
        self._trees, self._baseline = self._compile(est)
        self.trained_on = len(y)

    @staticmethod
    def _compile(est: Any) -> tuple[list[list[_Node]] | None, float]:
        """Extract the trees as plain tuples (None if the layout is unknown)."""
        try:
            baseline = float(est._baseline_prediction.ravel()[0])
            trees: list[list[_Node]] = []
            for (predictor,) in est._predictors:
                nodes = predictor.nodes
                if bool(nodes["is_categorical"].any()):
                    return None, 0.0
                trees.append(
                    [
                        (
                            int(n["feature_idx"]),
                            float(n["num_threshold"]),
                            int(n["left"]),
                            int(n["right"]),
                            bool(n["is_leaf"]),
                            float(n["value"]),
                        )
                        for n in nodes
                    ]
                )
        except Exception:
            return None, 0.0
        return trees, baseline

    def _snap(self, row: Sequence[float]) -> list[float]:
        out = list(row)
        for j, grid in enumerate(self._grids):
            if grid is not None:
                out[j] = grid[min(bisect.bisect_left(grid, out[j]), len(grid) - 1)]
        return out

    def predict_raw(self, row: Sequence[float]) -> float:
        """Decision function (log-odds) of one row from the compiled trees."""
        assert self._trees is not None
        row = self._snap(row)
        z = self._baseline
        for tree in self._trees:
            i = 0
            node = tree[0]
            while not node[4]:
                i = node[2] if row[node[0]] <= node[1] else node[3]
                node = tree[i]
            z += node[5]
        return z

    def predict_proba(self, row: Sequence[float]) -> float:
        """P(return within horizon) for one feature row, clamped to [0.01, 0.99]."""
        if self._trees is not None:
            return clamp(sigmoid(self.predict_raw(row)), P_MIN, P_MAX)
        if self._estimator is None:
            raise RuntimeError("challenger not trained")
        import numpy as np

        snapped = np.asarray([self._snap(row)], dtype=float)
        p = float(self._estimator.predict_proba(snapped)[0, 1])
        return clamp(p, P_MIN, P_MAX)

    def predict_many(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_proba(r) for r in rows]
