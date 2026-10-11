"""Tests for the in-house histogram gradient boosting (hrp_core.gbm)."""

from __future__ import annotations

import importlib
import json
import math
import sys
import time

import numpy as np
import pytest

from hrp_core import gbm as gbm_module
from hrp_core.gbm import HistGBM, numpy_available

PARAMS = {
    "max_iter": 80,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 64,
}


def _xor_data(n: int, n_feat: int = 11, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, n_feat))
    logit = 2.5 * np.sign(x[:, 0] * x[:, 1]) + 1.5 * (x[:, 2] > 0.3) - 0.5
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return x, y


def _smooth_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Non-linear but with marginal signal, so greedy trees are not path-dependent."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 11))
    logit = 2 * np.sin(3 * x[:, 0]) + 2 * x[:, 1] * (x[:, 2] > 0) + 1.5 * x[:, 3] ** 2 - 0.5
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return x, y


def _predict(model: HistGBM, x: np.ndarray) -> np.ndarray:
    return np.array([model.predict_proba(list(r)) for r in x])


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logistic_probs(x_tr, y_tr, x_te) -> np.ndarray:
    """Plain L2 logistic regression via Newton steps (baseline)."""
    a = np.hstack([np.ones((len(x_tr), 1)), x_tr])
    b = np.hstack([np.ones((len(x_te), 1)), x_te])
    beta = np.zeros(a.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-a @ beta))
        grad = a.T @ (p - y_tr) + 1e-3 * beta
        hess = (a * (p * (1 - p))[:, None]).T @ a + 1e-3 * np.eye(a.shape[1])
        beta -= np.linalg.solve(hess, grad)
    return 1.0 / (1.0 + np.exp(-b @ beta))


@pytest.fixture(scope="module")
def data():
    x, y = _xor_data(4000)
    return x[:3000], y[:3000], x[3000:], y[3000:]


@pytest.fixture(scope="module")
def fitted(data):
    x_tr, y_tr, _, _ = data
    return HistGBM(**PARAMS).fit(x_tr.tolist(), y_tr.tolist())


def test_beats_logistic_on_nonlinear(data, fitted):
    x_tr, y_tr, x_te, y_te = data
    b_gbm = _brier(_predict(fitted, x_te), y_te)
    b_log = _brier(_logistic_probs(x_tr, y_tr, x_te), y_te)
    print(f"brier gbm={b_gbm:.4f} logistic={b_log:.4f}")
    assert b_gbm < b_log - 0.03
    xs, ys = _smooth_data(4000, 1)
    smooth = HistGBM(**PARAMS).fit(xs[:3000], ys[:3000])
    b_gbm = _brier(_predict(smooth, xs[3000:]), ys[3000:])
    b_log = _brier(_logistic_probs(xs[:3000], ys[:3000], xs[3000:]), ys[3000:])
    print(f"smooth brier gbm={b_gbm:.4f} logistic={b_log:.4f}")
    assert b_gbm < b_log - 0.01


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_close_to_sklearn(seed):
    # Pure XOR is too path-dependent at 80 trees (first split has ~zero gain) for a
    # tight comparison, so use a smooth non-linear target here.
    ensemble = pytest.importorskip("sklearn.ensemble")
    x, y = _smooth_data(4000, seed)
    x_tr, y_tr, x_te, y_te = x[:3000], y[:3000], x[3000:], y[3000:]
    fitted = HistGBM(**PARAMS).fit(x_tr, y_tr)
    sk = ensemble.HistGradientBoostingClassifier(
        **PARAMS, early_stopping=False, random_state=0
    ).fit(x_tr, y_tr)
    b_sk = _brier(sk.predict_proba(x_te)[:, 1], y_te)
    b_gbm = _brier(_predict(fitted, x_te), y_te)
    print(f"brier gbm={b_gbm:.4f} sklearn={b_sk:.4f}")
    assert abs(b_gbm - b_sk) / b_sk < 0.10


def test_export_matches_and_json_round_trip(data, fitted):
    _, _, x_te, _ = data
    exported = fitted.export()
    restored = json.loads(json.dumps(exported, allow_nan=False))
    assert restored == exported
    for row in x_te[:200].tolist():
        p = fitted.predict_proba(row)
        assert p == HistGBM.predict_proba_exported(exported, row)
        assert p == HistGBM.predict_proba_exported(restored, row)
        assert 0.0 < p < 1.0
    assert len(exported["trees"]) == PARAMS["max_iter"]
    assert all(len(e) <= PARAMS["max_bins"] for e in exported["bin_edges"])


def test_deterministic(data, fitted):
    x_tr, y_tr, _, _ = data
    again = HistGBM(**PARAMS).fit(x_tr.tolist(), y_tr.tolist())
    assert again.export() == fitted.export()


def test_sample_weights(data):
    x_tr, y_tr, x_te, _ = data
    unweighted = HistGBM(**PARAMS).fit(x_tr, y_tr)
    ones = HistGBM(**PARAMS).fit(x_tr, y_tr, sample_weight=[1.0] * len(y_tr))
    assert ones.export() == unweighted.export()
    w = np.where(y_tr == 1, 5.0, 1.0)
    weighted = HistGBM(**PARAMS).fit(x_tr, y_tr, sample_weight=w)
    p_u = _predict(unweighted, x_te).mean()
    p_w = _predict(weighted, x_te).mean()
    assert p_w > p_u + 0.1  # upweighting positives raises predicted probabilities
    # Zero weight on one region removes its influence on that region's predictions.
    region = x_tr[:, 0] > 0
    y_flip = np.where(region, 1 - y_tr, y_tr)
    w_mask = np.where(region, 0.0, 1.0)
    masked = HistGBM(**PARAMS).fit(x_tr, y_flip, sample_weight=w_mask)
    clean = HistGBM(**PARAMS).fit(x_tr[~region], y_tr[~region])
    te = x_te[x_te[:, 0] <= 0]
    diff = np.abs(_predict(masked, te) - _predict(clean, te)).mean()
    assert diff < 0.05


def test_missing_values_go_left():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(1000, 2))
    y = (x[:, 0] > 0.5).astype(int)
    x[::7, 0] = np.nan
    model = HistGBM(**PARAMS).fit(x, y)
    exported = model.export()
    assert exported["missing"] == "left"
    p_nan = model.predict_proba([math.nan, 0.5])
    assert p_nan == model.predict_proba([None, 0.5])
    assert p_nan == model.predict_proba([-1.0, 0.5])  # same path as "small"
    assert model.predict_proba([0.9, 0.5]) > 0.8


def test_constant_target_and_feature():
    model = HistGBM(**PARAMS).fit([[1.0, 2.0]] * 50, [1] * 50)
    assert model.predict_proba([1.0, 2.0]) > 0.99


def test_fit_timing():
    x, y = _xor_data(3000)
    best = math.inf
    for _ in range(2):
        start = time.perf_counter()
        HistGBM(**PARAMS).fit(x, y)
        best = min(best, time.perf_counter() - start)
    print(f"fit 3000x11: {best:.3f}s")
    assert best < 1.0


@pytest.fixture
def reload_gbm():
    yield
    importlib.reload(gbm_module)


def test_import_without_numpy(monkeypatch, fitted, reload_gbm):
    exported = fitted.export()
    row = [0.2] * 11
    expected = fitted.predict_proba(row)
    monkeypatch.setitem(sys.modules, "numpy", None)
    mod = importlib.reload(gbm_module)
    assert mod.numpy_available() is False
    assert mod.HistGBM.predict_proba_exported(exported, row) == expected
    with pytest.raises(ImportError):
        mod.HistGBM().fit([[0.0]], [1])


def test_numpy_available():
    assert numpy_available() is True


def test_validate_exported_rejects_corrupt_payloads() -> None:
    import random

    rng = random.Random(3)
    X = [[rng.random(), rng.random()] for _ in range(400)]
    y = [int(a + b > 1.0) for a, b in X]
    model = HistGBM(max_iter=10).fit(X, y).export()
    assert HistGBM.validate_exported(model, 2)

    cyclic = json.loads(json.dumps(model))
    tree = next(t for t in cyclic["trees"] if t["f"][0] >= 0)
    tree["l"][0] = 0  # child pointing back to its parent -> endless traversal
    assert not HistGBM.validate_exported(cyclic, 2)

    out_of_range = json.loads(json.dumps(model))
    tree = next(t for t in out_of_range["trees"] if t["f"][0] >= 0)
    tree["f"][0] = 7
    assert not HistGBM.validate_exported(out_of_range, 2)

    assert not HistGBM.validate_exported({"trees": []}, 2)
