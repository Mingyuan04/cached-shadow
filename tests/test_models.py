"""Properties each model must have. Run with `python -m pytest -q`."""
import numpy as np
import pytest

from src import models as M
from src.prep import normalized_params


def env(o1, L, delta=0.75, R=40):
    g, l = normalized_params(R, 50, 12, 25)
    o1, L = np.atleast_2d(np.asarray(o1, float)), np.atleast_2d(np.asarray(L, float))
    N = o1.shape[0]
    return {"o1": o1, "L": L, "ET": np.full(N, 1 / (1 - delta)),
            "g": np.full(N, float(g)), "l": np.full(N, float(l))}


def test_delta_U_corner_cases():
    assert M.delta_U(1.0, 4.0, 0.5, 0.8) == pytest.approx(4.0 - 1.0 - 0.5)   # partner surely Grim
    assert M.delta_U(0.0, 4.0, 0.5, 0.8) == pytest.approx(-0.8)              # partner surely AD


def test_planner_ignores_realized_length():
    o1 = [1, 0, 1, 1, 0, 1]
    th = [1.5, -0.3, 0.3, 0.5]
    assert np.allclose(M.planner_probs(th, env(o1, [1] * 6)), M.planner_probs(th, env(o1, [9] * 6)))


def test_planner_learns_from_partners():
    p = M.planner_probs([1.5, 0.0, 0.3, 0.5], env([1] * 8, [4] * 8))
    assert np.all(np.diff(p[0]) > 0)


def test_value_cache_flat_when_lam_is_zero():
    p = M.value_cache_probs([0.4, 0.0, 0.5], env([1, 0, 1, 0], [3, 5, 2, 7]))
    assert np.allclose(p, M.sigmoid(0.4))


def test_value_cache_length_irrelevant_after_defectors():
    th = [0.0, 0.3, 1.0]
    short = M.value_cache_probs(th, env([0] * 6, [1] * 6))
    long_ = M.value_cache_probs(th, env([0] * 6, [9] * 6))
    assert np.allclose(short, long_)


def test_value_cache_length_matters_after_cooperators():
    th = [0.0, 0.3, 1.0]
    short = M.value_cache_probs(th, env([1] * 6, [1] * 6))
    long_ = M.value_cache_probs(th, env([1] * 6, [9] * 6))
    assert np.all(long_[0, 1:] > short[0, 1:])


def test_value_cache_alpha_zero_uses_expected_length_only():
    th = [0.0, 0.3, 0.0]
    a = M.value_cache_probs(th, env([1, 0, 1, 1], [1, 1, 1, 1]))
    b = M.value_cache_probs(th, env([1, 0, 1, 1], [9, 9, 9, 9]))
    assert np.allclose(a, b)


def test_prediction_cache_eta_zero_equals_planner():
    d = env([[1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]], [[3, 1, 7, 2, 2, 5], [1, 9, 2, 4, 1, 3]])
    th = [1.2, -0.2, 0.3, 0.6]
    assert np.allclose(M.prediction_cache_probs(th + [0.0], d), M.planner_probs(th, d))


def test_prediction_cache_length_matters_even_after_defectors():
    th = [1.5, 0.0, 0.2, 0.6, 0.3]   # p stays above 0, so H still counts
    short = M.prediction_cache_probs(th, env([0] * 6, [1] * 6))
    long_ = M.prediction_cache_probs(th, env([0] * 6, [9] * 6))
    assert np.all(long_[0, 1:] > short[0, 1:])


def test_prediction_cache_does_not_modify_data():
    d = env([1, 0, 1], [5, 9, 7])
    before = d["ET"].copy()
    M.prediction_cache_probs([1.0, 0.0, 0.2, 0.5, 0.5], d)
    assert np.array_equal(d["ET"], before)


@pytest.mark.parametrize("name", list(M.MODELS))
def test_vectorized_over_subjects(name):
    fn, bounds = M.MODELS[name]
    p = fn([np.mean(b) for b in bounds.values()], env([[1, 0, 1], [0, 1, 1]], [[2, 5, 1], [7, 1, 3]]))
    assert p.shape == (2, 3) and np.all((p > 0) & (p < 1))
