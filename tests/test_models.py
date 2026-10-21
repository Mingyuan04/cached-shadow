"""Each test is a theoretical claim about a model. Tests for a model you have
not written yet are skipped; once you fill in the TODO they must all pass.

    python -m pytest -q
"""
import numpy as np
import pytest

from src import models as M


def env(o1, L, delta=0.75, R=40):
    from src.prep import normalized_params
    g, l = normalized_params(R, 50, 12, 25)
    o1, L = np.atleast_2d(np.asarray(o1, float)), np.atleast_2d(np.asarray(L, float))
    N = o1.shape[0]
    return {"o1": o1, "L": L, "ET": np.full(N, 1 / (1 - delta)),
            "g": np.full(N, float(g)), "l": np.full(N, float(l))}


def call(fn, theta, d):
    try:
        return fn(theta, d)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


# --- shared value function -------------------------------------------------
def test_delta_U_corner_cases():
    assert M.delta_U(1.0, 4.0, 0.5, 0.8) == pytest.approx(4.0 - 1.0 - 0.5)   # partner surely Grim
    assert M.delta_U(0.0, 4.0, 0.5, 0.8) == pytest.approx(-0.8)              # partner surely AD


# --- planner (worked example) ------------------------------------------------
def test_planner_ignores_realized_length():
    o1 = [1, 0, 1, 1, 0, 1]
    th = [1.5, -0.3, 0.3, 0.5]
    short = call(M.planner_probs, th, env(o1, [1, 1, 1, 1, 1, 1]))
    long_ = call(M.planner_probs, th, env(o1, [9, 9, 9, 9, 9, 9]))
    assert np.allclose(short, long_)


def test_planner_learns_from_partners():
    th = [1.5, 0.0, 0.3, 0.5]
    p = call(M.planner_probs, th, env([1] * 8, [4] * 8))
    assert np.all(np.diff(p[0]) > 0)          # cooperative partners -> more C


# --- value cache ---------------------------------------------------------------
def test_value_cache_flat_when_lam_is_zero():
    p = call(M.value_cache_probs, [0.4, 0.0, 0.5], env([1, 0, 1, 0], [3, 5, 2, 7]))
    assert np.allclose(p, M.sigmoid(0.4))


def test_value_cache_length_irrelevant_after_defectors():
    th = [0.0, 0.3, 1.0]                       # alpha = 1: all weight on realized length
    short = call(M.value_cache_probs, th, env([0] * 6, [1] * 6))
    long_ = call(M.value_cache_probs, th, env([0] * 6, [9] * 6))
    assert np.allclose(short, long_)


def test_value_cache_length_matters_after_cooperators():
    th = [0.0, 0.3, 1.0]
    short = call(M.value_cache_probs, th, env([1] * 6, [1] * 6))
    long_ = call(M.value_cache_probs, th, env([1] * 6, [9] * 6))
    assert np.all(long_[0, 1:] > short[0, 1:])


def test_value_cache_alpha_zero_uses_expected_length_only():
    th = [0.0, 0.3, 0.0]
    a = call(M.value_cache_probs, th, env([1, 0, 1, 1], [1, 1, 1, 1]))
    b = call(M.value_cache_probs, th, env([1, 0, 1, 1], [9, 9, 9, 9]))
    assert np.allclose(a, b)


# --- prediction cache ------------------------------------------------------------
def test_prediction_cache_eta_zero_equals_planner():
    d = env([[1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1]], [[3, 1, 7, 2, 2, 5], [1, 9, 2, 4, 1, 3]])
    th = [1.2, -0.2, 0.3, 0.6]
    pc = call(M.prediction_cache_probs, th + [0.0], d)
    assert np.allclose(pc, M.planner_probs(th, d))


def test_prediction_cache_length_matters_even_after_defectors():
    th = [1.5, 0.0, 0.2, 0.6, 0.3]             # p stays above 0, so H still counts
    short = call(M.prediction_cache_probs, th, env([0] * 6, [1] * 6))
    long_ = call(M.prediction_cache_probs, th, env([0] * 6, [9] * 6))
    assert np.all(long_[0, 1:] > short[0, 1:])


def test_vectorized_over_subjects():
    d = env([[1, 0, 1], [0, 1, 1]], [[2, 5, 1], [7, 1, 3]])
    for name, (fn, bounds) in M.MODELS.items():
        th = [np.mean(b) for b in bounds.values()]
        p = call(fn, th, d)
        assert p.shape == (2, 3) and np.all((p > 0) & (p < 1)), name
