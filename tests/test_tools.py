"""Testes das caixas de ferramentas — valores conhecidos, calculados à mão."""

import numpy as np
import pytest
import sympy as sp

from claims import main as claims_main
from mathbox import (
    R,
    absorbing_analysis,
    dice_pattern_probability,
    expected_value,
    stationary_distribution,
    sum_distribution,
    transition_matrix,
    verify_identity,
)
from montecarlo import check_mean, check_proportion, make_rng, wilson_interval


# --- mathbox ---------------------------------------------------------------

def test_verify_identity():
    assert verify_identity("(1+x)**2", "1 + 2*x + x**2")
    assert not verify_identity("(1+x)**2", "1 + x**2")


def test_sum_distribution_two_dice():
    d = sum_distribution(2)
    assert d[7] == R(6, 36)
    assert d[2] == R(1, 36)
    assert sum(d.values()) == 1
    assert expected_value(d) == 7


def test_dice_pattern_yahtzee():
    # cinco dados iguais numa jogada: 6 / 6^5
    p = dice_pattern_probability(5, lambda r: len(set(r)) == 1)
    assert p == R(1, 1296)


def test_stationary_two_state():
    P = transition_matrix(["A", "B"], {"A": {"A": R(1, 2), "B": R(1, 2)}, "B": {"A": R(1, 4), "B": R(3, 4)}})
    pi = stationary_distribution(P)
    assert list(pi) == [R(1, 3), R(2, 3)]


def test_absorbing_gamblers_ruin():
    # ruína do jogador com moeda justa, capital 0..3, começando em 1
    states = [0, 1, 2, 3]
    P = transition_matrix(states, {1: {0: R(1, 2), 2: R(1, 2)}, 2: {1: R(1, 2), 3: R(1, 2)}})
    a = absorbing_analysis(P, states)
    assert a["transient"] == [1, 2] and a["absorbing"] == [0, 3]
    assert a["B"][0, 1] == R(1, 3)  # de 1, chega a 3 com prob 1/3
    assert a["t"][0] == 2           # passos esperados: k(N-k) = 1*2


def test_transition_matrix_rejects_bad_row():
    with pytest.raises(ValueError):
        transition_matrix(["A"], {"A": {"A": R(1, 2)}})


# --- montecarlo ------------------------------------------------------------

def test_wilson_contains_truth():
    lo, hi = wilson_interval(500, 1000)
    assert lo < 0.5 < hi


def test_check_proportion_agrees_and_diverges():
    rng = make_rng()
    hits = rng.random(200_000) < 0.3
    assert check_proportion(hits, expected=0.3).status == "validada"
    assert check_proportion(hits, expected=0.31).status == "divergente"


def test_check_mean_dice():
    rng = make_rng()
    rolls = rng.integers(1, 7, size=(200_000, 2)).sum(axis=1)
    assert check_mean(rolls, expected=float(sp.Integer(7))).status == "validada"


# --- claims ----------------------------------------------------------------

def test_claims_lifecycle(tmp_path, capsys):
    post = tmp_path / "post"
    post.mkdir()
    assert claims_main(["init", str(post)]) == 0
    f = post / "_work" / "claims.yaml"
    f.write_text(
        "pergunta: teste\nclaims:\n  - id: C1\n    enunciado: x\n    status: derivada\n",
        encoding="utf-8",
    )
    assert claims_main(["gate", str(post)]) == 1
    assert claims_main(["set", str(post), "C1", "validada", "--nota", "ok"]) == 0
    assert claims_main(["gate", str(post)]) == 0
