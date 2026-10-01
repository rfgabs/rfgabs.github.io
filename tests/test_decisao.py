"""Testes de MDP, RL e teoria dos jogos sobre jogos com resposta conhecida."""

import numpy as np
import pytest
import sympy as sp

from mdp import MDP, policy_evaluation, policy_evaluation_exact, value_iteration
from montecarlo import check_mean, make_rng
from rl import greedy_policy, mc_control, q_learning, rollout_returns

FIM = "fim"
TETO = 40


def pig_um_turno() -> MDP:
    """Pig de um turno: rola um d6 e acumula; tirar 1 zera tudo. Parar embolsa o total.

    Resultado clássico: a política ótima rola enquanto o total < 20.
    """

    def actions(s):
        if s == FIM:
            return []
        return ["parar"] if s >= TETO else ["parar", "rolar"]

    def transitions(s, a):
        if a == "parar":
            return [(1, FIM, s)]
        sexto = sp.Rational(1, 6)
        return [(sexto, FIM, 0)] + [(sexto, s + k, 0) for k in range(2, 7)]

    return MDP(initial=0, actions=actions, transitions=transitions)


@pytest.fixture(scope="module")
def jogo():
    return pig_um_turno()


@pytest.fixture(scope="module")
def otimo(jogo):
    return value_iteration(jogo)


def test_politica_otima_limiar_20(jogo, otimo):
    V, pol = otimo
    assert all(pol[s] == "rolar" for s in range(0, 20) if s in pol)
    assert all(pol[s] == "parar" for s in range(20, TETO + 6) if s in pol)


def test_avaliacao_exata_bate_com_numerica(jogo, otimo):
    V, pol = otimo
    exato = policy_evaluation_exact(jogo, pol, state=0)
    assert isinstance(exato, sp.Rational)
    assert float(exato) == pytest.approx(V[0], abs=1e-9)
    assert float(exato) == pytest.approx(8.14, abs=0.01)  # valor clássico de "parar aos 20"


def test_politica_ruim_vale_menos(jogo, otimo):
    V, _ = otimo
    sempre_para_aos_10 = lambda s: "parar" if s >= 10 else "rolar"
    assert policy_evaluation(jogo, sempre_para_aos_10)[0] < V[0]


def test_q_learning_aprende_politica_e_valor(jogo, otimo):
    V, _ = otimo
    env = jogo.as_env()
    treino = jogo.as_env(exploring_starts=True)
    rng = make_rng()
    Q, log = q_learning(treino, episodes=60_000, rng=rng, q_init=TETO + 6)  # otimista, passo por visita
    pol = greedy_policy(Q, env, jogo.states)
    assert pol[4] == "rolar" and pol[32] == "parar"
    retornos = rollout_returns(env, pol, n=40_000, rng=rng)
    assert retornos.mean() == pytest.approx(V[0], abs=0.3)  # quase ótimo: erra só perto do limiar
    x, curva = log.curve()
    assert len(curva) == 60  # uma média por janela de 1.000 episódios


def test_mc_control_roda(jogo):
    env = jogo.as_env()
    Q, log = mc_control(env, episodes=5_000, rng=make_rng())
    assert len(log.returns) == 5_000


def test_validacao_por_simulacao_do_otimo(jogo, otimo):
    V, pol = otimo
    retornos = rollout_returns(jogo.as_env(), pol, n=200_000, rng=make_rng())
    assert check_mean(retornos, expected=V[0]).status == "validada"


def test_nash_pedra_papel_tesoura():
    nash = pytest.importorskip("nashpy")
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    eqs = list(nash.Game(A).support_enumeration())
    assert len(eqs) == 1
    sigma_1, sigma_2 = eqs[0]
    assert np.allclose(sigma_1, 1 / 3) and np.allclose(sigma_2, 1 / 3)
