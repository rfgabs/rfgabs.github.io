"""Reinforcement learning tabular — aprender políticas jogando.

Para quando o jogo é grande demais para programação dinâmica exata, ou quando
o próprio aprendizado é o assunto do post ("quantas partidas um agente
precisa para descobrir a estratégia ótima?").

Ambiente (qualquer objeto com estes três métodos):

    reset(rng) -> estado inicial
    actions(s) -> lista de ações ([] = terminal)
    step(s, a, rng) -> (próximo_estado, recompensa, terminou)

`MDP.as_env()` (tools/mdp.py) gera um ambiente a partir de um MDP descrito
por transições — assim dá para comparar a política aprendida com a ótima.

Deep RL (redes neurais, estados contínuos/enormes): instale o extra
`uv sync --extra rl` (gymnasium + stable-baselines3) e embrulhe o ambiente
em um `gymnasium.Env`. Só vale a pena quando tabular não cabe na memória.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["MDPEnv", "TrainingLog", "q_learning", "mc_control", "greedy_policy", "rollout_returns"]


class MDPEnv:
    """Ambiente que amostra as transições de um MDP enumerável.

    exploring_starts=True começa cada episódio num estado não terminal
    aleatório — para TREINAR quando a política tende a nunca visitar regiões
    onde a decisão importa. Avalie sempre com um ambiente normal.
    """

    def __init__(self, mdp, exploring_starts: bool = False):
        self.mdp = mdp
        if exploring_starts:
            starts = [s for s in mdp.states if not mdp.is_terminal(s)]
        else:
            starts = mdp.initial if isinstance(mdp.initial, (list, tuple, set)) else [mdp.initial]
        self._starts = list(starts)

    def reset(self, rng):
        return self._starts[rng.integers(len(self._starts))]

    def actions(self, s):
        return self.mdp.actions(s)

    def step(self, s, a, rng):
        s2, r = self.mdp.sample(s, a, rng)
        return s2, float(r), self.mdp.is_terminal(s2)


@dataclass
class TrainingLog:
    """Curva de aprendizado: retorno médio por janela de episódios."""

    window: int
    returns: list = field(default_factory=list)

    def add(self, g: float):
        self.returns.append(g)

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(self.returns)
        k = len(r) // self.window
        means = r[: k * self.window].reshape(k, self.window).mean(axis=1)
        x = (np.arange(k) + 1) * self.window
        return x, means


def _eps_greedy(Q, s, acts, eps, rng):
    if rng.random() < eps:
        return acts[rng.integers(len(acts))]
    qs = [Q[(s, a)] for a in acts]
    return acts[int(np.argmax(qs))]


def q_learning(
    env,
    episodes: int,
    rng,
    alpha: float | str | Callable[[int], float] = "visitas",
    epsilon: float | Callable[[int], float] = lambda ep: max(0.05, 1 - ep / 5_000),
    gamma: float = 1.0,
    log_window: int = 1_000,
    q_init: float = 0.0,
    alpha_power: float = 0.8,
) -> tuple[dict, TrainingLog]:
    """Q-learning tabular off-policy. Retorna (Q, log da curva de aprendizado).

    `q_init` otimista (acima do maior retorno possível) força a exploração de
    estados raros; com q_init=0 em jogos de recompensa positiva o agente tende
    a subestimar continuar e aprende a parar cedo demais.

    alpha="visitas" (padrão) usa passo 1/N(s,a)^alpha_power, que decai com o
    número de visitas ao par — converge (Robbins–Monro). Passo constante deixa
    Q oscilando, e em decisões apertadas (diferença de valor menor que o ruído)
    a política gulosa escolhe pelo ruído.
    """
    Q = defaultdict(lambda: q_init)
    N = defaultdict(int)
    por_visita = alpha == "visitas"
    log = TrainingLog(log_window)
    for ep in range(episodes):
        a_t = None if por_visita else (alpha(ep) if callable(alpha) else alpha)
        e_t = epsilon(ep) if callable(epsilon) else epsilon
        s = env.reset(rng)
        g, done = 0.0, not env.actions(s)
        while not done:
            acts = env.actions(s)
            a = _eps_greedy(Q, s, acts, e_t, rng)
            s2, r, done = env.step(s, a, rng)
            g += r
            nxt = 0.0 if done else max(Q[(s2, b)] for b in env.actions(s2))
            if por_visita:
                N[(s, a)] += 1
                step = 1.0 / N[(s, a)] ** alpha_power
            else:
                step = a_t
            Q[(s, a)] += step * (r + gamma * nxt - Q[(s, a)])
            s = s2
        log.add(g)
    return dict(Q), log


def mc_control(
    env,
    episodes: int,
    rng,
    epsilon: float | Callable[[int], float] = lambda ep: max(0.05, 1 - ep / 5_000),
    gamma: float = 1.0,
    log_window: int = 1_000,
) -> tuple[dict, TrainingLog]:
    """Controle Monte Carlo on-policy (first-visit), médias incrementais."""
    Q = defaultdict(float)
    N = defaultdict(int)
    log = TrainingLog(log_window)
    for ep in range(episodes):
        e_t = epsilon(ep) if callable(epsilon) else epsilon
        s = env.reset(rng)
        traj = []
        done = not env.actions(s)
        while not done:
            acts = env.actions(s)
            a = _eps_greedy(Q, s, acts, e_t, rng)
            s2, r, done = env.step(s, a, rng)
            traj.append((s, a, r))
            s = s2
        g = 0.0
        first = {}
        for t, (s_, a_, _) in enumerate(traj):
            first.setdefault((s_, a_), t)
        for t in range(len(traj) - 1, -1, -1):
            s_, a_, r = traj[t]
            g = r + gamma * g
            if first[(s_, a_)] == t:
                N[(s_, a_)] += 1
                Q[(s_, a_)] += (g - Q[(s_, a_)]) / N[(s_, a_)]
        log.add(sum(r for _, _, r in traj))
    return dict(Q), log


def greedy_policy(Q: dict, env, states) -> dict:
    """Política gulosa em relação a Q para os estados dados (não terminais)."""
    pol = {}
    for s in states:
        acts = env.actions(s)
        if acts:
            pol[s] = max(acts, key=lambda a: Q.get((s, a), float("-inf")))
    return pol


def rollout_returns(env, policy: dict | Callable, n: int, rng, fallback=None) -> np.ndarray:
    """Retornos de n episódios jogando `policy` — passe para montecarlo.check_mean.

    Estados que a política não cobre usam `fallback(s, acts)` (padrão: primeira ação).
    """
    pol = policy if callable(policy) else (lambda s: policy.get(s))
    out = np.empty(n)
    for i in range(n):
        s = env.reset(rng)
        g, done = 0.0, not env.actions(s)
        while not done:
            acts = env.actions(s)
            a = pol(s)
            if a is None:
                a = fallback(s, acts) if fallback else acts[0]
            s, r, done = env.step(s, a, rng)
            g += r
        out[i] = g
    return out
