"""Jogos como Processos de Decisão de Markov (MDP) — núcleo comum de otimização.

Um jogo de um jogador com decisões (rerrolar ou parar, qual dado guardar,
qual casa marcar) é um MDP. Descrito uma vez aqui, ele serve para:

- programação dinâmica exata (value/policy iteration) → política ótima;
- avaliação exata de uma política qualquer (sistema linear em SymPy);
- RL (tools/rl.py), via o adaptador `as_env`, para comparar o que um agente
  aprende com o ótimo — ou para jogos grandes demais para enumerar.

Descrição de um jogo:

    def actions(s):           # ações disponíveis; [] = estado terminal
        ...
    def transitions(s, a):    # lista de (probabilidade, próximo_estado, recompensa)
        ...
    jogo = MDP(initial=s0, actions=actions, transitions=transitions)

Probabilidades podem ser sympy.Rational (para avaliação exata) ou float.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

import numpy as np
import sympy as sp

__all__ = ["MDP", "value_iteration", "policy_evaluation_exact", "policy_evaluation", "policy_table"]

State = Hashable
Action = Hashable


@dataclass
class MDP:
    initial: State | Iterable[State]
    actions: Callable[[State], list]
    transitions: Callable[[State, Action], list[tuple]]
    gamma: float = 1.0
    max_states: int = 2_000_000
    states: list = field(init=False)
    index: dict = field(init=False)
    _T: dict = field(init=False, repr=False)   # (s, a) -> transições originais (exatas)
    _Tf: dict = field(init=False, repr=False)  # (s, a) -> (cumprobs, próximos, recompensas) em float

    def __post_init__(self):
        starts = list(self.initial) if isinstance(self.initial, (list, tuple, set)) else [self.initial]
        seen = {s: i for i, s in enumerate(starts)}
        order = list(starts)
        queue = deque(starts)
        self._T, self._Tf = {}, {}
        while queue:
            s = queue.popleft()
            for a in self.actions(s):
                outs = list(self.transitions(s, a))
                self._T[(s, a)] = outs
                probs = np.array([float(p) for p, _, _ in outs])
                self._Tf[(s, a)] = (np.cumsum(probs), [s2 for _, s2, _ in outs], probs, np.array([float(r) for _, _, r in outs]))
                total = 0
                for p, s2, _ in outs:
                    total += p
                    if s2 not in seen:
                        if len(seen) >= self.max_states:
                            raise MemoryError(f"mais de {self.max_states} estados — considere RL ou abstração")
                        seen[s2] = len(order)
                        order.append(s2)
                        queue.append(s2)
                if abs(float(total) - 1) > 1e-9:
                    raise ValueError(f"transições de ({s!r}, {a!r}) somam {total}, não 1")
        self.states, self.index = order, seen

    def is_terminal(self, s) -> bool:
        return not self.actions(s)

    def trans(self, s, a) -> list[tuple]:
        """Transições exatas, em cache (não chama `transitions` de novo)."""
        return self._T[(s, a)]

    def sample(self, s, a, rng):
        """Sorteia (próximo_estado, recompensa) — rápido, para simulação e RL."""
        cum, nexts, _, rewards = self._Tf[(s, a)]
        k = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(nexts) - 1)
        return nexts[k], rewards[k]

    def q_value(self, s, a, V: dict) -> float:
        _, nexts, probs, rewards = self._Tf[(s, a)]
        return float(np.dot(probs, rewards + self.gamma * np.array([V[s2] for s2 in nexts])))

    def as_env(self, exploring_starts: bool = False):
        """Adaptador para tools/rl.py (amostra transições em vez de enumerá-las)."""
        from rl import MDPEnv

        return MDPEnv(self, exploring_starts=exploring_starts)


def value_iteration(mdp: MDP, tol: float = 1e-12, max_iter: int = 100_000) -> tuple[dict, dict]:
    """Política ótima por iteração de valor (Gauss-Seidel). Retorna (V, política).

    Empates são resolvidos pela primeira ação listada em `actions(s)` — liste
    a ação "padrão" primeiro para que a política reportada seja estável.
    """
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        delta = 0.0
        for s in mdp.states:
            acts = mdp.actions(s)
            if not acts:
                continue
            best = max(mdp.q_value(s, a, V) for a in acts)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < tol:
            break
    else:
        raise RuntimeError("value iteration não convergiu")
    policy = {}
    for s in mdp.states:
        acts = mdp.actions(s)
        if acts:
            qs = [mdp.q_value(s, a, V) for a in acts]
            m = max(qs)
            policy[s] = next(a for a, q in zip(acts, qs) if q >= m - 1e-9)
    return V, policy


def policy_evaluation(mdp: MDP, policy: dict | Callable) -> dict:
    """Valor de uma política fixa, numérico (numpy). Para estratégias heurísticas."""
    pol = policy if callable(policy) else policy.__getitem__
    n = len(mdp.states)
    A = np.eye(n)
    b = np.zeros(n)
    for s in mdp.states:
        if mdp.is_terminal(s):
            continue
        i = mdp.index[s]
        for p, s2, r in mdp.trans(s, pol(s)):
            A[i, mdp.index[s2]] -= mdp.gamma * float(p)
            b[i] += float(p) * float(r)
    v = np.linalg.solve(A, b)
    return {s: v[mdp.index[s]] for s in mdp.states}


def policy_evaluation_exact(mdp: MDP, policy: dict | Callable, state=None):
    """Valor EXATO de uma política (sistema linear racional em SymPy).

    Use para transformar a política ótima numérica em uma afirmação exata:
    "o valor esperado jogando de forma ótima é 8.142/1.296". Viável até
    alguns milhares de estados.
    """
    pol = policy if callable(policy) else policy.__getitem__
    live = [s for s in mdp.states if not mdp.is_terminal(s)]
    idx = {s: i for i, s in enumerate(live)}
    n = len(live)
    A = sp.zeros(n, n)
    b = sp.zeros(n, 1)
    g = sp.nsimplify(mdp.gamma)
    for s in live:
        i = idx[s]
        A[i, i] += 1
        for p, s2, r in mdp.trans(s, pol(s)):
            p, r = sp.nsimplify(p), sp.nsimplify(r)
            b[i] += p * r
            if s2 in idx:
                A[i, idx[s2]] -= g * p
    v = A.LUsolve(b)
    values = {s: sp.nsimplify(v[idx[s]]) for s in live}
    if state is not None:
        return values.get(state, sp.Integer(0))
    return values


def policy_table(policy: dict, key: Callable | None = None) -> list[tuple]:
    """Lista ordenada (estado, ação) — útil para mostrar a política no post."""
    items = sorted(policy.items(), key=(lambda kv: key(kv[0])) if key else None)
    return items
