"""Caixa de ferramentas do agente validador.

O validador confere cada afirmação do matemático por um caminho independente —
simulação, enumeração por força bruta ou teste estatístico — e devolve um
veredito com intervalo de confiança. Sempre com semente fixa, para que o
número no post seja reprodutível.

    import sys; sys.path.insert(0, "tools")
    from montecarlo import *

    rng = make_rng()
    hits = simulate_bernoulli(lambda rng: (rng.integers(1, 7, 5) == 6).all(), n=10**6, rng=rng)
    print(check_proportion(hits, expected=1/7776))
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "SEED",
    "make_rng",
    "simulate_bernoulli",
    "simulate_values",
    "wilson_interval",
    "mean_interval",
    "Verdict",
    "check_proportion",
    "check_mean",
    "chi2_distribution",
]

SEED = 20260101


def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulate_bernoulli(trial: Callable[[np.random.Generator], bool], n: int, rng=None) -> np.ndarray:
    """Roda `trial(rng)` n vezes e devolve o vetor booleano de sucessos.

    Para n grande, prefira escrever o experimento vetorizado em numpy e passar
    o vetor direto para check_proportion — este laço é para experimentos com
    lógica sequencial (ex.: estratégia de rerrolagem) difíceis de vetorizar.
    """
    rng = rng or make_rng()
    return np.fromiter((bool(trial(rng)) for _ in range(n)), dtype=bool, count=n)


def simulate_values(trial: Callable[[np.random.Generator], float], n: int, rng=None) -> np.ndarray:
    """Igual a simulate_bernoulli, mas para uma variável numérica."""
    rng = rng or make_rng()
    return np.fromiter((float(trial(rng)) for _ in range(n)), dtype=float, count=n)


def wilson_interval(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """IC de Wilson para proporção — se comporta bem perto de 0 e 1."""
    z = stats.norm.ppf(0.5 + conf / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def mean_interval(values: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """IC t de Student para a média."""
    m = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    t = stats.t.ppf(0.5 + conf / 2, len(values) - 1)
    return m - t * se, m + t * se


@dataclass
class Verdict:
    estimate: float
    ci_low: float
    ci_high: float
    expected: float | None
    n: int
    conf: float
    agrees: bool | None  # None quando não há valor esperado

    @property
    def status(self) -> str:
        if self.agrees is None:
            return "estimada"
        return "validada" if self.agrees else "divergente"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d

    def __str__(self) -> str:
        exp = "" if self.expected is None else f" | esperado {self.expected:.6g}"
        return (
            f"[{self.status.upper()}] estimativa {self.estimate:.6g} "
            f"IC{int(self.conf * 100)}% [{self.ci_low:.6g}, {self.ci_high:.6g}] "
            f"(n={self.n:,}){exp}"
        )


def check_proportion(hits, expected: float | None = None, conf: float = 0.99) -> Verdict:
    """Compara uma proporção simulada com o valor exato do matemático.

    `hits` é um vetor booleano. Usa IC de 99% por padrão: com dezenas de
    afirmações por post, 95% geraria falsos alarmes demais.
    """
    hits = np.asarray(hits, dtype=bool)
    n, k = len(hits), int(hits.sum())
    lo, hi = wilson_interval(k, n, conf)
    agrees = None if expected is None else bool(lo <= float(expected) <= hi)
    return Verdict(k / n, lo, hi, None if expected is None else float(expected), n, conf, agrees)


def check_mean(values, expected: float | None = None, conf: float = 0.99) -> Verdict:
    """Compara uma média simulada (ex.: número esperado de jogadas) com o valor exato."""
    values = np.asarray(values, dtype=float)
    lo, hi = mean_interval(values, conf)
    agrees = None if expected is None else bool(lo <= float(expected) <= hi)
    return Verdict(float(values.mean()), lo, hi, None if expected is None else float(expected), len(values), conf, agrees)


def chi2_distribution(observed_counts: dict, expected_probs: dict, alpha: float = 0.01) -> dict:
    """Teste qui-quadrado de aderência: a distribuição inteira bate com a teórica?

    observed_counts = {valor: contagem}, expected_probs = {valor: probabilidade}.
    """
    keys = sorted(expected_probs)
    obs = np.array([observed_counts.get(k, 0) for k in keys], dtype=float)
    exp = np.array([float(expected_probs[k]) for k in keys]) * obs.sum()
    stat, p = stats.chisquare(obs, exp)
    return {"chi2": float(stat), "p_value": float(p), "status": "validada" if p > alpha else "divergente"}
