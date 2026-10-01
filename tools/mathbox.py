"""Caixa de ferramentas simbólica do agente matemático.

Tudo aqui trabalha com aritmética exata (sympy.Rational) — o matemático entrega
valores exatos e o validador confere por simulação. Nada de float até a hora
de imprimir.

Uso típico, de dentro de um script de derivação:

    import sys; sys.path.insert(0, "tools")
    from mathbox import *

ou pela linha de comando, para conferências rápidas:

    uv run python tools/mathbox.py identity "(1+x)**2" "1 + 2*x + x**2"
"""

from __future__ import annotations

import itertools
import sys
from typing import Hashable, Iterable, Sequence

import sympy as sp

__all__ = [
    "R",
    "verify_identity",
    "solve_exact",
    "to_latex",
    "transition_matrix",
    "stationary_distribution",
    "absorbing_analysis",
    "dice_outcomes",
    "dice_pattern_probability",
    "sum_distribution",
    "expected_value",
    "show",
]


def R(p, q=1) -> sp.Rational:
    """Atalho para um racional exato: R(1, 6) == 1/6."""
    return sp.Rational(p, q)


# ---------------------------------------------------------------------------
# Álgebra
# ---------------------------------------------------------------------------

def verify_identity(lhs, rhs, assumptions: dict | None = None) -> bool:
    """True se lhs - rhs simplifica para zero.

    Aceita expressões sympy ou strings. `assumptions` mapeia nome de símbolo
    para hipóteses do sympy, ex.: {"n": {"integer": True, "positive": True}}.
    """
    local = {}
    for name, flags in (assumptions or {}).items():
        local[name] = sp.Symbol(name, **flags)
    lhs = sp.sympify(lhs, locals=local)
    rhs = sp.sympify(rhs, locals=local)
    diff = sp.simplify(sp.expand(lhs - rhs))
    return diff == 0


def solve_exact(equations, unknowns):
    """sympy.solve com saída em dicionário; strings são aceitas."""
    eqs = [sp.sympify(e) for e in (equations if isinstance(equations, (list, tuple)) else [equations])]
    syms = [sp.Symbol(u) if isinstance(u, str) else u for u in (unknowns if isinstance(unknowns, (list, tuple)) else [unknowns])]
    return sp.solve(eqs, syms, dict=True)


def to_latex(expr) -> str:
    """LaTeX pronto para colar entre $$ ... $$ no .qmd."""
    return sp.latex(sp.sympify(expr))


# ---------------------------------------------------------------------------
# Cadeias de Markov (exatas)
# ---------------------------------------------------------------------------

def transition_matrix(states: Sequence[Hashable], transitions: dict) -> sp.Matrix:
    """Monta a matriz de transição P[i, j] = P(i -> j) a partir de um dict.

    transitions = {"A": {"A": R(1,2), "B": R(1,2)}, "B": {"B": 1}}
    Linhas ausentes são tratadas como estado absorvente. Valida que cada
    linha soma exatamente 1.
    """
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = sp.zeros(n, n)
    for s in states:
        row = transitions.get(s, {s: 1})
        for t, p in row.items():
            P[idx[s], idx[t]] += sp.nsimplify(p)
        total = sum(P.row(idx[s]))
        if sp.simplify(total - 1) != 0:
            raise ValueError(f"linha {s!r} soma {total}, não 1")
    return P


def stationary_distribution(P: sp.Matrix) -> sp.Matrix:
    """Distribuição estacionária pi (vetor linha) com pi P = pi, sum(pi) = 1.

    Levanta erro se a solução não for única (cadeia não irredutível).
    """
    n = P.shape[0]
    A = (P.T - sp.eye(n)).col_join(sp.ones(1, n))
    b = sp.zeros(n, 1).col_join(sp.Matrix([1]))
    sol, params = A.gauss_jordan_solve(b)
    if params.shape[0] > 0:
        raise ValueError("distribuição estacionária não é única")
    return sol.T


def absorbing_analysis(P: sp.Matrix, states: Sequence[Hashable] | None = None) -> dict:
    """Análise de cadeia absorvente na forma canônica.

    Retorna:
      transient, absorbing : listas de estados
      N  : matriz fundamental (I - Q)^-1 — visitas esperadas
      t  : passos esperados até absorção, por estado transiente inicial
      B  : probabilidades de absorção B[i, k] = P(terminar em absorbing[k] | início em transient[i])
    """
    n = P.shape[0]
    states = list(states) if states is not None else list(range(n))
    absorbing = [i for i in range(n) if P[i, i] == 1]
    transient = [i for i in range(n) if i not in absorbing]
    if not absorbing:
        raise ValueError("cadeia sem estados absorventes")
    Q = P.extract(transient, transient)
    Rm = P.extract(transient, absorbing)
    N = (sp.eye(len(transient)) - Q).inv()
    t = N * sp.ones(len(transient), 1)
    B = N * Rm
    return {
        "transient": [states[i] for i in transient],
        "absorbing": [states[i] for i in absorbing],
        "N": sp.simplify(N),
        "t": sp.simplify(t),
        "B": sp.simplify(B),
    }


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def dice_outcomes(n: int, faces: int = 6) -> Iterable[tuple[int, ...]]:
    """Todas as faces^n jogadas ordenadas (equiprováveis)."""
    return itertools.product(range(1, faces + 1), repeat=n)


def dice_pattern_probability(n: int, predicate, faces: int = 6) -> sp.Rational:
    """P(predicate(jogada)) por enumeração exata das faces^n jogadas.

    Bom para padrões tipo "full house", "sequência", "quadra" — e serve de
    conferência independente de qualquer fórmula combinatória fechada.
    Fica lento acima de ~8 dados de 6 faces.
    """
    hits = sum(1 for roll in dice_outcomes(n, faces) if predicate(roll))
    return sp.Rational(hits, faces**n)


def sum_distribution(n: int, faces: int = 6) -> dict[int, sp.Rational]:
    """Distribuição exata da soma de n dados, via função geradora."""
    x = sp.Symbol("x")
    poly = sp.Poly(sp.expand((sum(x**k for k in range(1, faces + 1))) ** n), x)
    total = sp.Integer(faces) ** n
    return {int(m[0]): sp.Rational(c, total) for m, c in zip(poly.monoms(), poly.coeffs())}


def expected_value(dist: dict) -> sp.Expr:
    """E[X] para uma distribuição {valor: probabilidade}."""
    return sp.nsimplify(sum(sp.sympify(v) * p for v, p in dist.items()))


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------

def show(name: str, value) -> None:
    """Imprime valor exato, aproximação e LaTeX — formato que vai para o modelo.md."""
    value = sp.sympify(value)
    approx = sp.N(value, 10) if value.is_number else "—"
    print(f"{name} = {value}  ≈ {approx}\n  LaTeX: {sp.latex(value)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd, *args = argv[1:]
    if cmd == "identity" and len(args) == 2:
        ok = verify_identity(*args)
        print("IDENTIDADE OK" if ok else "NÃO É IDENTIDADE")
        return 0 if ok else 2
    if cmd == "latex" and len(args) == 1:
        print(to_latex(args[0]))
        return 0
    if cmd == "simplify" and len(args) == 1:
        show("resultado", sp.simplify(sp.sympify(args[0])))
        return 0
    print(f"comando desconhecido: {cmd} (use identity | latex | simplify)")
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
