"""Figuras sugeridas na pauta (matplotlib, rótulos em português, sem título).

    import sys; sys.path.insert(0, "posts/estrategia-bozo-markov/_work")
    import figuras
    fig = figuras.fig_distribuicao()
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validacao as v  # noqa: E402

_cache = {}


def _dados():
    if not _cache:
        V = v.solve_dp(True)
        dec, outs = v.policy_tables(V, True, v.RANK_H6)
        decg, outsg = v.greedy_tables()
        _cache.update(V=V, dec=dec, outs=outs, outsg=outsg, decg=decg)
        _cache["Fo"], _cache["pos_o"], _ = v.flow(outs)
        _cache["Fg"], _cache["pos_g"], _ = v.flow(outsg)
    return _cache


def fig_distribuicao():
    d = _dados()
    x = np.arange(v.L)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, d["Fo"], label="política ótima", color="#1f77b4")
    ax.plot(x, d["Fg"], label="gulosa", color="#d62728")
    ax.set_xlim(20, 260)
    ax.set_xlabel("pontuação final")
    ax.set_ylabel("probabilidade")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_casas():
    d = _dados()
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.2, d["pos_o"], 0.4, label="ótima", color="#1f77b4")
    ax.bar(x + 0.2, d["pos_g"], 0.4, label="gulosa", color="#d62728")
    ax.set_xticks(x, v.CASAS, rotation=45)
    ax.set_ylabel("probabilidade de pontuar na casa")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def mapa_decisao(top=25):
    """Tabela (lista de linhas) com a ação ótima do 1º lançamento na cartela vazia, para as mãos mais prováveis."""
    d = _dados()
    a = d["dec"][0][1023]
    rows = []
    for h in np.argsort(-v.PI0)[:top]:
        act = a[h]
        txt = "marcar " + v.CASAS[act] if act < 10 else "guardar " + "".join(map(str, v.KEEPS[act - 10])) if act > 10 else "relançar tudo"
        rows.append(("".join(map(str, v.HANDS[h])), float(v.PI0[h]), txt))
    return rows
