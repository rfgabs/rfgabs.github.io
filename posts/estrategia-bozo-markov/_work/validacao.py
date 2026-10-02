"""Validação independente das afirmações do post estrategia-bozo-markov.

Escrito só a partir das regras (pauta.md) e dos enunciados (claims.yaml).
Rodar:  uv run python posts/estrategia-bozo-markov/_work/validacao.py [--escrever]
  --escrever : grava `validacao` e o status em claims.yaml.

Estrutura:
  * motor literal vetorizado de partidas (dados, guardar, marcar) -> `play`
  * DP própria: valor por cartela (1024 máscaras), níveis de lançamento r=3,2,1
    com matriz T (210 guardas x 252 mãos) montada por enumeração de lançamentos ORDENADOS
  * fluxo de probabilidade para distribuições exatas (ótima e gulosa)
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from montecarlo import (check_mean, check_proportion, chi2_distribution, make_rng,  # noqa: E402
                        wilson_interval)
from scipy import stats  # noqa: E402

WORK = Path(__file__).resolve().parent
CASAS = ["Ás", "Duque", "Terno", "Quadra", "Quina", "Sena", "Fú", "Seguida", "Quadrada", "General"]
FACES = np.arange(1, 7)
TOL = 1e-9
NEG = -np.inf

# ---------------------------------------------------------------- estruturas
HANDS = list(itertools.combinations_with_replacement(range(1, 7), 5))  # 252
HIDX = {h: i for i, h in enumerate(HANDS)}
HC = np.zeros((252, 6), dtype=np.int64)
for i, h in enumerate(HANDS):
    for x in h:
        HC[i, x - 1] += 1

KEEPS = [k for j in range(5) for k in itertools.combinations_with_replacement(range(1, 7), j)]  # 210
KIDX = {k: i for i, k in enumerate(KEEPS)}
KC = np.zeros((210, 6), dtype=np.int64)
for i, k in enumerate(KEEPS):
    for x in k:
        KC[i, x - 1] += 1
KSZ = KC.sum(1)
OK = (KC[:, None, :] <= HC[None, :, :]).all(2)  # (210,252): guarda k é sub-multiconjunto da mão

# T[k,h] (contagens) por enumeração de lançamentos ordenados dos dados relançados
TN = np.zeros((210, 252), dtype=np.int64)
for i, k in enumerate(KEEPS):
    for roll in itertools.product(range(1, 7), repeat=5 - len(k)):
        TN[i, HIDX[tuple(sorted(k + roll))]] += 1
DEN = np.array([6 ** (5 - len(k)) for k in KEEPS], dtype=np.int64)
assert (TN.sum(1) == DEN).all()
T = TN / DEN[:, None]
PI0 = T[0]  # guarda vazia = lançar os 5

W6 = 6 ** np.arange(5)
LUT = np.zeros(7776, dtype=np.int16)
for h, i in HIDX.items():
    LUT[((np.array(h) - 1) * W6).sum()] = i
P6 = 6 ** np.arange(6)
KLUT = np.full(int(P6.sum() * 5) + 1, -1)
for i in range(210):
    KLUT[int((KC[i] * P6).sum())] = i


def score_counts(cnt):
    """cnt (...,6) -> pontos (...,10) sem boca."""
    out = np.zeros(cnt.shape[:-1] + (10,), dtype=np.int64)
    out[..., :6] = cnt * FACES
    mx = cnt.max(-1)
    out[..., 6] = 20 * (((cnt == 3).sum(-1) == 1) & ((cnt == 2).sum(-1) == 1))
    seg = (cnt[..., 0:5] == 1).all(-1) | (cnt[..., 1:6] == 1).all(-1)
    out[..., 7] = 30 * seg
    out[..., 8] = 40 * (mx == 4)
    out[..., 9] = 50 * (mx == 5)
    return out


S = score_counts(HC).T.copy()  # (10,252)
BOCA = np.zeros((10, 252), dtype=np.int64)
BOCA[6:] = 5 * (S[6:] > 0)
S1 = S + BOCA


def hidx_of(d):
    s = np.sort(d, axis=1).astype(np.int64)
    return LUT[((s - 1) * W6).sum(1)].astype(np.int64)


def hold_from_keep(d, kidx):
    oh = d[:, :, None] == FACES
    cum = np.cumsum(oh, axis=1) - oh
    rank = (cum * oh).sum(2)
    allowed = KC[kidx[:, None], d.astype(np.int64) - 1]
    return rank < allowed


# ranks de desempate (H6 e o oposto)
_kk = sorted(range(210), key=lambda i: (-KSZ[i], -sum(KEEPS[i]), tuple(-x for x in sorted(KEEPS[i], reverse=True))))
RANK_H6 = np.zeros(220, dtype=np.int64)
RANK_H6[:10] = np.arange(10)
for pos, i in enumerate(_kk):
    RANK_H6[10 + i] = 10 + pos
RANK_OPP = 219 - RANK_H6


# ---------------------------------------------------------------- DP
def levels(mask, V, boca=True):
    bits = [c for c in range(10) if mask >> c & 1]
    Vn = np.array([V[mask ^ (1 << c)] for c in bits])[:, None]
    M = (S[bits] + Vn).max(0)
    M1 = ((S1 if boca else S)[bits] + Vn).max(0)
    E3 = T @ M
    K3 = np.where(OK, E3[:, None], NEG).max(0)
    W2 = np.maximum(M, K3)
    E2 = T @ W2
    K2 = np.where(OK, E2[:, None], NEG).max(0)
    W1 = np.maximum(M1, K2)
    return bits, Vn, E3, E2, W1


def solve_dp(boca=True):
    V = np.zeros(1024)
    for mask in range(1, 1024):
        V[mask] = PI0 @ levels(mask, V, boca)[4]
    return V


def action_matrices(mask, V, boca, lv=None):
    """Matrizes de valor das ações (220,252) para r=1,2,3."""
    bits, Vn, E3, E2, _ = lv if lv is not None else levels(mask, V, boca)
    out = []
    for r in (1, 2, 3):
        A = np.full((220, 252), NEG)
        A[bits] = (S1 if (boca and r == 1) else S)[bits] + Vn
        if r == 1:
            A[10:] = np.where(OK, E2[:, None], NEG)
        elif r == 2:
            A[10:] = np.where(OK, E3[:, None], NEG)
        out.append(A)
    return out


def choose(A, rank, tol=TOL):
    best = A.max(0)
    cand = A >= best - tol
    key = np.where(cand, rank[:, None], 10 ** 6)
    return key.argmin(0)


def outcomes_of(acts, boca=True):
    """acts: 3 vetores de ação por mão -> distribuição (casa, pontos) da rodada (10,56)."""
    p = PI0.copy()
    out = np.zeros((10, 56))
    ar = np.arange(252)
    for r in range(3):
        a = acts[r]
        Sr = S1 if (r == 0 and boca) else S
        m = a < 10
        np.add.at(out, (a[m], Sr[a[m], ar[m]]), p[m])
        if r < 2:
            kw = np.bincount(a[~m] - 10, weights=p[~m], minlength=210)
            p = kw @ T
        else:
            assert m.all()
    return out


def policy_tables(V, boca=True, rank=RANK_H6, want_gap=False):
    dec = np.zeros((3, 1024, 252), dtype=np.int16)
    outs = np.zeros((1024, 10, 56))
    gaps = []
    for mask in range(1, 1024):
        As = action_matrices(mask, V, boca)
        acts = []
        for r in range(3):
            a = choose(As[r], rank)
            dec[r, mask] = a
            acts.append(a)
            if want_gap:
                sec = np.partition(As[r], -2, axis=0)[-2]
                gaps.append(As[r].max(0) - sec)
        outs[mask] = outcomes_of(acts, boca)
    if want_gap:
        return dec, outs, np.concatenate(gaps)
    return dec, outs


# ---------------------------------------------------------------- gulosa
def _keepG(h):
    c = HC[h]
    f = max(range(6), key=lambda x: (c[x], x))  # mais frequente; empate -> maior face
    return KIDX[tuple([f + 1] * min(int(c[f]), 4))]  # 5 iguais: a gulosa sempre para (guarda não usada)


KEEPG = np.array([_keepG(h) for h in range(252)])
FIVE = (HC.max(1) == 5)


def greedy_tables():
    dec = np.zeros((3, 1024, 252), dtype=np.int16)
    outs = np.zeros((1024, 10, 56))
    ar = np.arange(252)
    for mask in range(1, 1024):
        free = np.array([(mask >> c) & 1 for c in range(10)], bool)
        acts = []
        for r in range(3):
            Sr = S1 if r == 0 else S
            imm = np.where(free[:, None], Sr, -1)
            casa = imm.argmax(0)
            best = imm[casa, ar]
            stop = (r == 2) | ((casa >= 6) & (best > 0)) | FIVE
            a = np.where(stop, casa, 10 + KEEPG)
            dec[r, mask] = a
            acts.append(a)
        outs[mask] = outcomes_of(acts, True)
    return dec, outs


def eval_policy(outs):
    Vg = np.zeros(1024)
    pts = np.arange(56)
    for mask in range(1, 1024):
        t = 0.0
        for c in range(10):
            if mask >> c & 1:
                t += (outs[mask, c] * (pts + Vg[mask ^ (1 << c)])).sum()
        Vg[mask] = t
    return Vg


L = 400


def flow(outs):
    F = np.zeros((1024, L))
    F[1023, 0] = 1.0
    pos = np.zeros(10)
    for mask in range(1023, 0, -1):
        reach = F[mask].sum()
        for c in range(10):
            if mask >> c & 1:
                pos[c] += reach * outs[mask, c, 1:].sum()
                F[mask ^ (1 << c)] += np.convolve(F[mask], outs[mask, c])[:L]
    reach = F.sum(1)
    last = np.array([reach[1 << c] for c in range(10)])
    return F[0], pos, last


def dist_stats(p):
    x = np.arange(len(p))
    m = (p * x).sum()
    sd = np.sqrt((p * (x - m) ** 2).sum())
    cdf = np.cumsum(p)
    q = lambda a: int(np.argmax(cdf >= a - 1e-12))
    return dict(mean=m, sd=sd, q5=q(0.05), med=q(0.5), q95=q(0.95), moda=int(p.argmax()))


# ---------------------------------------------------------------- motor literal
def play(rng, N, decide, rounds=10, boca=True, free0=1023, first=None):
    free = np.full(N, free0, np.int32)
    pts = np.zeros((N, 10), np.int16)
    last = np.full(N, -1, np.int8)
    fdice = np.zeros((N, 5), np.int8)
    stop0 = np.zeros(N, np.int8)
    for rd in range(rounds):
        use_first = rd == 0 and first is not None
        if use_first:
            dice = first["dice"].copy()
            r0 = first["r"]
        else:
            dice = rng.integers(1, 7, (N, 5), dtype=np.int8)
            r0 = 1
        active = np.ones(N, bool)
        for r in range(r0, 4):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            d = dice[idx]
            fr = free[idx]
            dec = first["decide"] if (use_first and r == r0 and first.get("decide")) else decide
            casa, hold = dec(r, d, fr)
            casa = casa.astype(np.int64)
            if r == 3:
                low = fr & -fr
                casa = np.where(casa < 0, np.log2(low).astype(np.int64), casa)
            stop = casa >= 0
            if stop.any():
                ds = d[stop]
                cs = casa[stop]
                assert ((fr[stop] >> cs) & 1).all(), "marcou casa ocupada"
                cnt = (ds[:, :, None] == FACES).sum(1)
                sc = score_counts(cnt)[np.arange(len(cs)), cs]
                if boca and r == 1:
                    sc = sc + 5 * ((cs >= 6) & (sc > 0))
                si = idx[stop]
                pts[si, cs] = sc
                free[si] = free[si] & ~(1 << cs).astype(np.int32)
                last[si] = cs
                fdice[si] = ds
                if rd == 0:
                    stop0[si] = r
                active[si] = False
            ns = ~stop
            if ns.any():
                ki = idx[ns]
                h = hold[ns]
                dice[ki] = np.where(h, d[ns], rng.integers(1, 7, h.shape, dtype=np.int8))
    return dict(pts=pts, last=last, fdice=fdice, stop0=stop0, free=free)


def table_decider(dec):
    def f(r, d, fr):
        a = dec[r - 1][fr, hidx_of(d)].astype(np.int64)
        casa = np.where(a < 10, a, -1)
        hold = np.zeros(d.shape, bool)
        kp = a >= 10
        if kp.any():
            hold[kp] = hold_from_keep(d[kp], a[kp] - 10)
        return casa, hold
    return f


def greedy_decider(r, d, fr):
    n = len(d)
    cnt = (d[:, :, None] == FACES).sum(1)
    sc = score_counts(cnt)
    if r == 1:
        sc = sc + 5 * ((np.arange(10) >= 6) & (sc > 0))
    free = ((fr[:, None] >> np.arange(10)) & 1).astype(bool)
    imm = np.where(free, sc, -1)
    casa = imm.argmax(1)
    best = imm[np.arange(n), casa]
    mx = cnt.max(1)
    stop = (r == 3) | ((casa >= 6) & (best > 0)) | (mx == 5)
    f = (cnt * 8 + np.arange(6)).argmax(1)
    hold = d == (f + 1)[:, None]
    return np.where(stop, casa, -1), hold


def target_np(kind, cnt):
    mx = cnt.max(1)
    if kind == "geral":
        return mx == 5
    if kind == "quadrada":
        return mx == 4
    if kind == "fu":
        return ((cnt == 3).sum(1) == 1) & ((cnt == 2).sum(1) == 1)
    if kind == "seguida":
        return (cnt[:, 0:5] == 1).all(1) | (cnt[:, 1:6] == 1).all(1)


# ---------------------------------------------------------------- utilidades
RES = {}


def reg(cid, ok, metodo, n, est, ic, nota):
    RES[cid] = dict(ok=bool(ok), metodo=metodo, n=n, estimativa=est, ic=ic, nota=nota)
    print(f"{cid:>4} {'VALIDADA ' if ok else 'DIVERGENTE'} | est {est} | ic {ic} | {nota}", flush=True)


def fmt_ic(lo, hi):
    return f"[{lo:.6g}; {hi:.6g}]"


def mean_ic(x, conf=0.99):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.norm.ppf(0.5 + conf / 2)
    return x.mean() - t * se, x.mean() + t * se


CL = {c["id"]: c for c in yaml.safe_load((WORK / "claims.yaml").read_text(encoding="utf-8"))["claims"]}


def cval(cid):
    return float(Fraction(CL[cid]["valor"]))


# ---------------------------------------------------------------- C1-C8
def brute_hand_probs():
    cls = {"geral": 0, "quadrada": 0, "fu": 0, "seguida": 0}
    for dice in itertools.product(range(1, 7), repeat=5):
        c = sorted(collections.Counter(dice).values(), reverse=True)
        s = sorted(dice)
        if c == [5]:
            cls["geral"] += 1
        elif c == [4, 1]:
            cls["quadrada"] += 1
        elif c == [3, 2]:
            cls["fu"] += 1
        if s in ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]):
            cls["seguida"] += 1
    return {k: Fraction(v, 7776) for k, v in cls.items()}


def valida_c1_c4(rng):
    ex = brute_hand_probs()
    n = 10 ** 7
    hits = {k: 0 for k in ex}
    for _ in range(4):
        d = rng.integers(1, 7, (n // 4, 5), dtype=np.int8)
        cnt = (d[:, :, None] == FACES).sum(1)
        for k in ex:
            hits[k] += int(target_np(k, cnt).sum())
    for cid, k in [("C1", "geral"), ("C2", "quadrada"), ("C3", "fu"), ("C4", "seguida")]:
        lo, hi = wilson_interval(hits[k], n, 0.99)
        exp = Fraction(CL[cid]["valor"])
        ok = ex[k] == exp and lo <= float(exp) <= hi
        reg(cid, ok, "enumeração das 6^5 jogadas (exata) + MC literal", n, f"{hits[k]/n:.6g} (exato {ex[k]})",
            fmt_ic(lo, hi), "enumeração == valor afirmado (Fraction) e IC de Wilson 99% contém o valor")


def maxprob_exact(kind):
    tg = target_np(kind, HC)
    P = [Fraction(int(t)) for t in tg]  # nível 3: sucesso = alvo
    Es = []
    for lev in (3, 2):
        E = []
        for k in range(210):
            nz = np.flatnonzero(TN[k])
            E.append(sum(int(TN[k, h]) * P[h] for h in nz) / int(DEN[k]))
        Es.append(E)
        P = [Fraction(1) if tg[h] else max(E[k] for k in np.flatnonzero(OK[:, h])) for h in range(252)]
    ans = sum(Fraction(int(TN[0, h]), int(DEN[0])) * P[h] for h in range(252))
    pol = {}
    for r, E in ((2, Es[0]), (1, Es[1])):
        Ef = np.array([float(e) for e in E])
        pol[r] = np.where(OK, Ef[:, None], NEG).argmax(0)
    pol[3] = np.zeros(252, dtype=int)
    return ans, pol


def prob_decider(kind, pol, c):
    def f(r, d, fr):
        cnt = (d[:, :, None] == FACES).sum(1)
        casa = np.where(target_np(kind, cnt), c, -1)
        hold = hold_from_keep(d, pol[r][hidx_of(d)])
        return casa, hold
    return f


def valida_c5_c8(rng):
    for cid, kind, c in [("C5", "geral", 9), ("C6", "quadrada", 8), ("C7", "fu", 6), ("C8", "seguida", 7)]:
        ans, pol = maxprob_exact(kind)
        exp = Fraction(CL[cid]["valor"])
        n = 4 * 10 ** 6 if kind == "geral" else 2 * 10 ** 6
        hits = 0
        for _ in range(n // 10 ** 6):
            R = play(rng, 10 ** 6, prob_decider(kind, pol, c), rounds=1, free0=1 << c)
            cnt = (R["fdice"][:, :, None] == FACES).sum(1)
            hits += int(target_np(kind, cnt).sum())
        lo, hi = wilson_interval(hits, n, 0.99)
        ok = ans == exp and lo <= float(exp) <= hi
        reg(cid, ok, "DP exata (Fraction) sobre 252 mãos x 210 guardas + MC literal da política", n,
            f"{hits/n:.6g} (DP exata {float(ans):.9f})", fmt_ic(lo, hi), f"DP exata == afirmado: {ans == exp}")


# ---------------------------------------------------------------- C9
def valida_c9(rng):
    Q = np.zeros((252, 252))
    for h in range(252):
        if not FIVE[h]:
            Q[h] = T[KEEPG[h]]
    Nn = np.linalg.solve(np.eye(252) - Q, np.where(FIVE, 0.0, 1.0))  # absorvente: 0 lançamentos adicionais
    exact = 1 + PI0 @ Nn
    n = 2 * 10 ** 6
    d = rng.integers(1, 7, (n, 5), dtype=np.int8)
    rolls = np.ones(n)
    active = np.ones(n, bool)
    while True:
        cnt = (d[:, :, None] == FACES).sum(1)
        active &= ~(cnt.max(1) == 5)
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        dd = d[idx]
        c = (cnt[idx] * 8 + np.arange(6)).argmax(1)
        hold = dd == (c + 1)[:, None]
        d[idx] = np.where(hold, dd, rng.integers(1, 7, hold.shape, dtype=np.int8))
        rolls[idx] += 1
    v = check_mean(rolls, cval("C9"))
    ok = abs(exact - cval("C9")) < 1e-9 and v.agrees
    reg("C9", ok, "sistema linear em 252 mãos (outra cadeia) + MC literal", n,
        f"{rolls.mean():.5f} (linear {exact:.9f})", fmt_ic(v.ci_low, v.ci_high), "")


# ---------------------------------------------------------------- DP principal
def main_dp_checks(rng, V, dec, outs, Vnb, decnb):
    singles = np.array([V[1 << c] for c in range(10)])
    fN = table_decider(dec)
    N1 = 10 ** 6
    mcs = []
    for c in range(10):
        R = play(rng, N1, fN, rounds=1, free0=1 << c)
        mcs.append(R["pts"][:, c].astype(float))
    ci = [check_mean(mcs[c], singles[c]) for c in range(10)]
    ok10 = all(abs(singles[k] - 455 * (k + 1) / 216) < 1e-9 for k in range(6)) and all(ci[k].agrees for k in range(6))
    reg("C10", ok10, "DP própria nas cartelas de 1 casa + MC literal (6 casas, 1e6 cada)", N1,
        f"Ás {mcs[0].mean():.4f} (DP {singles[0]:.9f}); Sena {mcs[5].mean():.4f} (DP {singles[5]:.9f})",
        f"Ás {fmt_ic(ci[0].ci_low, ci[0].ci_high)}", "identidade 455k/216 confere em k=1..6 (|dif|<1e-9)")
    for cid, c in [("C11", 6), ("C12", 7), ("C13", 8), ("C14", 9)]:
        ok = abs(singles[c] - cval(cid)) < 1e-9 and ci[c].agrees
        reg(cid, ok, "DP própria (1 casa livre) + MC literal da política", N1,
            f"{mcs[c].mean():.4f} (DP {singles[c]:.9f})", fmt_ic(ci[c].ci_low, ci[c].ci_high),
            "ponte: DP nas cartelas de 1 casa reproduz C10-C14" if cid == "C14" else "")
    tot = singles.sum()
    ok15 = abs(tot - cval("C15")) < 1e-9 and tot < V[1023] / 2
    reg("C15", ok15, "soma dos valores DP das 10 cartelas de uma casa", None,
        f"{tot:.9f} (afirmado {cval('C15'):.9f})", "exato (float64, tol 1e-9)",
        f"{tot:.4f} < C17/2 = {V[1023]/2:.4f}: {tot < V[1023]/2}")
    cnt = sum(1 for m in range(1, 1024) for r in (1, 2, 3) for _ in HANDS)
    reg("C16", cnt == 773388, "contagem por enumeração (máscaras x lançamentos x mãos)", None, str(cnt), "exato",
        "252 = C(10,5) multiconjuntos de 5 dados de 6 faces")
    N = 10 ** 6
    R = play(rng, N, fN)
    tot_pts = R["pts"].sum(1).astype(float)
    v = check_mean(tot_pts, V[1023])
    ok = abs(V[1023] - cval("C17")) < 1e-9 and v.agrees
    reg("C17", ok, "DP própria (float64) + MC literal 1e6 partidas da política ótima", N,
        f"MC {tot_pts.mean():.4f}; DP {V[1023]:.10f}", fmt_ic(v.ci_low, v.ci_high),
        f"|DP - afirmado| = {abs(V[1023]-cval('C17')):.2e}")
    Rn = play(rng, N, table_decider(decnb), boca=False)
    totn = Rn["pts"].sum(1).astype(float)
    vn = check_mean(totn, Vnb[1023])
    ok = abs(Vnb[1023] - cval("C18")) < 1e-9 and vn.agrees
    reg("C18", ok, "DP própria sem boca + MC literal 1e6 partidas (boca=False)", N,
        f"MC {totn.mean():.4f}; DP {Vnb[1023]:.10f}", fmt_ic(vn.ci_low, vn.ci_high),
        f"|DP - afirmado| = {abs(Vnb[1023]-cval('C18')):.2e}")
    diff = tot_pts.mean() - totn.mean()
    se = np.sqrt(tot_pts.var(ddof=1) / N + totn.var(ddof=1) / N)
    z = stats.norm.ppf(0.995)
    d19 = V[1023] - Vnb[1023]
    ok = abs(d19 - cval("C19")) < 1e-9 and abs(diff - d19) < z * se
    reg("C19", ok, "DP(com boca) - DP(sem boca) + MC (duas amostras independentes)", N,
        f"MC {diff:.4f}; DP {d19:.10f}", fmt_ic(diff - z * se, diff + z * se), "")
    Fo, pos_o, last_o = flow(outs)
    so = dist_stats(Fo)
    ok = (abs(so["sd"] - 31.5055287427) < 1e-6 and so["q5"] == 98 and so["med"] == 146 and so["q95"] == 202
          and so["moda"] == 147 and round(so["mean"], 4) == 146.7284 and abs(so["mean"] - V[1023]) < 1e-9)
    chi = chi_bins(tot_pts, Fo, N)
    mcq = [int(np.quantile(tot_pts, q, method="inverted_cdf")) for q in (0.05, 0.5, 0.95)]
    ok = ok and chi["status"] == "validada"
    reg("C20", ok, "fluxo de probabilidade exato (política ótima) + MC 1e6 com qui-quadrado", N,
        f"exato: dp {so['sd']:.10f}, q5 {so['q5']}, med {so['med']}, q95 {so['q95']}, moda {so['moda']}; "
        f"MC dp {tot_pts.std(ddof=1):.3f}, quantis {mcq}", f"chi2 p={chi['p_value']:.3f}", f"média exata {so['mean']:.6f}")
    c21 = [0.8460, 0.9743, 0.9833, 0.9932, 0.9931, 0.9953, 0.9196, 0.7649, 0.9112, 0.2750]
    mine = [round(x, 4) for x in pos_o]
    vs = [check_proportion(R["pts"][:, c] > 0, pos_o[c]) for c in range(10)]
    ok = mine == c21 and abs(pos_o[9] - cval("C21")) < 1e-6 and all(x.agrees for x in vs)
    reg("C21", ok, "fluxo exato + MC literal (1e6) por casa, Wilson 99%", N,
        "exato " + ", ".join(f"{x:.4f}" for x in pos_o), f"MC dentro do IC em {sum(x.agrees for x in vs)}/10 casas",
        f"arredondado 4 casas == afirmado: {mine == c21}")
    vl = [check_proportion(R["last"] == c, last_o[c]) for c in (9, 7, 8)]
    ok = ([round(last_o[c], 4) for c in (9, 7, 8)] == [0.2348, 0.2364, 0.1196]
          and abs(last_o[9] - cval("C22")) < 1e-6 and all(x.agrees for x in vl))
    reg("C22", ok, "fluxo exato (P(chegar à 10ª rodada só com a casa c livre)) + MC literal 1e6", N,
        f"exato General {last_o[9]:.6f}, Seguida {last_o[7]:.6f}, Quadrada {last_o[8]:.6f}",
        f"MC General {vl[0].estimate:.4f} {fmt_ic(vl[0].ci_low, vl[0].ci_high)}", "")
    return dict(R=R, tot_pts=tot_pts, Fo=Fo, pos_o=pos_o, last_o=last_o, so=so)


def chi_bins(samples, F, N):
    obs = np.bincount(samples.astype(int), minlength=L)
    ob, ex, o_acc, e_acc, b = {}, {}, 0, 0.0, 0
    for x in range(L):
        o_acc += obs[x]
        e_acc += F[x] * N
        if e_acc >= 20:
            ob[b], ex[b] = o_acc, e_acc / N
            b += 1
            o_acc, e_acc = 0, 0.0
    ob[b - 1] += o_acc
    ex[b - 1] += e_acc / N
    s = sum(ex.values())
    return chi2_distribution(ob, {k: v / s for k, v in ex.items()})


# ---------------------------------------------------------------- C23-C28 (política)
def q_values(V, mask, r, hand, boca=True):
    A = action_matrices(mask, V, boca)[r - 1]
    return A[:, HIDX[tuple(sorted(hand))]]


def q_mc(rng, V, dec, hand, r, action, n=4 * 10 ** 6, boca=True, mask=1023):
    """MC de uma rodada a partir do estado, após a ação, mais V da cartela restante."""
    d0 = np.tile(np.array(sorted(hand), dtype=np.int8), (n, 1))
    if action < 10:
        fdec = lambda rr, d, fr: (np.full(len(d), action), np.zeros(d.shape, bool))
    else:
        def fdec(rr, d, fr):
            return np.full(len(d), -1), hold_from_keep(d, np.full(len(d), action - 10))
    R = play(rng, n, table_decider(dec), rounds=1, boca=boca, free0=mask, first=dict(dice=d0, r=r, decide=fdec))
    return R["pts"].sum(1) + V[R["free"]]


def valida_c23_c28(rng, V, dec, decg):
    full = 1023
    q = q_values(V, full, 1, (5, 6, 6, 6, 6))
    k6 = 10 + KIDX[(6, 6, 6, 6)]
    best = int(np.argmax(q))
    mark_q = q[8]
    quads = [h for h in range(252) if HC[h].max() == 4]
    never = all(dec[r][full][h] >= 10 for r in (0, 1) for h in quads)
    mc = q_mc(rng, V, dec, (5, 6, 6, 6, 6), 1, k6)
    v = check_mean(mc, q[k6])
    mm = q_mc(rng, V, dec, (5, 6, 6, 6, 6), 1, 8, n=1000)
    lo, hi = mean_ic(mc)
    ok = (best == k6 and abs(q[k6] - 160.559383) < 1e-6 and abs(mark_q - 156.312679) < 1e-6 and never
          and abs((q[k6] - mark_q) - cval("C23")) < 1e-9 and v.agrees and abs(mm.mean() - mark_q) < 1e-9)
    reg("C23", ok, "DP própria (Q das ações) + MC literal da rodada com guarda 6666 (4e6) + marcação literal", len(mc),
        f"guardar 6666 {q[k6]:.6f} (MC {mc.mean():.4f}); marcar Quadrada {mark_q:.6f}; vantagem {q[k6]-mark_q:.6f}",
        f"MC guardar: {fmt_ic(lo, hi)}",
        f"melhor ação = guardar 6666: {best == k6}; quadra+avulso nunca marcada em r=1,2 (cartela vazia): {never}")
    q = q_values(V, full, 2, (1, 1, 4, 4, 4))
    k444 = 10 + KIDX[(4, 4, 4)]
    fu = q[6]
    best1 = int(np.argmax(q))
    q2 = q_values(V, full, 2, (1, 1, 1, 4, 4))
    k111 = 10 + KIDX[(1, 1, 1)]
    best2 = int(np.argmax(q2))
    adv1 = q[k444] - fu
    adv2 = q2[6] - q2[k111]
    mc = q_mc(rng, V, dec, (1, 1, 4, 4, 4), 2, k444, n=8 * 10 ** 6)
    v = check_mean(mc, q[k444])
    lo, hi = mean_ic(mc)
    mc2 = q_mc(rng, V, dec, (1, 1, 1, 4, 4), 2, k111, n=4 * 10 ** 6)
    v2 = check_mean(mc2, q2[k111])
    ok = (best1 == k444 and abs(q[k444] - 146.768173) < 1e-6 and abs(fu - 146.684218) < 1e-6
          and abs(adv1 - cval("C24")) < 1e-9 and best2 == 6 and abs(adv2 - 1.309725) < 1e-6 and v.agrees and v2.agrees)
    reg("C24", ok, "DP própria + MC literal da rodada a partir do estado (8e6 e 4e6)", len(mc),
        f"{{1,1,4,4,4}}: guardar 444 {q[k444]:.6f} (MC {mc.mean():.4f}) vs Fú {fu:.6f}; vantagem {adv1:.6f}. "
        f"{{1,1,1,4,4}}: Fú {q2[6]:.6f} vs guardar 111 {q2[k111]:.6f} (MC {mc2.mean():.4f}) vantagem {adv2:.6f}",
        f"MC guardar 444: {fmt_ic(lo, hi)}", f"ações ótimas conforme afirmado: {best1 == k444}, {best2 == 6}")
    q = q_values(V, full, 3, (1, 2, 3, 4, 6))
    best = int(np.argmax(q))
    adv = q[0] - q[5]
    mca = q_mc(rng, V, dec, (1, 2, 3, 4, 6), 3, 0, n=1000)
    mcb = q_mc(rng, V, dec, (1, 2, 3, 4, 6), 3, 5, n=1000)
    ok = (best == 0 and abs(q[0] - 138.267962) < 1e-6 and abs(q[5] - 131.675316) < 1e-6 and abs(adv - cval("C25")) < 1e-9
          and abs(mca.mean() - q[0]) < 1e-9 and abs(mcb.mean() - q[5]) < 1e-9)
    reg("C25", ok, "DP própria + marcação literal (determinística) + V da cartela restante (validado em C17)", 1000,
        f"Ás {q[0]:.6f}, Sena {q[5]:.6f}, vantagem {adv:.6f}", "exato (determinístico)", f"melhor ação é marcar Ás: {best == 0}")
    q = q_values(V, full, 1, (2, 2, 3, 4, 5))
    k2345 = 10 + KIDX[(2, 3, 4, 5)]
    k22 = 10 + KIDX[(2, 2)]
    best = int(np.argmax(q))
    adv = q[k2345] - q[k22]
    mc1 = q_mc(rng, V, dec, (2, 2, 3, 4, 5), 1, k2345)
    mc2 = q_mc(rng, V, dec, (2, 2, 3, 4, 5), 1, k22)
    v1 = check_mean(mc1, q[k2345])
    v2 = check_mean(mc2, q[k22])
    ok = (best == k2345 and abs(q[k2345] - 147.452503) < 1e-6 and abs(q[k22] - 144.634927) < 1e-6
          and abs(adv - cval("C26")) < 1e-9 and v1.agrees and v2.agrees)
    reg("C26", ok, "DP própria + MC literal da rodada para as duas ações (4e6 cada)", 4 * 10 ** 6,
        f"guardar 2345 {q[k2345]:.6f} (MC {mc1.mean():.4f}); guardar 22 {q[k22]:.6f} (MC {mc2.mean():.4f}); vantagem {adv:.6f}",
        f"2345: {fmt_ic(v1.ci_low, v1.ci_high)}; 22: {fmt_ic(v2.ci_low, v2.ci_high)}", f"melhor ação: {best == k2345}")
    # C27
    a1 = dec[0][full]
    marks = np.flatnonzero(a1 < 10)
    expected_marks = [h for h in range(252) if S[6, h] > 0 or S[7, h] > 0 or S[9, h] > 0]
    same_set = set(marks) == set(expected_marks) and len(marks) == 38
    prob = float(PI0[marks].sum())
    quad_marked = [h for h in marks if S[8, h] > 0]
    two = [h for h in range(252) if sorted(HC[h], reverse=True)[:2] == [2, 2]]
    ok2 = [a1[h] == 10 + KIDX[(max(f + 1 for f in range(6) if HC[h, f] == 2),) * 2] for h in two]
    pairs = {(1, 2, 3, 4, 6): (2, 3, 4), (1, 2, 3, 5, 6): (2, 3, 5), (1, 2, 4, 5, 6): (2, 4, 5), (1, 3, 4, 5, 6): (3, 4, 5)}
    ok3 = [a1[HIDX[h]] == 10 + KIDX[k] for h, k in pairs.items()]
    As = action_matrices(full, V, True)[0]
    srt = np.sort(As, axis=0)
    gap = srt[-1] - srt[-2]
    strict = all(gap[h] > 1e-9 for h in list(two) + [HIDX[h] for h in pairs])
    N = 10 ** 6
    Rm = play(rng, N, table_decider(dec), rounds=1)
    vv = check_proportion(Rm["stop0"] == 1, 91 / 1296)
    ok = (same_set and abs(prob - 91 / 1296) < 1e-12 and not quad_marked and len(two) == 60 and all(ok2) and all(ok3) and vv.agrees)
    reg("C27", ok, "decisões da minha DP na cartela vazia (252 mãos) + MC literal da 1ª rodada (1e6)", N,
        f"{len(marks)} mãos marcam (== General/Fú/Seguida: {same_set}); P(parar)={prob:.9f}; MC {vv.estimate:.5f}",
        fmt_ic(vv.ci_low, vv.ci_high),
        f"quadrada de boca marcada: {bool(quad_marked)}; mãos de dois pares: {len(two)}, todas guardam o par alto: {all(ok2)}; "
        f"12346 etc: {all(ok3)}; decisões estritas (sem empate): {strict}")
    # C28
    ag = decg[0][full]
    ao = dec[0][full]
    p_eq = float(PI0[ag == ao].sum())
    best = As.max(0)
    gv = As[ag, np.arange(252)]
    p_any = float(PI0[gv >= best - 1e-9].sum())
    d = rng.integers(1, 7, (N, 5), dtype=np.int8)
    h = hidx_of(d)
    cg, hg = greedy_decider(1, d, np.full(N, 1023))
    kc = np.stack([(hg & (d == f)).sum(1) for f in FACES], 1)
    code_g = np.where(cg >= 0, cg, 10 + KLUT[(kc * P6).sum(1)])
    code_o = ao[h]
    vm = check_proportion(code_g == code_o, 1151 / 1296)
    ok = abs(p_eq - 1151 / 1296) < 1e-12 and vm.agrees
    reg("C28", ok, "tabelas de ação (minha DP e gulosa própria) ponderadas por P(mão) + MC literal 1e6 (gulosa literal x tabela ótima)", N,
        f"exato {p_eq:.9f} (1151/1296={1151/1296:.9f}); MC {vm.estimate:.5f}", fmt_ic(vm.ci_low, vm.ci_high),
        f"lido com desempate H6; se 'mesma ação' = a ação gulosa é ótima (valor igual a 1e-9): {p_any:.6f}")


# ---------------------------------------------------------------- gulosa C29-C34 e C35
def valida_gulosa(rng, V, outs_g, opt):
    Vg = eval_policy(outs_g)
    Fg, pos_g, last_g = flow(outs_g)
    sg = dist_stats(Fg)
    N = 10 ** 6
    Rg = play(rng, N, greedy_decider)
    tg = Rg["pts"].sum(1).astype(float)
    v = check_mean(tg, Vg[1023])
    ok = abs(Vg[1023] - cval("C29")) < 1e-9 and v.agrees
    reg("C29", ok, "avaliação exata da política gulosa (minha) + MC literal 1e6 (gulosa escrita direto sobre os dados)", N,
        f"exato {Vg[1023]:.10f}; MC {tg.mean():.4f}", fmt_ic(v.ci_low, v.ci_high), f"|dif| = {abs(Vg[1023]-cval('C29')):.2e}")
    d30 = V[1023] - Vg[1023]
    tot = opt["tot_pts"]
    diff = tot.mean() - tg.mean()
    se = np.sqrt(tot.var(ddof=1) / len(tot) + tg.var(ddof=1) / N)
    z = stats.norm.ppf(0.995)
    ok = abs(d30 - cval("C30")) < 1e-9 and abs(diff - d30) < z * se
    reg("C30", ok, "C17 - C29 (exatos) + MC (2 amostras independentes)", N, f"exato {d30:.10f}; MC {diff:.4f}",
        fmt_ic(diff - z * se, diff + z * se), "")
    ok = abs(tg.mean() - cval("C29")) < 0.1 and v.agrees and abs(tg.std(ddof=1) - 32.39) < 0.2
    reg("C31", ok, "MC vetorizado independente, semente 20260101, 1e6 partidas gulosas", N,
        f"média {tg.mean():.3f}, dp {tg.std(ddof=1):.3f}", fmt_ic(v.ci_low, v.ci_high),
        f"afirmado 120.633 / IC [120.549; 120.716] / dp 32.39; |média - C29| = {abs(tg.mean()-cval('C29')):.3f}")
    chi = chi_bins(tg, Fg, N)
    ok = (abs(sg["sd"] - 32.3875761005) < 1e-6 and sg["q5"] == 72 and sg["med"] == 114 and sg["q95"] == 180 and sg["moda"] == 100
          and round(sg["mean"], 4) == 120.6650 and chi["status"] == "validada")
    reg("C32", ok, "fluxo exato (gulosa) + MC literal 1e6 com qui-quadrado", N,
        f"exato dp {sg['sd']:.10f}, q5 {sg['q5']}, med {sg['med']}, q95 {sg['q95']}, moda {sg['moda']}",
        f"chi2 p={chi['p_value']:.3f}", f"média exata {sg['mean']:.6f}; MC dp {tg.std(ddof=1):.3f}")
    c33 = [0.5722, 0.7851, 0.8712, 0.9204, 0.9511, 0.9742, 0.8584, 0.2587, 0.9282, 0.2983]
    mine = [round(x, 4) for x in pos_g]
    vs = [check_proportion(Rg["pts"][:, c] > 0, pos_g[c]) for c in range(10)]
    ok = mine == c33 and abs(pos_g[7] - cval("C33")) < 1e-6 and all(x.agrees for x in vs)
    reg("C33", ok, "fluxo exato (gulosa) + MC literal por casa", N, "exato " + ", ".join(f"{x:.4f}" for x in pos_g),
        f"MC dentro do IC em {sum(x.agrees for x in vs)}/10", f"arredondado == afirmado: {mine == c33}")
    cdf_g = np.cumsum(Fg)
    po = opt["Fo"]
    gt = float((po * np.concatenate([[0], cdf_g[:-1]])).sum())
    eq = float((po * Fg).sum())
    mcgt = check_proportion(tot > tg, gt)
    ok = abs(gt - cval("C34")) < 1e-6 and round(eq, 5) == 0.00784 and mcgt.agrees
    reg("C34", ok, "convolução exata das duas distribuições (independentes) + MC pareado 1e6", N,
        f"exato P(ótima>gulosa)={gt:.9f}, empate {eq:.6f}; MC {mcgt.estimate:.5f}", fmt_ic(mcgt.ci_low, mcgt.ci_high), "")
    return Vg


def valida_c35(V, dec, gaps, outs_h6):
    n_tie = {t: int((gaps <= t).sum()) for t in (1e-12, 1e-10, 1e-9, 1e-7, 1e-6)}
    decopp, outs_opp = policy_tables(V, True, RANK_OPP)
    Fh, pos_h, last_h = flow(outs_h6)
    Fo, pos_o, last_o = flow(outs_opp)
    dF = np.abs(Fh - Fo).max()
    dP = np.abs(pos_h - pos_o).max()
    dec_diff = int((dec != decopp).sum())
    ok = n_tie[1e-9] == 9441 and dF < 1e-12 and dP < 1e-12
    reg("C35", ok, "minha DP: gap entre melhor e 2º melhor valor de ação nas 773388 situações; fluxo com desempate oposto",
        len(gaps), f"{n_tie[1e-9]} empates (tol 1e-9)", "contagens por tolerância: " + str(n_tie),
        f"desempate oposto: máx|dist final| = {dF:.2e}, máx|prob casa| = {dP:.2e}; decisões que mudam: {dec_diff}")


# ---------------------------------------------------------------- main
def tudo():
    t0 = time.time()
    rng = make_rng()
    print("construindo DP...", flush=True)
    V = solve_dp(True)
    Vnb = solve_dp(False)
    print(f"  DP pronta {time.time()-t0:.1f}s; V[1023]={V[1023]:.10f}  sem boca {Vnb[1023]:.10f}", flush=True)
    dec, outs, gaps = policy_tables(V, True, RANK_H6, want_gap=True)
    decnb, _ = policy_tables(Vnb, False, RANK_H6)
    decg, outsg = greedy_tables()
    print(f"  políticas prontas {time.time()-t0:.1f}s", flush=True)
    valida_c1_c4(rng)
    valida_c5_c8(rng)
    valida_c9(rng)
    opt = main_dp_checks(rng, V, dec, outs, Vnb, decnb)
    valida_c23_c28(rng, V, dec, decg)
    valida_gulosa(rng, V, outsg, opt)
    valida_c35(V, dec, gaps, outs)
    el = time.time() - t0
    print(f"\nTempo total: {el:.0f}s")
    (WORK / "validacao_resultados.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for cid in sorted(RES, key=lambda s: int(s[1:])):
        print(cid, "OK" if RES[cid]["ok"] else "DIVERGENTE")
    print("sem veredito:", [f"C{i}" for i in range(1, 36) if f"C{i}" not in RES])


def escrever_claims():
    import claims as cm
    data = cm.load(str(WORK.parent))
    res = json.loads((WORK / "validacao_resultados.json").read_text(encoding="utf-8"))
    for c in data["claims"]:
        r = res.get(c["id"])
        if r:
            c["validacao"] = dict(metodo=r["metodo"], n=r["n"], estimativa=r["estimativa"], ic=r["ic"], nota=r["nota"])
    cm.save(str(WORK.parent), data)
    for cid, r in res.items():
        cm.main(["set", str(WORK.parent), cid, "validada" if r["ok"] else "divergente", "--nota",
                 (r["metodo"] + " | " + r["nota"])[:400]])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--so-escrever" in sys.argv:
        escrever_claims()
    else:
        tudo()
        if "--escrever" in sys.argv:
            escrever_claims()
