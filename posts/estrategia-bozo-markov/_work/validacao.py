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
    fillr = np.zeros((N, 10), np.int8)
    freeh = np.zeros((N, rounds), np.int16)
    for rd in range(rounds):
        freeh[:, rd] = free
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
                fillr[si, cs] = rd + 1
                fdice[si] = ds
                if rd == 0:
                    stop0[si] = r
                active[si] = False
            ns = ~stop
            if ns.any():
                ki = idx[ns]
                h = hold[ns]
                dice[ki] = np.where(h, d[ns], rng.integers(1, 7, h.shape, dtype=np.int8))
    return dict(pts=pts, last=last, fdice=fdice, stop0=stop0, free=free, fillr=fillr, freeh=freeh)


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


# ======================================================================================
# V2 — colas, emendas, lema da diferença de desempenho, Shapley, busca local, teto, MQ, PI
# Escrito só a partir de claims.yaml (hipoteses, gulosa, colas) e dos enunciados.
# ======================================================================================
import re  # noqa: E402
from collections import Counter  # noqa: E402

RAWY = yaml.safe_load((WORK / "claims.yaml").read_text(encoding="utf-8"))
COLAS = RAWY["colas"]
PRECOS = {k: tuple(v["tabela"]) for k, v in COLAS["precos"].items()}
ITENS_TAB_YAML = {k: v["itens_cola"] for k, v in COLAS["precos"].items()}
EM_ITENS = {k: v["itens_cola"] for k, v in COLAS["emendas"].items()}
EM_ORDEM = ["seg4x", "quad_gen", "alvo", "risca_gen", "doispar", "fu_alto"]
R4 = frozenset(["alvo", "seg4x", "quad_gen", "doispar"])
SEG_LO = np.array([1, 1, 1, 1, 1, 0], bool)
SEG_HI = np.array([0, 1, 1, 1, 1, 1], bool)
HREP = np.array(HANDS, dtype=np.int8)
# parâmetros a decorar de cada tabela, lidos do texto da definição de itens_cola
TAB_PARAMS = {"sozinha_int": 10, "2face_8_2": 3, "2face_78112": 5, "2face_comb0": 2, "face_comb0": 2,
              "3face_78112": 5, "num0_8_2": 3, "sozinha_prelim": 10, "busca_local": 10}
SEED_V2 = 20261002


def parse_cola(cid):
    parts = cid.split("+")
    base, em = parts[0], frozenset(parts[1:])
    assert all(e in EM_ITENS for e in em), cid
    return base, (None if base == "gulosa" else PRECOS[base]), em


def itens_cola(cid):
    base, _, em = parse_cola(cid)
    it = 0 if base == "gulosa" else TAB_PARAMS[base] + 1
    return it + sum(EM_ITENS[e] for e in em), (0 if base == "gulosa" else 1) + len(em)


def cola_id(base, em):
    return "+".join([base] + [e for e in EM_ORDEM if e in em])


# ---------------------------------------------------------------- jogador vetorizado (literal sobre os dados)
def cola_core(r, d, fr, prices, em):
    """d (N,5) dados, fr (N,) casas livres (máscara). Devolve (casa|-1, kc (N,6) contagens guardadas)."""
    N = len(d)
    ar = np.arange(N)
    cnt = (d[:, :, None] == FACES).sum(1)
    pts = score_counts(cnt)
    if r == 1:
        pts[:, 6:] += 5 * (pts[:, 6:] > 0)
    free = ((fr[:, None] >> np.arange(10)) & 1).astype(bool)
    pr = np.zeros(10, np.int64) if prices is None else np.asarray(prices, np.int64)
    h = np.where(free, pts - pr, -10 ** 6).argmax(1)  # empate -> menor índice
    if prices is None and "risca_gen" in em:
        anypos = (free & (pts > 0)).any(1)
        low = free.argmax(1)
        h = np.where(anypos, h, np.where(free[:, 9], 9, low))
    hp = pts[ar, h]
    mx = cnt.max(1)
    stop = (r == 3) | ((h >= 6) & (hp > 0)) | (mx == 5)
    fu_ap = np.zeros(N, bool)
    if r < 3:
        exc = np.zeros(N, bool)
        if "quad_gen" in em:
            exc |= (h == 8) & (hp > 0) & free[:, 9]
        if "fu_alto" in em and r == 2:
            tri = (cnt == 3).argmax(1)
            fu_ap = (h == 6) & (hp > 0) & (tri >= 3) & (free[:, 8] | free[:, 9])
            exc |= fu_ap
        stop = stop & ~exc
    fmax = (cnt * 8 + np.arange(6)).argmax(1)  # mais frequente; empate -> face maior
    kc = np.zeros((N, 6), np.int64)
    kc[ar, fmax] = cnt[ar, fmax]
    pres = cnt > 0
    cL = (pres & SEG_LO).sum(1)
    cH = (pres & SEG_HI).sum(1)
    if "alvo" in em:
        useful = free[:, :6] | (free[:, 6] | free[:, 8] | free[:, 9])[:, None]
        cu = np.where(useful, cnt, 0)
        hasu = (cu > 0).any(1)
        fu = (cu * 8 + np.arange(6)).argmax(1)
        kca = np.zeros_like(kc)
        kca[ar, fu] = cu[ar, fu]
        segm = np.where((cH >= cL)[:, None], SEG_HI, SEG_LO)
        kcs = (pres & segm).astype(np.int64) * free[:, 7][:, None]
        kc = np.where(hasu[:, None], kca, kcs)
    ok12 = (~stop) & (r < 3)
    if "seg4x" in em:
        apH = free[:, 7] & ok12 & (cH >= 4)
        apL = free[:, 7] & ok12 & ~apH & (cL >= 4)
        assert not (apH & (cH == 5)).any() and not (apL & (cL == 5)).any(), "seg4x com Seguida completa"
        kc = np.where(apH[:, None], pres & SEG_HI, np.where(apL[:, None], pres & SEG_LO, kc))
    if "doispar" in em and r == 2:
        ap = free[:, 6] & ok12 & ((cnt == 2).sum(1) == 2) & ((cnt == 1).sum(1) == 1)
        kc = np.where(ap[:, None], np.where(cnt == 2, 2, 0), kc)
    if fu_ap.any():
        kc = np.where(fu_ap[:, None], np.where(cnt == 3, 3, 0), kc)
    return np.where(stop, h, -1), kc.astype(np.int64)


def cola_decider(prices, em):
    def f(r, d, fr):
        casa, kc = cola_core(r, d, fr, prices, em)
        kidx = KLUT[(kc * P6).sum(1)]
        kidx = np.where(casa >= 0, 0, kidx)
        assert (kidx >= 0).all()
        return casa, hold_from_keep(d, kidx)
    return f


def cola_dec(prices, em):
    """Tabela de ações (3,1024,252): casa 0..9 ou 10+índice da guarda, para todo estado (S, r, mão)."""
    dec = np.zeros((3, 1024, 252), np.int16)
    d = np.tile(HREP, (1023, 1))
    fr = np.repeat(np.arange(1, 1024), 252).astype(np.int64)
    for r in (1, 2, 3):
        casa, kc = cola_core(r, d, fr, prices, em)
        kidx = KLUT[(kc * P6).sum(1)]
        assert (kidx[casa < 0] >= 0).all()
        dec[r - 1, 1:] = np.where(casa >= 0, casa, 10 + kidx).reshape(1023, 252)
    return dec


# ---------------------------------------------------------------- jogador de referência (escalar, outra estrutura)
def ref_pontos(hand, b, r):
    c = Counter(hand)
    v = sorted(c.values())
    if b < 6:
        return (b + 1) * c[b + 1]
    if b == 6:
        p = 20 if v == [2, 3] else 0
    elif b == 7:
        p = 30 if sorted(hand) in ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) else 0
    elif b == 8:
        p = 40 if v == [1, 4] else 0
    else:
        p = 50 if v == [5] else 0
    return p + 5 if (p > 0 and r == 1) else p


def ref_jogador(r, hand, livres, prices, em):
    """Segue o texto de `colas` como uma pessoa: devolve ('marca', casa) ou ('guarda', faces guardadas)."""
    c = Counter(hand)
    pr = prices if prices is not None else [0] * 10
    h, best = None, None
    for b in sorted(livres):
        u = ref_pontos(hand, b, r) - pr[b]
        if best is None or u > best:
            h, best = b, u
    if prices is None and "risca_gen" in em and all(ref_pontos(hand, b, r) == 0 for b in livres):
        h = 9 if 9 in livres else min(livres)
    ph = ref_pontos(hand, h, r)
    parar = (r == 3) or (h >= 6 and ph > 0) or max(c.values()) == 5
    trinca = [f for f in c if c[f] == 3]
    fu_alto = False
    if r < 3:
        if "quad_gen" in em and h == 8 and ph > 0 and 9 in livres:
            parar = False
        if "fu_alto" in em and r == 2 and h == 6 and ph > 0 and trinca[0] >= 4 and (8 in livres or 9 in livres):
            parar, fu_alto = False, True
    if parar:
        return ("marca", h)
    f = max(c, key=lambda x: (c[x], x))
    guarda = [f] * c[f]
    if "alvo" in em:
        qualquer = bool(livres & {6, 8, 9})
        uteis = [x for x in c if (x - 1) in livres or qualquer]
        if uteis:
            f = max(uteis, key=lambda x: (c[x], x))
            guarda = [f] * c[f]
        elif 7 in livres:
            lo, hi = set(c) & {1, 2, 3, 4, 5}, set(c) & {2, 3, 4, 5, 6}
            guarda = sorted(hi if len(hi) >= len(lo) else lo)
        else:
            guarda = []
    if "seg4x" in em and 7 in livres:
        for seg in ({2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}):  # alta primeiro
            com = set(c) & seg
            if len(com) >= 4:
                assert len(com) == 4
                guarda = sorted(com)
                break
    if "doispar" in em and r == 2 and 6 in livres and sorted(c.values()) == [1, 2, 2]:
        guarda = sorted([x for x in c if c[x] == 2] * 2)
    if fu_alto:
        guarda = [trinca[0]] * 3
    return ("guarda", tuple(sorted(guarda)))


def vec_action_scalar(r, hand, mask, prices, em):
    d = np.array([hand], dtype=np.int8)
    casa, kc = cola_core(r, d, np.array([mask], np.int64), prices, em)
    if casa[0] >= 0:
        return ("marca", int(casa[0]))
    return ("guarda", tuple(sorted(f + 1 for f in range(6) for _ in range(kc[0, f]))))


# ---------------------------------------------------------------- avaliação de política (Bellman) e visitação
def bellman_eval(dec):
    """V^pi por recursão de Bellman de avaliação. Devolve V (1024,) e val[r-1, mask, mão] = V^pi(S, r, mão)."""
    V = np.zeros(1024)
    val = np.zeros((3, 1024, 252))
    ar = np.arange(252)
    for mask in range(1, 1024):
        a3, a2, a1 = (dec[i, mask].astype(np.int64) for i in (2, 1, 0))
        assert ((mask >> a3) & 1).all()
        v3 = S[a3, ar] + V[mask ^ (1 << a3)]
        E3 = T @ v3
        m2 = a2 < 10
        c2 = np.where(m2, a2, 0)
        assert ((mask >> c2[m2]) & 1).all()
        v2 = np.where(m2, S[c2, ar] + V[mask ^ (1 << c2)], E3[np.where(m2, 0, a2 - 10)])
        E2 = T @ v2
        m1 = a1 < 10
        c1 = np.where(m1, a1, 0)
        assert ((mask >> c1[m1]) & 1).all()
        v1 = np.where(m1, S1[c1, ar] + V[mask ^ (1 << c1)], E2[np.where(m1, 0, a1 - 10)])
        V[mask] = PI0 @ v1
        val[0, mask], val[1, mask], val[2, mask] = v1, v2, v3
    return V, val


def visits(dec):
    """P[mask, r, h]: prob. de a rodada passar pelo estado (mask, r, h) dada a cartela mask no início;
    MK[mask, c]: prob. de a rodada terminar marcando c; reach[mask]: prob. de a partida passar por mask."""
    P = np.zeros((1024, 3, 252))
    MK = np.zeros((1024, 10))
    for mask in range(1, 1024):
        p = PI0.copy()
        for r in range(3):
            P[mask, r] = p
            a = dec[r, mask].astype(np.int64)
            m = a < 10
            MK[mask] += np.bincount(a[m], weights=p[m], minlength=10)
            if r < 2:
                p = np.bincount(a[~m] - 10, weights=p[~m], minlength=210) @ T
        assert abs(MK[mask].sum() - 1) < 1e-12
    reach = np.zeros(1024)
    reach[1023] = 1.0
    for mask in range(1023, 0, -1):
        for c in range(10):
            if mask >> c & 1:
                reach[mask ^ (1 << c)] += reach[mask] * MK[mask, c]
    return P, MK, reach


def round_dec(Vc, rank=RANK_H6):
    """Rodada ótima (3 lançamentos, guardas H4, desempate H6) com valor de continuação Vc[cartela restante]."""
    dec = np.zeros((3, 1024, 252), np.int16)
    for mask in range(1, 1024):
        As = action_matrices(mask, Vc, True)
        for r in range(3):
            dec[r, mask] = choose(As[r], rank)
    return dec


_VC = {}


def cola_val(cid):
    """Valor exato (Bellman de avaliação) da cola; em cache."""
    if cid not in _VC:
        base, prices, em = parse_cola(cid)
        _VC[cid] = bellman_eval(cola_dec(prices, em))[0]
    return _VC[cid]


def cv(cid):
    return float(cola_val(cid)[1023])


def pv(prices, em):
    """Valor exato de uma cola dada por (tabela de 10 preços, emendas)."""
    key = (tuple(prices) if prices is not None else None, frozenset(em))
    if key not in _VC:
        _VC[key] = bellman_eval(cola_dec(key[0], key[1]))[0]
    return float(_VC[key][1023])


# ---------------------------------------------------------------- MC literal
def mc_policy(decider, n, rng, chunk=250_000, boca=True):
    out = dict(tot=[], pts=[], fillr=[], freeh=[], last=[])
    for _ in range(n // chunk):
        R = play(rng, chunk, decider, boca=boca)
        out["tot"].append(R["pts"].sum(1).astype(float))
        out["pts"].append(R["pts"])
        out["fillr"].append(R["fillr"])
        out["freeh"].append(R["freeh"])
        out["last"].append(R["last"])
    return {k: np.concatenate(v) for k, v in out.items()}


def ckey(x):
    """cola (id str) ou (preços|None, emendas) -> chave hashable (preços tuple|None, emendas tuple ordenada)."""
    if isinstance(x, str):
        base, prices, em = parse_cola(x)
    else:
        prices, em = x
    return (tuple(prices) if prices is not None else None, tuple(sorted(em)))


def _mc_worker(key, n, seed):
    prices, em = key
    rng = make_rng(seed)
    M = mc_policy(cola_decider(prices, frozenset(em)), n, rng)
    t = M["tot"]
    return key, float(t.mean()), float(t.std(ddof=1)), len(t)


_MCC = {}


def mc_colas(cids, n=10 ** 6, workers=10):
    """MC literal de várias colas em paralelo; devolve {chave: (média, dp, n)}."""
    import zlib
    from concurrent.futures import ProcessPoolExecutor
    keys = [ckey(c) for c in cids]
    todo = [k for k in dict.fromkeys(keys) if k not in _MCC]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_mc_worker, k, n, SEED_V2 + zlib.crc32(repr(k).encode()) % 10 ** 6) for k in todo]
            for f in futs:
                k, m, sd, nn = f.result()
                _MCC[k] = (m, sd, nn)
    return {k: _MCC[k] for k in keys}


def mc_ok(x, exact):
    m, sd, nn = _MCC[ckey(x)]
    hw = stats.norm.ppf(0.995) * sd / np.sqrt(nn)
    return bool(abs(m - exact) <= hw), m, (m - hw, m + hw)


def near(a, b, tol):
    return abs(a - b) <= tol


def r_ok(x, claimed, ndec):
    """x arredondado a ndec casas == claimed (com folga de ponto flutuante)."""
    return abs(x - claimed) <= 0.5 * 10 ** (-ndec) + 1e-9


def t_ci(x, conf=0.99):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.5 + conf / 2, len(x) - 1)
    return x.mean() - t * se, x.mean() + t * se


def decs(txt):
    """Decimais à brasileira (vírgula) de um texto -> floats."""
    return [float(x.replace(",", ".")) for x in re.findall(r"-?\d+,\d+", txt)]


# ---------------------------------------------------------------- contexto da v2
def ctx_v2():
    V = solve_dp(True)
    dec, outs = policy_tables(V, True, RANK_H6)
    decg, outsg = greedy_tables()
    Vg, valg = bellman_eval(decg)
    Vo, valo = bellman_eval(dec)
    return dict(V=V, dec=dec, outs=outs, decg=decg, outsg=outsg, Vg=Vg, valg=valg, Vo=Vo, valo=valo)


def mc_lemma(decider, term, cat, n=10 ** 6, chunk=50_000, seed_off=0):
    """MC literal do lema: soma, ao longo de cada partida, de term[r, mask, mão] nos estados visitados,
    separada por (lançamento, categoria). Devolve médias por bloco (n/chunk blocos x 12)."""
    rng = make_rng(SEED_V2 + seed_off)
    rows = []
    for _ in range(n // chunk):
        acc = np.zeros(12)

        def wrap(r, d, fr):
            h = hidx_of(d)
            idx = (r - 1) * 4 + cat[r - 1, fr, h]
            acc[:] += np.bincount(idx, weights=term[r - 1, fr, h], minlength=12)
            return decider(r, d, fr)
        play(rng, chunk, wrap)
        rows.append(acc / chunk)
    return np.array(rows)


# ---------------------------------------------------------------- C39-C41 (lema da diferença de desempenho)
def valida_c39_c41(cx):
    V, dec, decg, Vg, valg = cx["V"], cx["dec"], cx["decg"], cx["Vg"], cx["valg"]
    Pg, MKg, rg = visits(decg)
    Po, MKo, ro = visits(dec)
    ar = np.arange(252)
    cell_i = np.zeros((3, 4))
    cell_ii = np.zeros((3, 4))
    by_n = np.zeros(11)
    PM = np.zeros((10, 10))
    termI = np.zeros((3, 1024, 252))
    termII = np.zeros((3, 1024, 252))
    catI = np.zeros((3, 1024, 252), np.int64)
    for mask in range(1, 1024):
        As = action_matrices(mask, V, True)
        n = bin(mask).count("1")
        for r in range(3):
            ag = decg[r, mask].astype(np.int64)
            ao = dec[r, mask].astype(np.int64)
            idx = (ag < 10) * 2 + (ao < 10)
            catI[r, mask] = idx
            # forma (i): V*(s) - Q*(s, G(s)), ponderada pela visitação da gulosa
            ti = As[r].max(0) - As[r][ag, ar]
            termI[r, mask] = ti
            wi = rg[mask] * Pg[mask, r] * ti
            cell_i[r] += np.bincount(idx, weights=wi, minlength=4)
            by_n[n] += wi.sum()
            if r == 2:
                both = (ag < 10) & (ao < 10)
                np.add.at(PM, (ag[both], ao[both]), wi[both])
            # forma (ii): Q^G(s, pi*(s)) - V^G(s), ponderada pela visitação da ótima
            Sr = S1 if r == 0 else S
            mk = ao < 10
            acm = np.where(mk, ao, 0)
            qm = Sr[acm, ar] + Vg[mask ^ (1 << acm)]
            if r < 2:
                qk = (T @ valg[r + 1, mask])[np.where(mk, 0, ao - 10)]
            else:
                assert mk.all()
                qk = 0.0
            tii = np.where(mk, qm, qk) - valg[r, mask]
            termII[r, mask] = tii
            cell_ii[r] += np.bincount(idx, weights=ro[mask] * Po[mask, r] * tii, minlength=4)
    c30 = V[1023] - Vg[1023]
    # mapeamento: idx = 2*G_marca + O_marca ; (r, idx)
    def partes(cell):
        return dict(g1=cell[0, 0], g2=cell[1, 0], p2=cell[1, 2], p1=cell[0, 2], c3=cell[2, 3])
    pi_, pii = partes(cell_i), partes(cell_ii)
    resto_i = abs(cell_i).sum() - sum(abs(x) for x in pi_.values())
    resto_ii = abs(cell_ii).sum() - sum(abs(x) for x in pii.values())
    e39 = dict(g1=8.727112, g2=5.740843, p2=1.761433, p1=0.273237, c3=9.560718)
    e40 = dict(g1=10.918087, g2=10.219743, p2=2.117441, p1=0.021142, c3=2.786930)
    ok39 = (abs(cell_i.sum() - c30) < 1e-9 and abs(c30 - cval("C30")) < 1e-9
            and all(r_ok(pi_[k], e39[k], 6) for k in e39) and resto_i < 1e-9
            and r_ok(pi_["g1"] + pi_["g2"], 14.467955, 6) and r_ok(pi_["p1"] + pi_["p2"], 2.034670, 6)
            and abs(pi_["g1"] + pi_["g2"] - cval("C39")) < 1e-9)
    ok40 = (abs(cell_ii.sum() - c30) < 1e-9 and all(r_ok(pii[k], e40[k], 6) for k in e40) and resto_ii < 1e-9
            and r_ok(pii["g1"] + pii["g2"], 21.137830, 6) and r_ok(pii["p1"] + pii["p2"], 2.138583, 6)
            and abs(pii["g1"] + pii["g2"] - cval("C40")) < 1e-9)
    # MC literal dos dois lados (1e6 partidas cada)
    rows_i = mc_lemma(greedy_decider, termI, catI, seed_off=1)
    rows_ii = mc_lemma(table_decider(dec), termII, catI, seed_off=2)
    def mc_check(rows, parts):
        tot_rows = rows.sum(1)
        lo, hi = t_ci(tot_rows)
        okt = lo <= c30 <= hi
        cols = dict(g1=0 * 4 + 0, g2=1 * 4 + 0, p2=1 * 4 + 2, p1=0 * 4 + 2, c3=2 * 4 + 3)
        oks = {}
        for k, c in cols.items():
            l, h_ = t_ci(rows[:, c])
            oks[k] = l <= parts[k] <= h_
        return okt, oks, (lo, hi)
    mt_i, mo_i, ci_i = mc_check(rows_i, pi_)
    mt_ii, mo_ii, ci_ii = mc_check(rows_ii, pii)
    ok39 = ok39 and mt_i and all(mo_i.values())
    ok40 = ok40 and mt_ii and all(mo_ii.values())
    reg("C39", ok39, "forma (i): V*(s)-Q*(s,G(s)) ponderada por visitação da gulosa (visitação por fluxo, Q* da minha DP) "
        "+ MC literal 1e6 (acumula os termos ao longo de partidas gulosas)", 10 ** 6,
        f"soma {cell_i.sum():.9f} (C30 {c30:.9f}); g1 {pi_['g1']:.6f}, g2 {pi_['g2']:.6f}, p2 {pi_['p2']:.6f}, "
        f"p1 {pi_['p1']:.6f}, c3 {pi_['c3']:.6f}; guardas {pi_['g1']+pi_['g2']:.6f}; MC total {rows_i.sum(1).mean():.4f}",
        fmt_ic(*ci_i), f"outras células |.|={resto_i:.1e}; MC dentro do IC nas 5 partes: {all(mo_i.values())}")
    reg("C40", ok40, "forma (ii): Q^G(s,pi*(s))-V^G(s) ponderada por visitação da ótima (Q^G e V^G por Bellman de avaliação) "
        "+ MC literal 1e6 (partidas da ótima)", 10 ** 6,
        f"soma {cell_ii.sum():.9f}; g1 {pii['g1']:.6f}, g2 {pii['g2']:.6f}, p2 {pii['p2']:.6f}, p1 {pii['p1']:.6f}, "
        f"c3 {pii['c3']:.6f}; guardas {pii['g1']+pii['g2']:.6f}; MC total {rows_ii.sum(1).mean():.4f}",
        fmt_ic(*ci_ii), f"outras células |.|={resto_ii:.1e}; MC dentro do IC nas 5 partes: {all(mo_ii.values())}")
    exp_n = {10: 1.1706, 9: 1.3527, 8: 1.6039, 7: 1.8831, 6: 2.1473, 5: 2.4213, 4: 2.9278, 3: 4.0508, 2: 6.9926, 1: 1.5134}
    okn = all(r_ok(by_n[k], exp_n[k], 4) for k in exp_n) and abs(by_n.sum() - c30) < 1e-9
    exp_p = [((7, 9), 2.9032), ((5, 0), 1.1711), ((4, 0), 1.0167), ((3, 0), 0.6607), ((8, 9), 0.5201), ((6, 9), 0.4185)]
    okp = all(r_ok(PM[a, b], x, 4) for (a, b), x in exp_p)
    top6 = sorted(np.ndindex(10, 10), key=lambda ij: -PM[ij])[:6]
    okp = okp and set(top6) == {ab for ab, _ in exp_p}
    okp = okp and abs(PM[7, 9] - cval("C41")) < 1e-9 and abs(PM.sum() - cell_i[2, 3]) < 1e-9
    # MC: parcela 'casa no 3º' por nº de casas livres e por par, via partidas gulosas
    reg("C41", okn and okp, "forma (i) decomposta por nº de casas livres no início da rodada e por par (casa gulosa -> casa ótima) no 3º lançamento",
        None, "por n livres: " + ", ".join(f"{k}:{by_n[k]:.4f}" for k in range(10, 0, -1)) + f"; Seguida->General {PM[7,9]:.6f}, "
        f"Sena->Ás {PM[5,0]:.4f}, Quina->Ás {PM[4,0]:.4f}, Quadra->Ás {PM[3,0]:.4f}, Quadrada->General {PM[8,9]:.4f}, Fú->General {PM[6,9]:.4f}",
        "exato (float64, tol 1e-9); soma por n == C30; soma dos pares == parcela c3 de C39",
        f"top-6 de pares == afirmados: {set(top6) == {ab for ab, _ in exp_p}}; reaproveita a visitação validada em C39 (MC total dentro do IC)")
    return dict(Pg=Pg, rg=rg, Po=Po, ro=ro)


# ---------------------------------------------------------------- forward / Shapley
def forward(prices, cands, start=frozenset()):
    cur = set(start)
    rem = list(cands)
    path = []
    while rem:
        vals = {e: pv(prices, cur | {e}) for e in rem}
        best = max(rem, key=lambda e: vals[e])
        cur.add(best)
        rem.remove(best)
        path.append((best, vals[best], frozenset(cur)))
    return path


def shapley(nums, combs, players_em):
    names = ["num", "comb"] + players_em
    n = len(names)
    val = {}
    for bits in range(1 << n):
        T_ = {names[i] for i in range(n) if bits >> i & 1}
        prices = (nums if "num" in T_ else (0,) * 6) + (combs if "comb" in T_ else (0,) * 4)
        val[bits] = (prices, frozenset(T_ & set(players_em)), pv(prices, T_ & set(players_em)))
    from math import factorial
    phi = []
    for i in range(n):
        s = 0.0
        for bits in range(1 << n):
            if bits >> i & 1:
                continue
            k = bin(bits).count("1")
            s += factorial(k) * factorial(n - k - 1) / factorial(n) * (val[bits | 1 << i][2] - val[bits][2])
        phi.append(s)
    return dict(zip(names, phi)), val


# ---------------------------------------------------------------- C42-C47
def valida_c42():
    g = cv("gulosa")
    ids = {e: cola_id("gulosa", {e}) for e in EM_ORDEM}
    vals = {e: cv(ids[e]) for e in EM_ORDEM}
    gains = {e: vals[e] - g for e in EM_ORDEM}
    eg = dict(seg4x=7.4513, quad_gen=2.0948, alvo=2.8359, risca_gen=-0.6637, doispar=0.6769, fu_alto=-0.8041)
    mcs = {e: mc_ok(ids[e], vals[e]) for e in EM_ORDEM}
    mg = mc_ok("gulosa", g)
    ok = (abs(vals["seg4x"] - cval("C42")) < 1e-9 and r_ok(vals["seg4x"], 128.116360, 6)
          and all(r_ok(gains[e], eg[e], 4) for e in EM_ORDEM) and all(m[0] for m in mcs.values()) and mg[0])
    reg("C42", ok, "Bellman de avaliação da cola (jogador vetorizado a partir do texto; conferido contra jogador escalar independente) "
        "+ MC literal 1e6 por cola", 10 ** 6,
        f"gulosa+seg4x {vals['seg4x']:.6f}; ganhos " + ", ".join(f"{e} {gains[e]:+.4f}" for e in EM_ORDEM),
        "MC: " + "; ".join(f"{e} {mcs[e][1]:.3f} {fmt_ic(*mcs[e][2])}" for e in EM_ORDEM),
        f"MC dentro do IC99 em {sum(m[0] for m in mcs.values())}/6 (+gulosa: {mg[0]})")


def valida_c43():
    path = forward(None, EM_ORDEM)
    exp = [("seg4x", 128.1164, 3), ("quad_gen", 130.2412, 5), ("alvo", 131.6100, 7), ("risca_gen", 134.0498, 8),
           ("doispar", 135.1318, 11), ("fu_alto", 134.6547, 14)]
    ok = True
    mcs = []
    for (e, x, it), (pe, pval, em) in zip(exp, path):
        cid = cola_id("gulosa", em)
        ok &= (e == pe and r_ok(pval, x, 4) and itens_cola(cid)[0] == it)
        mcs.append(mc_ok(cid, pval))
    best = max(path, key=lambda t: t[1])
    ok &= best[0] == "doispar" and abs(best[1] - cval("C43")) < 1e-9 and path[5][1] < path[4][1]
    ok &= all(m[0] for m in mcs)
    reg("C43", ok, "seleção forward refeita com minha avaliação exata (6 passos, 36 colas avaliadas) + MC literal 1e6 por passo",
        10 ** 6, "caminho: " + ", ".join(f"+{e} {v_:.4f} (itens {it})" for (e, x, it), (_, v_, _) in zip(exp, path)),
        "MC: " + "; ".join(f"{m[1]:.3f}" for m in mcs), f"melhor = passo 5 {best[1]:.9f}; MC dentro do IC {sum(m[0] for m in mcs)}/6")


def valida_c44():
    g = cv("gulosa")
    a = cv("gulosa+alvo") - g
    b = cv("2face_8_2+alvo") - cv("2face_8_2")
    c = cv("sozinha_int+alvo") - cv("sozinha_int")
    mcs = [mc_ok(x, cv(x)) for x in ("gulosa+alvo", "2face_8_2", "2face_8_2+alvo", "sozinha_int", "sozinha_int+alvo")]
    ok = (r_ok(a, 2.835866, 6) and r_ok(b, 10.697723, 6) and r_ok(c, 11.062990, 6) and abs(b - cval("C44")) < 1e-9
          and all(m[0] for m in mcs))
    reg("C44", ok, "diferenças de valores exatos (Bellman de avaliação) + MC literal 1e6 de cada cola envolvida", 10 ** 6,
        f"alvo/gulosa {a:.6f}; alvo/2face_8_2 {b:.6f}; alvo/sozinha_int {c:.6f}",
        "MC: " + "; ".join(f"{m[1]:.3f}" for m in mcs), f"MC dentro do IC {sum(m[0] for m in mcs)}/5")


def valida_c45_c46(cx):
    g, o = cv("gulosa"), float(cx["V"][1023])
    for cid, claim, nome, itx, share in (("C45", "sozinha_int", "i", 21, 78.7), ("C46", "2face_8_2", "ii", 14, 78.4)):
        c = cola_id(claim, R4)
        val = cv(c)
        it, regras = itens_cola(c)
        m = mc_ok(c, val)
        sh = (val - g) / (o - g) * 100
        ok = abs(val - cval(cid)) < 1e-9 and r_ok(val, {"C45": 141.173041, "C46": 141.103896}[cid], 6) and it == itx \
            and r_ok(sh, share, 1) and m[0]
        reg(cid, ok, f"Bellman de avaliação da cola ({nome}) + MC literal 1e6", 10 ** 6,
            f"{c}: exato {val:.9f}; itens {it}; {sh:.2f}% do caminho C29->C17; MC {m[1]:.4f}", fmt_ic(*m[2]),
            f"|exato - afirmado| = {abs(val - cval(cid)):.2e}")


def valida_c47():
    a, b = cv("2face_8_2"), cv("sozinha_int")
    ma, mb = mc_ok("2face_8_2", a), mc_ok("sozinha_int", b)
    it = (itens_cola("2face_8_2")[0], itens_cola("sozinha_int")[0])
    ok = (abs(a - cval("C47")) < 1e-9 and r_ok(a, 123.012426, 6) and r_ok(b, 122.968115, 6) and it == (4, 11)
          and ma[0] and mb[0] and a > cv("gulosa") and b > cv("gulosa"))
    reg("C47", ok, "Bellman de avaliação (guardas G4, marcação por pontos - preço) + MC literal 1e6", 10 ** 6,
        f"2face_8_2 {a:.9f} (MC {ma[1]:.4f}); sozinha_int {b:.9f} (MC {mb[1]:.4f}); itens {it}",
        f"{fmt_ic(*ma[2])}; {fmt_ic(*mb[2])}", f"contra gulosa {cv('gulosa'):.6f}")


def valida_c48_c49():
    for cid, nums, combs, exp, claimed, tot_id in (
            ("C48", PRECOS["2face_8_2"][:6], PRECOS["2face_8_2"][6:],
             dict(num=1.2743, comb=5.1980, alvo=5.2608, seg4x=5.6729, quad_gen=2.0457, doispar=0.9870), "seg4x", "2face_8_2"),
            ("C49", PRECOS["sozinha_int"][:6], PRECOS["sozinha_int"][6:],
             dict(num=-0.8005, comb=7.3820, alvo=5.6062, seg4x=5.3714, quad_gen=2.0152, doispar=0.9337), "comb", "sozinha_int")):
        phi, val = shapley(nums, combs, ["alvo", "seg4x", "quad_gen", "doispar"])
        full = pv(nums + combs, R4)
        empty = cv("gulosa")
        soma = sum(phi.values())
        # MC spot-check de 4 coalições
        sample = [(nums + (0,) * 4, frozenset()), ((0,) * 6 + combs, frozenset()), ((0,) * 10, frozenset({"alvo", "seg4x"})),
                  (nums + combs, frozenset({"alvo", "seg4x", "quad_gen"}))]
        mc_colas(sample)
        mcs = [mc_ok(s, pv(*s)) for s in sample]
        valor = phi[claimed]
        ok = (all(r_ok(phi[k], exp[k], 4) for k in exp) and abs(soma - (full - empty)) < 1e-9
              and abs(full - cval({"C48": "C46", "C49": "C45"}[cid])) < 1e-9 and abs(valor - cval(cid)) < 1e-9
              and max(phi, key=phi.get) == claimed and all(m[0] for m in mcs))
        reg(cid, ok, "Shapley exato com 64 coalições avaliadas por Bellman de avaliação + MC literal 1e6 em 4 coalições", 10 ** 6,
            "φ: " + ", ".join(f"{k} {phi[k]:.4f}" for k in phi) + f"; soma {soma:.6f} = v(cheia)-v(vazia) {full - empty:.6f}",
            "MC coalições: " + "; ".join(f"{m[1]:.3f}" for m in mcs),
            f"|φ_{claimed} - valor| = {abs(valor - cval(cid)):.2e}; MC dentro do IC {sum(m[0] for m in mcs)}/4")


def valida_c50():
    p = PRECOS["2face_8_2"]
    path = forward(p, ["alvo", "seg4x", "quad_gen", "doispar", "fu_alto"])
    exp = [("alvo", 133.7101), ("seg4x", 138.0744), ("quad_gen", 139.9576), ("doispar", 141.1039), ("fu_alto", 140.6065)]
    ok = all(e == pe and r_ok(pval, x, 4) for (e, x), (pe, pval, _) in zip(exp, path))
    ok &= abs(path[0][1] - cval("C50")) < 1e-9
    mcs = [mc_ok(cola_id("2face_8_2", em), pval) for (_, pval, em) in path]
    ok &= all(m[0] for m in mcs)
    reg("C50", ok, "seleção forward refeita (5 passos, 15 colas) + MC literal 1e6 por passo", 10 ** 6,
        "; ".join(f"+{e} {pval:.4f}" for (e, _), (_, pval, _) in zip(exp, path)),
        "MC: " + "; ".join(f"{m[1]:.3f}" for m in mcs), f"MC dentro do IC {sum(m[0] for m in mcs)}/5; passo 1 = {path[0][1]:.9f}")


def valida_c51():
    start = list(PRECOS["2face_8_2"])
    f = lambda q: pv(tuple(q), R4)  # noqa: E731
    def vizinhos(q):
        for i in range(1, 10):
            for dlt in (-1, 1):
                w = list(q)
                w[i] += dlt
                yield w
    # subida de encosta de maior ganho (steepest ascent)
    cur, fc = list(start), f(start)
    path = [(tuple(cur), fc)]
    while True:
        best = max(((w, f(w)) for w in vizinhos(cur)), key=lambda t: t[1])
        if best[1] > fc + 1e-12:
            cur, fc = best
            path.append((tuple(cur), fc))
        else:
            break
    # variante: primeira melhora na ordem índice crescente, -1 antes de +1
    cur2, fc2 = list(start), f(start)
    mov = True
    while mov:
        mov = False
        for w in vizinhos(cur2):
            fw = f(w)
            if fw > fc2 + 1e-12:
                cur2, fc2, mov = w, fw, True
                break
    claimed = [2, 4, 6, 8, 10, 13, 5, 7, 8, 3]
    fv = f(claimed)
    viz = [f(w) for w in vizinhos(claimed)]
    m = mc_ok((claimed, R4), fv)
    nties = sum(abs(x - fv) < 1e-9 for x in viz)
    ok = (abs(fv - cval("C51")) < 1e-9 and r_ok(fv, 141.292401, 6) and len(viz) == 18 and max(viz) <= fv + 1e-9
          and list(path[-1][0]) == claimed and m[0])
    reg("C51", ok, "avaliação exata da tabela e dos 18 vizinhos (±1 em 9 preços) + subida de encosta refeita + MC literal 1e6", 10 ** 6,
        f"ótimo local {claimed}: exato {fv:.9f}; melhor vizinho {max(viz):.6f}; steepest termina em {list(path[-1][0])} ({path[-1][1]:.6f}) "
        f"após {len(path)-1} passos; primeira-melhora termina em {cur2} ({fc2:.6f}); MC {m[1]:.4f}", fmt_ic(*m[2]),
        f"nenhum dos 18 vizinhos com valor maior: {max(viz) <= fv + 1e-9} (empates exatos com o ótimo: {nties}; max(viz)-v = {max(viz)-fv:.2e}); "
        f"busca steepest refeita chega à mesma tabela: {list(path[-1][0]) == claimed}; atenção: a variante primeira-melhora acha outro "
        f"ótimo local ({fc2:.6f} > {fv:.6f}), então '141,29' é ótimo LOCAL dependente da regra de busca")


# ---------------------------------------------------------------- C52, C53, C54
def valor_rodada_perfeita(prices):
    pr = np.asarray(prices, float)
    Vadd = np.array([sum(pr[b] for b in range(10) if m >> b & 1) for m in range(1024)])
    return float(bellman_eval(round_dec(Vadd))[0][1023])


def fit_aditivo(V):
    masks = np.arange(1, 1024)
    X = np.zeros((1023, 20))
    for i, m in enumerate(masks):
        X[i, bin(m).count("1") - 1] = 1
        for b in range(10):
            if m >> b & 1:
                X[i, 10 + b] = 1
    y = V[masks]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rms = float(np.sqrt(np.mean((y - X @ beta) ** 2)))
    return beta[10:] - beta[10], rms


def valida_c53(cx):
    p, rms = fit_aditivo(cx["V"])
    exp = [0.00, 1.81, 3.88, 6.15, 8.56, 11.02, 8.53, 9.74, 19.98, 2.35]
    ok = all(abs(round(p[i], 2) - exp[i]) < 0.0101 for i in range(10)) and abs(rms - 1.6629) < 1e-3 \
        and abs(rms - cval("C53")) < 1e-3
    # conferência alternativa: equações normais com restrição p_Ás = 0 (sem pseudo-inversa)
    masks = np.arange(1, 1024)
    X = np.zeros((1023, 19))
    for i, m in enumerate(masks):
        X[i, bin(m).count("1") - 1] = 1
        for b in range(1, 10):
            if m >> b & 1:
                X[i, 9 + b] = 1
    y = cx["V"][masks]
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    rms2 = float(np.sqrt(np.mean((y - X @ beta) ** 2)))
    p2 = np.concatenate([[0.0], beta[10:]])
    ok = ok and abs(rms2 - rms) < 1e-9 and np.abs(p2 - p).max() < 1e-7
    reg("C53", ok, "mínimos quadrados (lstsq mínima norma, normalizado Ás=0) + conferência por equações normais com p_Ás=0", 1023,
        "preços " + ", ".join(f"{x:.2f}" for x in p) + f"; RMS {rms:.6f}", "exato (float64)",
        f"equações normais: RMS {rms2:.6f}, máx|dp| = {np.abs(p2 - p).max():.1e}")
    return p


def valida_c52(cx, p_fit):
    V = cx["V"]
    singles = [float(V[1 << c]) for c in range(10)]
    marg = [float(V[1023] - V[1023 ^ (1 << b)]) for b in range(10)]
    tabelas = {
        "zeros": ((0,) * 10, 138.5362),
        "C10-C14 exatos": (singles, 145.9827),
        "C10-C14 arredondados": (PRECOS["sozinha_int"], 145.9388),
        "2face_8_2": (PRECOS["2face_8_2"], 145.5578),
        "marginais da cartela cheia": (marg, 145.8249),
        "ajuste C53 (exato)": (list(p_fit), 146.3452),
        "ajuste C53 (0,01)": ([round(x, 2) for x in p_fit], 146.3452),
    }
    res = {k: valor_rodada_perfeita(pr) for k, (pr, _) in tabelas.items()}
    # o enunciado cita 'preços do ajuste aditivo de C53' = os listados em C53 (arredondados a 0,01)
    ok = (all(r_ok(res[k], tabelas[k][1], 4) for k in tabelas if k != "ajuste C53 (exato)")
          and abs(res["C10-C14 exatos"] - cval("C52")) < 1e-9)
    # MC literal da rodada perfeita com preços exatos de C10-C14: decide por tabela (jogo completo, 1e6)
    pr = np.array(singles)
    Vadd = np.array([sum(pr[b] for b in range(10) if m >> b & 1) for m in range(1024)])
    decp = round_dec(Vadd)
    M = play(make_rng(SEED_V2 + 52), 10 ** 6, table_decider(decp))
    tot = M["pts"].sum(1).astype(float)
    vm = check_mean(tot, res["C10-C14 exatos"])
    ok = ok and vm.agrees and all(res[k] < float(V[1023]) for k in res)
    reg("C52", ok, "rodada ótima por E[pts - preço] (DP de 3 lançamentos com valor futuro aditivo) + Bellman de avaliação no jogo completo "
        "+ MC literal 1e6 (preços C10-C14 exatos)", 10 ** 6,
        "; ".join(f"{k} {res[k]:.4f}" for k in res) + f"; MC {tot.mean():.4f}", fmt_ic(vm.ci_low, vm.ci_high),
        f"todos < C17 = {V[1023]:.4f}; |exato - afirmado| = {abs(res['C10-C14 exatos'] - cval('C52')):.2e}; "
        f"observação: com os preços do ajuste SEM arredondar o valor é {res['ajuste C53 (exato)']:.4f} (146,3452 só vale com os preços "
        f"arredondados a 0,01 listados em C53)")


def valida_c54(cx):
    V, Vk = cx["V"], cx["Vg"]
    exp = [(120.665037, 28.9, 1, 1), (142.436816, 7.29, 2, 20), (146.086695, 1.49, 2, 121), (146.700005, 0.0947, 4, 443),
           (146.728291, 0.00109, 5, 872), (146.728381, 0.0, 0, 1023)]
    rows, ok = [], True
    pi1 = None
    for k in range(6):
        gap = V[1:] - Vk[1:]
        row = (float(Vk[1023]), float(gap.max()), int((np.abs(gap) < 1e-9).sum()))
        rows.append(row)
        e = exp[k]
        gap_ok = abs(row[1]) < 1e-9 if e[1] == 0 else abs(row[1] - e[1]) <= 0.5 * 10 ** (-e[2]) + 1e-9
        ok &= r_ok(row[0], e[0], 6) and gap_ok and row[2] == e[3]
        if k == 1:
            pi1 = row[0]
        if k < 5:
            Vk = bellman_eval(round_dec(Vk))[0]
    ok &= abs(pi1 - cval("C54")) < 1e-9 and np.abs(V[1:] - Vk[1:]).max() < 1e-9
    reg("C54", ok, "iteração de política refeita (round_dec com continuação V^pi_k + Bellman de avaliação), 5 passos", None,
        "; ".join(f"pi_{k} {r[0]:.6f} (gap máx {r[1]:.4g}, = E* em {r[2]})" for k, r in enumerate(rows)), "exato (float64, tol 1e-9)",
        f"pi_5 == E* nas 1023 cartelas (máx|dif| {np.abs(V[1:] - Vk[1:]).max():.1e})")


# ---------------------------------------------------------------- C55
def valida_c55(cx):
    txt = CL["C55"]["enunciado"]
    lista_txt = txt.split("Fronteira de Pareto")[0]
    reg_ = re.findall(r"([A-Za-z0-9_+]+) \[itens (\d+), regras (\d+)\] ([\d,]+)", lista_txt)
    ids = [r_[0] for r_ in reg_]
    rows = {}
    ok = len(ids) == 38 and len(set(ids)) == 38
    dif = 0.0
    for cid, it, rg, vl in reg_:
        mi, mr = itens_cola(cid)
        val = cv(cid)
        rows[cid] = (mi, mr, val)
        ok &= (mi == int(it) and mr == int(rg))
        dif = max(dif, abs(val - float(vl.replace(",", "."))))
    ok &= dif < 1e-4
    def fronteira(ix):
        seq = sorted(rows, key=lambda c: (rows[c][ix], -rows[c][2]))
        out, best = [], -1e9
        for c in seq:
            if rows[c][2] > best + 1e-12:
                out.append(c)
                best = rows[c][2]
        return out
    fr_it = fronteira(0)
    fr_rg = fronteira(1)
    t1 = txt.split("Fronteira de Pareto")[1].split("Fronteira por nº de regras")
    c_it = re.findall(r"([A-Za-z0-9_+]+) \((\d+); ([\d,]+)\)", t1[0])
    c_rg = re.findall(r"([A-Za-z0-9_+]+) \((\d+); ([\d,]+)\)", t1[1])
    ok_fi = [c[0] for c in c_it] == fr_it and all(int(c[1]) == rows[c[0]][0] and r_ok(rows[c[0]][2], float(c[2].replace(",", ".")), 2)
                                                   for c in c_it)
    ok_fr = [c[0] for c in c_rg] == fr_rg and all(int(c[1]) == rows[c[0]][1] and r_ok(rows[c[0]][2], float(c[2].replace(",", ".")), 2)
                                                   for c in c_rg)
    amostra = MC_C55
    mcs = {c: mc_ok(c, rows[c][2]) for c in amostra}
    ok = ok and ok_fi and ok_fr and all(m[0] for m in mcs.values()) and len(amostra) >= 8
    reg("C55", ok, "as 38 colas avaliadas exatamente (Bellman de avaliação) + itens/regras recalculados da definição + fronteiras de Pareto "
        f"refeitas + MC literal 1e6 em {len(amostra)} colas", 10 ** 6,
        f"{len(ids)} colas distintas; máx|valor - afirmado| = {dif:.2e}; fronteira por itens {len(fr_it)} colas, por regras {len(fr_rg)} colas",
        f"MC dentro do IC {sum(m[0] for m in mcs.values())}/{len(mcs)}: " + "; ".join(f"{c} {m[1]:.2f}" for c, m in mcs.items()),
        f"fronteiras == afirmadas: itens {ok_fi}, regras {ok_fr}")


# ---------------------------------------------------------------- C56, C57, C58
def flow_flag(outs):
    """Distribuição da pontuação final separada por 'General feito' (pontos > 0): (2, L)."""
    F = np.zeros((1024, 2, L))
    F[1023, 0, 0] = 1.0
    for mask in range(1023, 0, -1):
        for c in range(10):
            if mask >> c & 1:
                o = outs[mask, c]
                nxt = mask ^ (1 << c)
                if c == 9:
                    o0 = np.zeros_like(o)
                    o0[0] = o[0]
                    F[nxt, 0] += np.convolve(F[mask, 0], o0)[:L]
                    F[nxt, 1] += np.convolve(F[mask, 0], o - o0)[:L]
                else:
                    for g in (0, 1):
                        F[nxt, g] += np.convolve(F[mask, g], o)[:L]
    return F[0]


def cond_stats(F2):
    out = []
    x = np.arange(L)
    for g in (1, 0):
        w = F2[g].sum()
        p = F2[g] / w
        m = (p * x).sum()
        out.append(dict(prob=w, mean=m, sd=float(np.sqrt((p * (x - m) ** 2).sum())), moda=int(p.argmax())))
    return out


def chi_joint(flag, tot, F2, N):
    ob, ex, oa, ea, b = {}, {}, 0, 0.0, 0
    for g in (0, 1):
        obs = np.bincount(tot[flag == g].astype(int), minlength=L)
        for x in range(L):
            oa += obs[x]
            ea += F2[g, x] * N
            if ea >= 20:
                ob[b], ex[b] = oa, ea / N
                b += 1
                oa, ea = 0, 0.0
    ob[b - 1] += oa
    ex[b - 1] += ea / N
    s = sum(ex.values())
    return chi2_distribution(ob, {k: v_ / s for k, v_ in ex.items()})


def valida_c56(cx, MCB):
    exp = {"opt": [(0.274993, 182.4807, 197, 23.5180), (0.725007, 133.1677, 146, 22.1491)],
           "gul": [(0.298316, 156.8562, 150, 21.6172), (0.701684, 105.2786, 100, 22.4198)]}
    Fo2 = flow_flag(cx["outs"])
    Fg2 = flow_flag(cx["outsg"])
    ok = True
    notas = []
    ics = []
    for nome, F2, B in (("opt", Fo2, MCB["opt"]), ("gul", Fg2, MCB["gul"])):
        cs = cond_stats(F2)
        tot = B["tot"]
        flag = (B["pts"][:, 9] > 0).astype(int)
        N = len(tot)
        for (pr, mu, mo, sd), e in zip(((c["prob"], c["mean"], c["moda"], c["sd"]) for c in cs), exp[nome]):
            ok &= r_ok(pr, e[0], 6) and r_ok(mu, e[1], 4) and mo == e[2] and r_ok(sd, e[3], 4)
        chi = chi_joint(flag, tot, F2, N)
        pf = check_proportion(flag == 1, cs[0]["prob"])
        lo, hi = t_ci(tot[flag == 1])
        lo0, hi0 = t_ci(tot[flag == 0])
        ok &= chi["status"] == "validada" and pf.agrees and lo <= cs[0]["mean"] <= hi and lo0 <= cs[1]["mean"] <= hi0
        mix = np.abs((F2[0] + F2[1]) - (cx["Fo"] if nome == "opt" else cx["Fg"])).max()
        ok &= mix < 1e-12
        notas.append(f"{nome}: feito P={cs[0]['prob']:.6f} média {cs[0]['mean']:.4f} moda {cs[0]['moda']} dp {cs[0]['sd']:.4f}; "
                     f"não feito P={cs[1]['prob']:.6f} média {cs[1]['mean']:.4f} moda {cs[1]['moda']} dp {cs[1]['sd']:.4f}; "
                     f"chi2 conjunto p={chi['p_value']:.3f}; mistura vs C20/C32 {mix:.1e}")
        ics.append(f"{nome} P(feito) MC {pf.estimate:.4f} [{pf.ci_low:.4f}; {pf.ci_high:.4f}]")
        if nome == "opt":
            d_opt = cs[0]["mean"] - cs[1]["mean"]
            v56 = cs[0]["mean"]
        else:
            d_gul = cs[0]["mean"] - cs[1]["mean"]
    ok &= r_ok(d_opt, 49.31, 2) and r_ok(d_gul, 51.58, 2) and abs(v56 - cval("C56")) < 1e-6
    reg("C56", ok, "fluxo exato de probabilidade com indicador 'General feito' + MC literal 1e6 (ótima e gulosa) com qui-quadrado conjunto",
        10 ** 6, " | ".join(notas), "; ".join(ics), f"distância entre componentes: ótima {d_opt:.2f}, gulosa {d_gul:.2f}")


def fill_stats(dec):
    P, MK, reach = visits(dec)
    pc = np.array([bin(m).count("1") for m in range(1024)])
    livre = np.zeros((12, 10))  # livre[t, b] = P(b livre no início da rodada t)
    for t in range(1, 11):
        for b in range(10):
            sel = (pc == 11 - t) & (np.arange(1024) >> b & 1 == 1)
            livre[t, b] = reach[sel].sum()
    fill = np.zeros((11, 10))  # fill[t, b] = P(b preenchida na rodada t)
    for t in range(1, 11):
        fill[t] = livre[t] - livre[t + 1]
    media = (np.arange(11)[:, None] * fill).sum(0)
    return dict(P=P, MK=MK, reach=reach, livre=livre, fill=fill, media=media)


def valida_c57(cx, MCB):
    FS = {"opt": fill_stats(cx["dec"]), "gul": fill_stats(cx["decg"]), "cola": fill_stats(MCB["cola_dec"])}
    exp = {"opt": [5.1130, 5.1897, 5.4155, 5.2766, 5.5548, 5.5918, 4.4424, 6.4937, 4.4129, 7.5097],
           "gul": [6.5945, 5.8752, 5.2428, 4.6822, 4.1576, 3.5803, 4.1474, 8.0612, 3.8795, 8.7794],
           "cola": [4.3987, 5.6124, 6.0168, 5.9949, 5.9810, 5.8273, 4.4555, 5.1806, 4.9900, 6.5429]}
    ok = True
    for k in exp:
        ok &= all(r_ok(FS[k]["media"][b], exp[k][b], 4) for b in range(10))
        ok &= np.allclose(FS[k]["fill"][1:].sum(0), 1.0, atol=1e-12)
    last = {k: FS[k]["livre"][10, 9] for k in FS}
    seg1 = {k: FS[k]["fill"][1, 7] for k in FS}
    ok &= r_ok(last["opt"], 0.234792, 6) and r_ok(last["gul"], 0.735540, 6) and r_ok(last["cola"], 0.055646, 6)
    ok &= r_ok(seg1["opt"], 0.065158, 6) and r_ok(seg1["gul"], 0.032092, 6) and r_ok(seg1["cola"], 0.154607, 6)
    ok &= abs(last["opt"] - cval("C22")) < 1e-6 and abs(last["gul"] - cval("C57")) < 1e-9
    # MC literal
    nm = 0
    nt = 0
    detalhes = []
    for k in FS:
        B = MCB[k]
        for b in range(10):
            lo, hi = t_ci(B["fillr"][:, b])
            nt += 1
            nm += int(lo <= FS[k]["media"][b] <= hi)
        pl = check_proportion(B["fillr"][:, 9] == 10, last[k])
        ps = check_proportion(B["fillr"][:, 7] == 1, seg1[k])
        nt += 2
        nm += int(pl.agrees) + int(ps.agrees)
        detalhes.append(f"{k}: P(Gen na 10ª) MC {pl.estimate:.4f} [{pl.ci_low:.4f}; {pl.ci_high:.4f}], P(Seg na 1ª) MC {ps.estimate:.4f} "
                        f"[{ps.ci_low:.4f}; {ps.ci_high:.4f}]")
    ok = ok and nm == nt
    reg("C57", ok, "fluxo exato (visitação por cartela) + MC literal 1e6 (ótima, gulosa, cola ii) registrando a rodada de preenchimento",
        10 ** 6, "; ".join(f"{k}: " + ", ".join(f"{x:.4f}" for x in FS[k]["media"]) for k in FS) +
        f"; P(General 10ª) ótima {last['opt']:.6f}, gulosa {last['gul']:.6f}, cola {last['cola']:.6f}; "
        f"P(Seguida 1ª) {seg1['opt']:.6f}, {seg1['gul']:.6f}, {seg1['cola']:.6f}",
        f"MC dentro do IC99 em {nm}/{nt} verificações: " + " | ".join(detalhes), "")


def valida_c58(cx, MCB):
    V = cx["V"]
    reach = visits(cx["dec"])[2]
    pc = np.array([bin(m).count("1") for m in range(1024)])
    def preco(t):
        out = []
        for b in range(10):
            sel = (pc == 11 - t) & (np.arange(1024) >> b & 1 == 1)
            w = reach[sel]
            out.append((w * (V[sel] - V[np.arange(1024)[sel] ^ (1 << b)])).sum() / w.sum())
        return np.array(out)
    p1, p5, p10 = preco(1), preco(5), preco(10)
    e1 = [9.4604, 11.3601, 13.5537, 15.9518, 18.4901, 21.0531, 20.0442, 22.4254, 35.4157, 15.1538]
    e5 = [8.3522, 10.1469, 12.3399, 14.6730, 17.1989, 19.7529, 17.4859, 19.0926, 30.0158, 10.8958]
    singles = np.array([V[1 << c] for c in range(10)])
    ok = (all(r_ok(p1[b], e1[b], 4) for b in range(10)) and all(r_ok(p5[b], e5[b], 4) for b in range(10))
          and np.abs(p10 - singles).max() < 1e-9 and int(p1.argmax()) == 8 and abs(p1[8] - cval("C58")) < 1e-9)
    B = MCB["opt"]
    nm = 0
    for t, pe in ((5, p5), (10, p10)):
        fh = B["freeh"][:, t - 1].astype(np.int64)
        for b in range(10):
            sel = (fh >> b & 1) == 1
            vals = V[fh[sel]] - V[fh[sel] ^ (1 << b)]
            lo, hi = t_ci(vals)
            nm += int(lo - 1e-9 <= pe[b] <= hi + 1e-9)
    # rodada 1: determinístico
    ok = ok and nm == 20 and np.abs(p1 - np.array([V[1023] - V[1023 ^ (1 << b)] for b in range(10)])).max() < 1e-12
    reg("C58", ok, "preço marginal = E[V*(S_t) - V*(S_t sem b) | b livre] com a distribuição de S_t por fluxo exato + MC literal 1e6 (S_t amostrado)",
        10 ** 6, "r1: " + ", ".join(f"{x:.4f}" for x in p1) + "; r5: " + ", ".join(f"{x:.4f}" for x in p5) +
        f"; r10 == C10-C14 (máx|dif| {np.abs(p10 - singles).max():.1e})", f"MC dentro do IC99 em {nm}/20 (rodadas 5 e 10)",
        f"maior preço na rodada 1: casa {CASAS[int(p1.argmax())]} {p1.max():.6f}")


# ---------------------------------------------------------------- C59
def valida_c59(cx):
    V, dec, decg = cx["V"], cx["dec"], cx["decg"]
    soma_S = sum(bin(m).count("1") for m in range(1, 1024))
    marcar = 3 * 252 * soma_S
    subm = 0
    for h in HANDS:
        conj = set()
        for j in range(5):
            conj |= set(itertools.combinations(sorted(h), j))
        subm += len(conj)
    guardar = 2 * 1023 * subm
    total = marcar + guardar
    # contagem direta nos conjuntos de ações do meu modelo (entradas finitas de action_matrices)
    cont = 0
    for mask in range(1, 1024):
        for A in action_matrices(mask, V, True):
            cont += int(np.isfinite(A).sum())
    Vg, _ = bellman_eval(decg)
    Vo, _ = bellman_eval(dec)
    vcola = cola_val(cola_id("2face_8_2", R4))
    ac = {}
    for nome, dd in (("ótima", dec), ("gulosa", decg), ("cola(ii)", cola_dec(PRECOS["2face_8_2"], R4))):
        ac[nome] = visits(dd)[2][1:].min()
    ok = (total == 12292056 == cont and marcar == 3870720 and guardar == 8421336 and subm == 4116 and abs(subm / 252 - 49 / 3) < 1e-12
          and abs(Vg[1023] - cval("C29")) < 1e-9 and np.abs(Vo - V).max() < 1e-9 and all(a > 0 for a in ac.values()))
    reg("C59", ok, "contagem por enumeração (sub-multiconjuntos de cada mão) + contagem direta das ações finitas da minha DP + "
        "Bellman de avaliação da gulosa e da ótima + alcançabilidade (visitação por fluxo)", None,
        f"{total} pares = {marcar} (marcar) + {guardar} (guardar; {subm} sub-multiconjuntos de 0-4 dados nas 252 mãos, média {subm/252:.4f}); "
        f"contagem direta {cont}; Bellman gulosa {Vg[1023]:.10f}; máx|Bellman(ótima) - E*| = {np.abs(Vo - V).max():.1e}",
        "exato", "menor prob. de alcançar uma cartela não vazia: " + ", ".join(f"{k} {v_:.3e}" for k, v_ in ac.items()))


MC_C55 = ["gulosa+risca_gen", "2face_comb0", "gulosa+alvo", "gulosa+seg4x+quad_gen", "2face_8_2+alvo", "num0_8_2", "3face_78112",
          "gulosa+seg4x+quad_gen+alvo+risca_gen", "2face_8_2+seg4x+alvo", "2face_8_2+seg4x+quad_gen+alvo",
          "face_comb0+seg4x+quad_gen+alvo+doispar", "2face_78112+seg4x+quad_gen+alvo+doispar",
          "sozinha_prelim+seg4x+quad_gen+alvo+doispar", "busca_local+seg4x+quad_gen+alvo+doispar",
          "2face_8_2+seg4x+quad_gen+alvo+doispar", "sozinha_int+seg4x+quad_gen+alvo+doispar",
          "2face_8_2+seg4x+quad_gen+alvo+doispar+fu_alto", "sozinha_int+seg4x+quad_gen+alvo+doispar+fu_alto",
          "gulosa+seg4x+quad_gen+alvo+risca_gen+doispar+fu_alto", "gulosa+seg4x+quad_gen+alvo+risca_gen+doispar"]


def lista_mc():
    lst = ["gulosa"] + [cola_id("gulosa", {e}) for e in EM_ORDEM]
    lst += [cola_id("gulosa", em) for (_, _, em) in forward(None, EM_ORDEM)]
    lst += ["2face_8_2", "2face_8_2+alvo", "sozinha_int", "sozinha_int+alvo"]
    lst += [cola_id("sozinha_int", R4), cola_id("2face_8_2", R4)]
    lst += [cola_id("2face_8_2", em) for (_, _, em) in forward(PRECOS["2face_8_2"], ["alvo", "seg4x", "quad_gen", "doispar", "fu_alto"])]
    lst += MC_C55
    return lst


def lista_mc_shapley():
    out = []
    for nm in ("2face_8_2", "sozinha_int"):
        nums, combs = PRECOS[nm][:6], PRECOS[nm][6:]
        out += [(nums + (0,) * 4, frozenset()), ((0,) * 6 + combs, frozenset()), ((0,) * 10, frozenset({"alvo", "seg4x"})),
                (nums + combs, frozenset({"alvo", "seg4x", "quad_gen"}))]
    out += [(PRECOS["busca_local"], R4)]
    return out


# ---------------------------------------------------------------- verificação do jogador (escalar x vetorizado)
def valida_jogador(n_por=1500, seed=11):
    """Compara o jogador de referência (escalar, texto das colas) com o vetorizado em estados aleatórios."""
    import random
    rnd = random.Random(seed)
    combos = [frozenset(c) for k in range(7) for c in itertools.combinations(EM_ORDEM, k)]
    bad = tot = 0
    for tb in (None, "2face_8_2", "sozinha_int", "num0_8_2", "face_comb0"):
        pr = None if tb is None else PRECOS[tb]
        for em in combos:
            masks = np.array([rnd.randint(1, 1023) for _ in range(n_por)])
            rs = np.array([rnd.randint(1, 3) for _ in range(n_por)])
            hands = [tuple(rnd.randint(1, 6) for _ in range(5)) for _ in range(n_por)]
            for r in (1, 2, 3):
                idx = np.flatnonzero(rs == r)
                if not len(idx):
                    continue
                casa, kc = cola_core(r, np.array([hands[i] for i in idx], dtype=np.int8), masks[idx], pr, em)
                for j, i in enumerate(idx):
                    livres = {c for c in range(10) if masks[i] >> c & 1}
                    ref = ref_jogador(r, hands[i], livres, list(pr) if pr else None, em)
                    vec = (("marca", int(casa[j])) if casa[j] >= 0 else
                           ("guarda", tuple(sorted(f + 1 for f in range(6) for _ in range(kc[j, f])))))
                    tot += 1
                    bad += int(ref != vec)
    return tot, bad


# ---------------------------------------------------------------- driver v2
def tudo_v2():
    t0 = time.time()
    print("v2: construindo DP e tabelas...", flush=True)
    cx = ctx_v2()
    cx["Fo"] = flow(cx["outs"])[0]
    cx["Fg"] = flow(cx["outsg"])[0]
    print(f"  pronto {time.time()-t0:.1f}s", flush=True)
    tot, bad = valida_jogador()
    print(f"  jogador escalar x vetorizado: {tot} estados, {bad} diferenças ({time.time()-t0:.0f}s)", flush=True)
    assert bad == 0, "jogador vetorizado diverge do escalar"
    dg = cola_dec(None, frozenset())
    assert (dg[:, 1:] == cx["decg"][:, 1:]).all()
    print("  cola vazia == gulosa G1-G4 (tabelas idênticas)", flush=True)
    print("  MC literal das colas (paralelo)...", flush=True)
    mc_colas(lista_mc(), 10 ** 6)
    mc_colas(lista_mc_shapley(), 10 ** 6)
    print(f"  MC das colas pronto ({time.time()-t0:.0f}s)", flush=True)
    valida_c39_c41(cx)
    print(f"  C39-41 ({time.time()-t0:.0f}s)", flush=True)
    valida_c42()
    valida_c43()
    valida_c44()
    valida_c45_c46(cx)
    valida_c47()
    valida_c48_c49()
    valida_c50()
    valida_c51()
    print(f"  C42-C51 ({time.time()-t0:.0f}s)", flush=True)
    p_fit = valida_c53(cx)
    valida_c52(cx, p_fit)
    valida_c54(cx)
    valida_c55(cx)
    print(f"  C52-C55 ({time.time()-t0:.0f}s)", flush=True)
    colad = cola_dec(PRECOS["2face_8_2"], R4)
    MCB = {"opt": mc_policy(table_decider(cx["dec"]), 10 ** 6, make_rng(SEED_V2 + 100)),
           "gul": mc_policy(greedy_decider, 10 ** 6, make_rng(SEED_V2 + 101)),
           "cola": mc_policy(cola_decider(PRECOS["2face_8_2"], R4), 10 ** 6, make_rng(SEED_V2 + 102)),
           "cola_dec": colad}
    print(f"  MC das políticas pronto ({time.time()-t0:.0f}s)", flush=True)
    valida_c56(cx, MCB)
    valida_c57(cx, MCB)
    valida_c58(cx, MCB)
    valida_c59(cx)
    el = time.time() - t0
    print(f"\nTempo v2: {el:.0f}s")
    path = WORK / "validacao_resultados.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(RES)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for cid in sorted(RES, key=lambda s: int(s[1:])):
        print(cid, "OK" if RES[cid]["ok"] else "DIVERGENTE")


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


def escrever_claims(apenas_v2=False):
    import claims as cm
    data = cm.load(str(WORK.parent))
    res = json.loads((WORK / "validacao_resultados.json").read_text(encoding="utf-8"))
    if apenas_v2:  # não mexe em C1-C35
        res = {k: r for k, r in res.items() if int(k[1:]) >= 39}
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
    # uso:  --v2 [--escrever]  roda só a seção v2 (C39-C59); --so-escrever [--v2] só grava claims.yaml
    #       --v2-c51-c52  refaz só C51 e C52 (e funde no json)
    v2 = "--v2" in sys.argv
    if "--v2-c51-c52" in sys.argv:
        cx_ = ctx_v2()
        mc_colas([(PRECOS["busca_local"], R4)], 10 ** 6)
        valida_c51()
        valida_c52(cx_, valida_c53(cx_))
        pth = WORK / "validacao_resultados.json"
        old_ = json.loads(pth.read_text(encoding="utf-8"))
        old_.update({k: RES[k] for k in ("C51", "C52")})
        pth.write_text(json.dumps(old_, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    elif "--so-escrever" in sys.argv:
        escrever_claims(apenas_v2=v2)
    else:
        (tudo_v2 if v2 else tudo)()
        if "--escrever" in sys.argv:
            escrever_claims(apenas_v2=v2)
