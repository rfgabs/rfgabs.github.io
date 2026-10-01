"""Derivação: Estratégia, Bozó e Cadeias de Markov (A + B + C + D1).

Recalcula todo número de modelo.md e confere claims.yaml.

    uv run python posts/estrategia-bozo-markov/_work/derivacao.py
    uv run python posts/estrategia-bozo-markov/_work/derivacao.py --refazer-rl   # retreina o RL (cache)

Regras: pauta.md (fonte Piano & Toillier 2010 + decisões D1–D6, E1–E4).
Hipóteses de modelagem H1–H6: modelo.md.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import factorial
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[2]
sys.path.insert(0, str(RAIZ / "tools"))
RES = AQUI / "resultados"

# ---------------------------------------------------------------------------
# 0. Dados como multiconjuntos
# ---------------------------------------------------------------------------
FACES = range(1, 7)
KEEPS = [tuple(m) for n in range(6) for m in combinations_with_replacement(FACES, n)]  # 462
KIDX = {m: i for i, m in enumerate(KEEPS)}
MS5 = [m for m in KEEPS if len(m) == 5]  # 252
IDX5 = {m: i for i, m in enumerate(MS5)}
K5 = np.array([KIDX[m] for m in MS5])  # índice em KEEPS de cada mão de 5
ND = len(MS5)


def n_ordenacoes(m) -> int:
    """Número de sequências ordenadas que formam o multiconjunto m."""
    w = factorial(len(m))
    for v in Counter(m).values():
        w //= factorial(v)
    return w


P0_CNT = [n_ordenacoes(m) for m in MS5]  # soma 6^5
assert sum(P0_CNT) == 6**5

# Transição de "guardar k": T[k][d'] = (nº de sequências do relançamento que completam k em d') / 6^(5-|k|)
# Guardamos inteiros escalados por 6^5: TINT[k] = [(d', cnt * 6^|k|)].
TROWS = []
for k in KEEPS:
    c = Counter()
    for m in combinations_with_replacement(FACES, 5 - len(k)):
        c[IDX5[tuple(sorted(k + m))]] += n_ordenacoes(m)
    TROWS.append(sorted(c.items()))
TINT = [[(j, cnt * 6 ** len(k)) for j, cnt in row] for k, row in zip(KEEPS, TROWS)]
assert all(sum(v for _, v in r) == 6**5 for r in TINT)
TF = np.zeros((len(KEEPS), ND))
for i, r in enumerate(TINT):
    for j, v in r:
        TF[i, j] = v / 6**5
P0F = np.array(P0_CNT) / 6**5


def submultis(m):
    c = Counter(m)
    fs = sorted(c)
    out = set()
    for cs in product(*[range(c[f] + 1) for f in fs]):
        out.add(tuple(sorted(sum(([f] * n for f, n in zip(fs, cs)), []))))
    return out


def pref_guarda(k):
    """Ordem de preferência no desempate entre guardas de mesmo valor (H6):
    mais dados guardados; depois maior soma; depois maior tupla em ordem decrescente."""
    return (-len(k), -sum(k), tuple(-x for x in sorted(k, reverse=True)))


# guardas próprias (exclui guardar os 5: relançar zero = parar, H3), já ordenadas por preferência
SUBS = [sorted((s for s in submultis(m) if len(s) < 5), key=pref_guarda) for m in MS5]
SUBS_I = [[KIDX[s] for s in ss] for ss in SUBS]
GUARDAS_USADAS = sorted({k for ss in SUBS_I for k in ss})

# ---------------------------------------------------------------------------
# Casas e pontuação
# ---------------------------------------------------------------------------
NOMES = ["Ás", "Duque", "Terno", "Quadra", "Quina", "Sena", "Fú", "Seguida", "Quadrada", "General"]
NB = 10
TODAS = list(range(NB))
BASE = {6: 20, 7: 30, 8: 40, 9: 50}


def valida(b, m) -> bool:
    perfil = sorted(Counter(m).values())
    if b == 6:
        return perfil == [2, 3]  # D4
    if b == 7:
        return m in ((1, 2, 3, 4, 5), (2, 3, 4, 5, 6))  # D5
    if b == 8:
        return perfil == [1, 4]  # D4
    if b == 9:
        return perfil == [5]
    raise ValueError


def pontos(b, m, boca: bool) -> int:
    if b < 6:
        return (b + 1) * m.count(b + 1)
    return BASE[b] + (5 if boca else 0) if valida(b, m) else 0  # D6: risco = 0


SC = [[pontos(b, m, False) for m in MS5] for b in range(NB)]
SCB = [[pontos(b, m, True) for m in MS5] for b in range(NB)]
SCF = np.array(SC)
SCBF = np.array(SCB)


def caixas(S):
    return [b for b in range(NB) if S >> b & 1]


def escala(n):
    """Denominador comum dos valores E(S) com |S| = n: 6^(15n)."""
    return 6 ** (15 * n)


# ---------------------------------------------------------------------------
# DP exata (inteiros escalados) — ótima ou avaliação de política fixa
# ---------------------------------------------------------------------------
# Política por cartela: tupla (A1, A2, A3), cada uma uma lista de 252 ações;
# ação >= 0: guardar KEEPS[ação]; ação < 0: marcar a casa -(ação+1).

def _matvec(V):
    """W[k] = sum_d' TINT[k][d'] V[d'] para as guardas usadas (escala x 6^5)."""
    W = {}
    for k in GUARDAS_USADAS:
        W[k] = sum(c * V[j] for j, c in TINT[k])
    return W


def rodada_exata(S, E, boca=True, pol=None, detalhe=False, alt=False):
    """Uma rodada a partir da cartela S. E: dict cartela -> valor escalado por escala(|cartela|).
    Retorna (E(S) escalado por escala(|S|), política (A1, A2, A3)[, detalhes]).
    alt=True usa o desempate oposto (relançar > marcar; casa de maior índice; guarda menos preferida)."""
    bs = caixas(S)
    if alt:
        bs = bs[::-1]
    n = len(bs)
    Dp = escala(n - 1)
    fut = {b: E[S & ~(1 << b)] for b in bs}
    sc1 = SCB if boca else SC
    # marcar, sem e com boca (escala Dp)
    M = {b: [SC[b][d] * Dp + fut[b] for d in range(ND)] for b in bs}
    MB = {b: [sc1[b][d] * Dp + fut[b] for d in range(ND)] for b in bs}

    def melhor_marca(Mx, d, fator):
        best, arg = None, None
        for b in bs:  # desempate: casa de menor índice (H6)
            v = Mx[b][d] * fator
            if best is None or v > best:
                best, arg = v, b
        return best, arg

    A3, V3 = [], []
    for d in range(ND):
        if pol is None:
            v, b = melhor_marca(M, d, 1)
        else:
            b = -(pol[2][d] + 1)
            v = M[b][d]
        V3.append(v)
        A3.append(-(b + 1))

    def nivel(Vprox, Mx, fator, idx):
        W = _matvec(Vprox)
        A, V = [], []
        for d in range(ND):
            if pol is None:
                vm, b = melhor_marca(Mx, d, fator)
                vk, kb = None, None
                for k in (SUBS_I[d][::-1] if alt else SUBS_I[d]):
                    w = W[k]
                    if vk is None or w > vk:
                        vk, kb = w, k
                if (vm > vk) if alt else (vm >= vk):  # desempate: marcar (H6)
                    A.append(-(b + 1)); V.append(vm)
                else:
                    A.append(kb); V.append(vk)
            else:
                a = pol[idx][d]
                A.append(a)
                V.append(Mx[-(a + 1)][d] * fator if a < 0 else W[a])
        return A, V, W

    A2, V2, W1 = nivel(V3, M, 6**5, 1)
    A1, V1, W2 = nivel(V2, MB, 6**10, 0)
    ES = sum(c * v for c, v in zip(P0_CNT, V1))
    out = (ES, (A1, A2, A3))
    if detalhe:
        out = out + ({"V3": V3, "W1": W1, "V2": V2, "W2": W2, "V1": V1, "M": M, "MB": MB},)
    return out


def subconjuntos_ordenados(universo):
    mask = sum(1 << b for b in universo)
    Ss = [S for S in range(1 << NB) if S & ~mask == 0 and S]
    return sorted(Ss, key=lambda S: (bin(S).count("1"), S))


def resolve_exato(universo, boca=True, politica=None, alt=False):
    """E[S] (inteiro escalado) e política para toda cartela S contida em `universo`.
    politica: None (ótima) ou dict S -> (A1, A2, A3) a avaliar."""
    E = {0: 0}
    POL = {}
    for S in subconjuntos_ordenados(universo):
        ES, P = rodada_exata(S, E, boca=boca, pol=None if politica is None else politica[S], alt=alt)
        E[S], POL[S] = ES, P
    return E, POL


def frac(E, S):
    return Fraction(E[S], escala(bin(S).count("1")))


# ---------------------------------------------------------------------------
# Política gulosa (B), definida em modelo.md (G1–G4)
# ---------------------------------------------------------------------------

def _mais_freq(m):
    c = Counter(m)
    f = max(c, key=lambda v: (c[v], v))  # G4: face mais frequente, a maior no empate
    return f, c[f]


GUL_C = np.array([_mais_freq(m)[1] for m in MS5])
GUL_K = np.array([KIDX[(_mais_freq(m)[0],) * _mais_freq(m)[1]] for m in MS5])


def politica_gulosa(S):
    """Tabelas (A1, A2, A3) da heurística gulosa G1–G4 na cartela S."""
    bs = np.array(caixas(S))
    A = []
    for r in range(3):
        pts = (SCBF if r == 0 else SCF)[bs]  # G1
        bh = bs[pts.argmax(0)]  # G2: maior pontuação agora; empate -> menor índice (argmax pega o 1º)
        pm = pts.max(0)
        para = (r == 2) | ((bh >= 6) & (pm > 0)) | (GUL_C == 5)  # G3
        A.append(np.where(para, -(bh + 1), GUL_K).tolist())
    return tuple(A)


# ---------------------------------------------------------------------------
# Simulação Monte Carlo vetorizada de uma política tabelada
# ---------------------------------------------------------------------------
CODE2MS = np.zeros(6**5, dtype=np.int64)
for _t in product(FACES, repeat=5):
    CODE2MS[sum((x - 1) * 6**i for i, x in enumerate(_t))] = IDX5[tuple(sorted(_t))]
POT6 = 6 ** np.arange(5)
KEEPDICE = np.zeros((len(KEEPS), 5), dtype=np.int64)
for _i, _k in enumerate(KEEPS):
    KEEPDICE[_i, : len(_k)] = _k


def tabela(POL):
    """dict S -> (A1, A2, A3)  ==>  array int16 [cartela, lançamento, mão]."""
    T = np.zeros((1 << NB, 3, ND), dtype=np.int16)
    for S, P in POL.items():
        T[S] = np.array(P)
    return T


def simula(TAB, universo, n, rng, boca=True):
    """Joga n partidas seguindo a tabela TAB. Retorna (pontuação final, matriz casa>0)."""
    S = np.full(n, sum(1 << b for b in universo), dtype=np.int64)
    total = np.zeros(n, dtype=np.int64)
    cheia = np.zeros((n, NB), dtype=bool)
    lin = np.arange(n)
    for _ in range(len(universo)):
        dados = rng.integers(1, 7, (n, 5))
        ativo = np.ones(n, dtype=bool)
        for r in range(3):
            ms = CODE2MS[((dados - 1) * POT6).sum(1)]
            a = TAB[S, r, ms].astype(np.int64)
            mk = ativo & (a < 0)
            b = -(a[mk] + 1)
            pts = (SCBF if (r == 0 and boca) else SCF)[b, ms[mk]]
            total[mk] += pts
            cheia[lin[mk], b] = pts > 0
            S[mk] &= ~(1 << b)
            ativo &= ~mk
            if r < 2 and ativo.any():
                kd = KEEPDICE[a[ativo]]
                novo = rng.integers(1, 7, kd.shape)
                dados[ativo] = np.where(kd > 0, kd, novo)
        assert not ativo.any()
    assert (S == 0).all()
    return total, cheia


# ---------------------------------------------------------------------------
# A: probabilidades exatas (frações)
# ---------------------------------------------------------------------------
TFR = [[(j, Fraction(c, 6 ** (5 - len(k)))) for j, c in row] for k, row in zip(KEEPS, TROWS)]
P0FR = [Fraction(c, 6**5) for c in P0_CNT]


def p_max_alvo(ok):
    """Máxima probabilidade de terminar a rodada (até 3 lançamentos) com uma mão em `ok`
    (lista booleana por mão), parando assim que conseguir. Recursão exata em frações."""
    V = [Fraction(int(o)) for o in ok]
    for _ in range(2):
        W = {k: sum(p * V[j] for j, p in TFR[k]) for k in GUARDAS_USADAS}
        V = [Fraction(1) if ok[d] else max(W[k] for k in SUBS_I[d]) for d in range(ND)]
    return sum(p * v for p, v in zip(P0FR, V))


def cadeia_general():
    """Cadeia de Markov do nº máximo de dados iguais (1..5) sob 'guardar a face mais
    frequente e relançar o resto' (com 1 dado guardado se todos diferem). Exata."""
    from mathbox import R, transition_matrix
    trans = {}
    for i in range(1, 5):
        c = Counter()
        for rol in product(FACES, repeat=5 - i):
            m = (1,) * i + rol
            c[max(Counter(m).values())] += 1
        trans[i] = {j: R(v, 6 ** (5 - i)) for j, v in c.items()}
    trans[5] = {5: 1}
    estados = [1, 2, 3, 4, 5]
    P = transition_matrix(estados, trans)
    c0 = Counter(max(Counter(t).values()) for t in product(FACES, repeat=5))
    pi0 = [R(c0[i], 6**5) for i in estados]
    return estados, P, pi0


PMAX = 400  # pontuação máxima possível é 265


def saida_rodada(S, P, boca=True):
    """out[b, s] = P(marcar a casa b com s pontos nesta rodada | início em S), sob a política P=(A1,A2,A3)."""
    out = np.zeros((NB, 61))
    m = P0F.copy()
    for r in range(3):
        A = np.array(P[r])
        sc = SCBF if (r == 0 and boca) else SCF
        marca = A < 0
        bsel = -(A[marca] + 1)
        np.add.at(out, (bsel, sc[bsel, np.nonzero(marca)[0]]), m[marca])
        if r < 2:
            km = np.bincount(A[~marca], weights=m[~marca], minlength=len(KEEPS))
            m = km @ TF
    return out


def para_frente(POL, universo, boca=True):
    """Distribuição exata (float64) da pontuação final, P(casa termina > 0) e P(alcançar cada cartela)."""
    full = sum(1 << b for b in universo)
    dist = {full: np.zeros(PMAX)}
    dist[full][0] = 1.0
    alcance = {}
    completa = np.zeros(NB)
    for S in sorted(subconjuntos_ordenados(universo), key=lambda S: -bin(S).count("1")):
        if S not in dist:
            continue
        dS = dist.pop(S)
        pS = dS.sum()
        alcance[S] = pS
        out = saida_rodada(S, POL[S], boca)
        for b in caixas(S):
            S2 = S & ~(1 << b)
            for s in np.nonzero(out[b])[0]:
                q = out[b, s]
                if s > 0:
                    completa[b] += pS * q
                acc = dist.setdefault(S2, np.zeros(PMAX))
                acc[s:] += q * dS[: PMAX - s]
    return dist[0], completa, alcance


# ---------------------------------------------------------------------------
# Ferramentas de análise da política
# ---------------------------------------------------------------------------

def q_exato(S, E, r, mao):
    """Valores exatos (Fraction) de cada ação na cartela S, lançamento r (1..3), mão `mao`."""
    d = IDX5[tuple(sorted(mao))]
    _, P, det = rodada_exata(S, E, detalhe=True)
    Dp = escala(bin(S).count("1") - 1)
    fator = {1: 6**10, 2: 6**5, 3: 1}[r]
    Mx = det["MB"] if r == 1 else det["M"]
    q = {("marca", NOMES[b]): Fraction(Mx[b][d], Dp) for b in caixas(S)}
    if r < 3:
        W = det["W2"] if r == 1 else det["W1"]
        for k in SUBS_I[d]:
            q[("guarda", KEEPS[k])] = Fraction(W[k], Dp * fator)
    a = P[r - 1][d]
    escolha = ("marca", NOMES[-(a + 1)]) if a < 0 else ("guarda", KEEPS[a])
    return q, escolha


def conta_empates(E, universo):
    """Nº de situações de decisão (cartela, lançamento, mão) com mais de uma ação ótima."""
    n_emp = 0
    for S in subconjuntos_ordenados(universo):
        _, _, det = rodada_exata(S, E, detalhe=True)
        bs = caixas(S)
        for r, (Mx, W, fator) in enumerate([(det["MB"], det["W2"], 6**10), (det["M"], det["W1"], 6**5), (det["M"], None, 1)]):
            for d in range(ND):
                vals = [Mx[b][d] * fator for b in bs]
                if W is not None:
                    vals += [W[k] for k in SUBS_I[d]]
                m = max(vals)
                if sum(v == m for v in vals) > 1:
                    n_emp += 1
    return n_emp


def resumo_dist(p):
    x = np.arange(len(p))
    media = float((x * p).sum())
    dp = float(np.sqrt(((x - media) ** 2 * p).sum()))
    cdf = np.cumsum(p)
    q = lambda a: int(np.searchsorted(cdf, a - 1e-12))
    return {"media": media, "dp": dp, "q05": q(0.05), "mediana": q(0.5), "q95": q(0.95),
            "moda": int(p.argmax()), "min": int(np.nonzero(p > 0)[0][0]), "max": int(np.nonzero(p > 0)[0][-1])}


def acao_txt(a):
    return f"marca {NOMES[-(a + 1)]}" if a < 0 else "guarda " + ("".join(map(str, KEEPS[a])) or "nada")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
NUM: dict = {}


def reg(chave, valor, txt=None):
    NUM[chave] = valor
    if isinstance(valor, Fraction):
        s = f"{valor}" if len(str(valor)) < 60 else f"<fração com {len(str(valor.denominator))} dígitos no denominador>"
        print(f"  {chave:<34} = {float(valor):.12g}   [{s}]" + (f"  {txt}" if txt else ""))
    else:
        print(f"  {chave:<34} = {valor}" + (f"  {txt}" if txt else ""))


def secao_A():
    from mathbox import absorbing_analysis, dice_pattern_probability
    import sympy as sp
    print("\n== A. Uma rodada, um alvo")
    reg("n_maos", ND, "multiconjuntos de 5 dados")
    reg("n_guardas", len(KEEPS), "submulticonjuntos de 0 a 5 dados")
    reg("n_estados_decisao", (2**NB - 1) * 3 * ND, "(cartela não vazia, lançamento 1–3, mão)")
    nomes_c = {6: "fu", 7: "seguida", 8: "quadrada", 9: "general"}
    for b, nm in nomes_c.items():
        ok = [valida(b, m) for m in MS5]
        p1 = sum(p for p, o in zip(P0FR, ok) if o)
        p1b = dice_pattern_probability(5, lambda t, b=b: valida(b, tuple(sorted(t))))
        assert Fraction(int(p1b.p), int(p1b.q)) == p1
        reg(f"p_boca_{nm}", p1)
        reg(f"p3_{nm}", p_max_alvo(ok), "máx. P(fazer em até 3 lançamentos)")
    reg("p_boca_qualquer", sum(NUM[f"p_boca_{nm}"] for nm in nomes_c.values()))
    # cadeia do General
    estados, P, pi0 = cadeia_general()
    NUM["cadeia_P"] = [[str(P[i, j]) for j in range(5)] for i in range(5)]
    print("  matriz de transição (estados 1..5):", NUM["cadeia_P"])
    v = sp.Matrix([pi0]) * P * P
    p3c = Fraction(int(sp.fraction(v[4])[0]), int(sp.fraction(v[4])[1]))
    reg("p3_general_cadeia", p3c)
    assert p3c == NUM["p3_general"], "cadeia != DP"
    ab = absorbing_analysis(P, estados)
    t = ab["t"]
    n_lanc = 1 + sum(pi0[i] * t[i] for i in range(4))
    n_lanc = Fraction(int(sp.fraction(n_lanc)[0]), int(sp.fraction(n_lanc)[1]))
    reg("lancamentos_ate_general", n_lanc, "esperança sem limite de lançamentos")
    # valor de cada casa sozinha (DP exata com uma casa)
    for b in range(NB):
        E1, _ = resolve_exato([b])
        v1 = frac(E1, 1 << b)
        if b < 6:
            assert v1 == Fraction(455 * (b + 1), 216)
        else:
            nm = nomes_c[b]
            assert v1 == BASE[b] * NUM[f"p3_{nm}"] + 5 * NUM[f"p_boca_{nm}"]
        reg(f"sozinha_{b}", v1, NOMES[b])
    reg("soma_sozinhas", sum(NUM[f"sozinha_{b}"] for b in range(NB)))


def secao_C():
    print("\n== C. MDP completo (DP exata em inteiros)")
    t0 = time.perf_counter()
    E, POL = resolve_exato(TODAS)
    full = (1 << NB) - 1
    print(f"  (DP exata: {time.perf_counter() - t0:.1f} s)")
    reg("E_otimo", frac(E, full))
    NUM["E_otimo_exato"] = str(frac(E, full))
    for b in range(NB):  # ponte: DP completa, cartela com uma casa = A
        assert frac(E, 1 << b) == NUM[f"sozinha_{b}"]
    Esb, _ = resolve_exato(TODAS, boca=False)
    reg("E_sem_boca", frac(Esb, full))
    reg("valor_boca", NUM["E_otimo"] - NUM["E_sem_boca"])
    reg("n_empates", conta_empates(E, TODAS), "situações com mais de uma ação ótima")
    dist, comp, alc = para_frente(POL, TODAS)
    assert abs(dist.sum() - 1) < 1e-12 and abs((np.arange(PMAX) * dist).sum() - float(NUM["E_otimo"])) < 1e-9
    for k, v in resumo_dist(dist).items():
        reg(f"otima_{k}", v)
    for b in range(NB):
        reg(f"otima_completa_{b}", float(comp[b]), NOMES[b])
        reg(f"otima_ultima_{b}", float(alc.get(1 << b, 0.0)), f"P({NOMES[b]} é a última casa)")
    # sensibilidade ao desempate: política ótima com o desempate oposto
    Ealt, POLalt = resolve_exato(TODAS, alt=True)
    assert Ealt == E
    dalt, calt, _ = para_frente(POLalt, TODAS)
    reg("desempate_dif_completa", float(np.abs(calt - comp).max()), "máx |ΔP(casa > 0)| com desempate oposto")
    reg("desempate_dif_dist", float(np.abs(dalt - dist).sum() / 2), "distância de variação total da pontuação final")
    ro = resumo_dist(dalt)
    reg("desempate_dif_dp", abs(ro["dp"] - NUM["otima_dp"]))
    # decisões contraintuitivas (valores exatos das ações)
    casos = {
        "p1": (full, 1, (5, 6, 6, 6, 6), ("guarda", (6, 6, 6, 6)), ("marca", "Quadrada")),
        "p2": (full, 2, (1, 1, 4, 4, 4), ("guarda", (4, 4, 4)), ("marca", "Fú")),
        "p2b": (full, 2, (1, 1, 1, 4, 4), ("marca", "Fú"), ("guarda", (1, 1, 1))),
        "p3": (full, 3, (1, 2, 3, 4, 6), ("marca", "Ás"), ("marca", "Sena")),
        "p4": (full, 1, (2, 2, 3, 4, 5), ("guarda", (2, 3, 4, 5)), ("guarda", (2, 2))),
    }
    for nome, (S, r, mao, otima, alt) in casos.items():
        q, esc = q_exato(S, E, r, mao)
        assert esc == otima, (nome, esc)
        assert q[otima] == max(q.values())
        reg(f"{nome}_q_otima", q[otima], f"{otima}")
        reg(f"{nome}_q_alt", q[alt], f"{alt}")
        reg(f"{nome}_vantagem", q[otima] - q[alt])
    # mapa de decisão: cartela vazia, 1º lançamento
    P = POL[full]
    GP = politica_gulosa(full)
    RES.mkdir(exist_ok=True)
    linhas = ["dados,prob_1o_lancamento,otima_lanc1,otima_lanc2,otima_lanc3,gulosa_lanc1"]
    for d, m in enumerate(MS5):
        linhas.append(f"{''.join(map(str, m))},{P0_CNT[d]}/7776,{acao_txt(P[0][d])},{acao_txt(P[1][d])},{acao_txt(P[2][d])},{acao_txt(GP[0][d])}")
    (RES / "mapa_decisao_cartela_vazia.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    # afirmações do mapa (C23, C27): quadra+avulso nunca é marcada no 1º/2º lançamento;
    # com dois pares no 1º lançamento guarda-se só o par mais alto
    for d, m in enumerate(MS5):
        if valida(8, m):
            assert P[0][d] >= 0 and P[1][d] >= 0
        perfil = sorted(Counter(m).values())
        if perfil == [1, 2, 2]:
            alto = max(f for f, c in Counter(m).items() if c == 2)
            assert KEEPS[P[0][d]] == (alto, alto)
    assert sum(1 for m in MS5 if sorted(Counter(m).values()) == [1, 2, 2]) == 60
    for mao, g in {(1, 2, 3, 4, 6): (2, 3, 4), (1, 2, 3, 5, 6): (2, 3, 5), (1, 2, 4, 5, 6): (2, 4, 5),
                   (1, 3, 4, 5, 6): (3, 4, 5)}.items():
        assert KEEPS[P[0][IDX5[mao]]] == g
    para1 = [d for d in range(ND) if P[0][d] < 0]
    assert all(valida(9, MS5[d]) or valida(6, MS5[d]) or valida(7, MS5[d]) for d in para1)
    assert len(para1) == 6 + 30 + 2
    reg("p_para_lanc1_vazia", sum(P0FR[d] for d in para1), "P(ótima marca logo no 1º lançamento, cartela vazia)")
    reg("concorda_gulosa_lanc1", sum(P0FR[d] for d in range(ND) if P[0][d] == GP[0][d]),
        "P(1º lançamento em que ótima e gulosa fazem o mesmo, cartela vazia)")
    # guarda da ótima no 1º lançamento por "tipo" de mão
    return E, POL, dist


def secao_B(E, POL, dist_ot):
    print("\n== B. Heurística gulosa")
    full = (1 << NB) - 1
    GP = {S: politica_gulosa(S) for S in POL}
    EG, _ = resolve_exato(TODAS, politica=GP)
    reg("E_gulosa", frac(EG, full))
    NUM["E_gulosa_exato"] = str(frac(EG, full))
    reg("perda_gulosa", NUM["E_otimo"] - NUM["E_gulosa"])
    reg("perda_gulosa_pct", float(100 * NUM["perda_gulosa"] / NUM["E_otimo"]), "% do ótimo")
    distg, compg, _ = para_frente(GP, TODAS)
    assert abs((np.arange(PMAX) * distg).sum() - float(NUM["E_gulosa"])) < 1e-9
    for k, v in resumo_dist(distg).items():
        reg(f"gulosa_{k}", v)
    for b in range(NB):
        reg(f"gulosa_completa_{b}", float(compg[b]), NOMES[b])
    cdf_g = np.cumsum(distg)
    reg("p_otima_vence", float((dist_ot[1:] * cdf_g[:-1]).sum()), "partidas independentes")
    reg("p_empate", float((dist_ot * distg).sum()))
    np.savez(RES / "distribuicoes.npz", pontos=np.arange(PMAX), otima=dist_ot, gulosa=distg)
    # Monte Carlo vetorizado (semente fixa)
    from montecarlo import mean_interval
    n = 1_000_000
    for nome, POLx, sem in (("gulosa", GP, 20261001), ("otima", POL, 20261002)):
        rng = np.random.default_rng(sem)
        t0 = time.perf_counter()
        tot, cheia = simula(tabela(POLx), TODAS, n, rng)
        lo, hi = mean_interval(tot.astype(float), 0.99)
        print(f"  (MC {nome}: {time.perf_counter() - t0:.1f} s)")
        reg(f"mc_{nome}_media", float(tot.mean()), f"IC99% [{lo:.3f}, {hi:.3f}], n={n:,}, semente {sem}")
        NUM[f"mc_{nome}_ic99"] = [lo, hi]
        alvo = NUM["E_gulosa"] if nome == "gulosa" else NUM["E_otimo"]
        assert lo <= float(alvo) <= hi, f"MC {nome} fora do IC"
        reg(f"mc_{nome}_dp", float(tot.std(ddof=1)))
    return GP


# ---------------------------------------------------------------------------
# D1: Q-learning tabular num Bozó reduzido
# ---------------------------------------------------------------------------
RL_UNIVERSO = [5, 7, 9]  # Sena, Seguida, General (justificativa em modelo.md)
RL_SEMENTES = [1, 2, 3, 4, 5]
RL_ORCAMENTOS = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
RL_EPS = 0.5  # ε fixo no treino (Q-learning é off-policy: aprende Q* mesmo explorando muito)
RL_QINIT = 120.0  # = 30 + 35 + 55, maior retorno possível do jogo reduzido (otimista)
RL_CACHE = RES / "rl_curva.json"


class BozoReduzido:
    """Ambiente para tools/rl.py. Estado (cartela, lançamentos feitos 1..3, índice da mão);
    terminal (0, 0, 0). Ações: -(b+1) marca a casa b; k >= 0 guarda KEEPS[k] e relança o resto."""

    def __init__(self, universo, exploring_starts=False):
        self.U = sum(1 << b for b in universo)
        self.subs = subconjuntos_ordenados(universo)
        self.es = exploring_starts
        self._a = {}

    @staticmethod
    def _sorteia(rng, k=()):
        return IDX5[tuple(sorted(k + tuple(int(x) for x in rng.integers(1, 7, 5 - len(k)))))]

    def reset(self, rng):
        if self.es:  # estado não terminal uniforme
            return (self.subs[rng.integers(len(self.subs))], int(rng.integers(1, 4)), int(rng.integers(ND)))
        return (self.U, 1, self._sorteia(rng))

    def actions(self, s):
        if s[0] == 0:
            return []
        a = self._a.get(s)
        if a is None:
            S, r, d = s
            a = [-(b + 1) for b in caixas(S)] + (SUBS_I[d] if r < 3 else [])
            self._a[s] = a
        return a

    def step(self, s, a, rng):
        S, r, d = s
        if a < 0:
            b = -(a + 1)
            pts = float((SCB if r == 1 else SC)[b][d])
            S2 = S & ~(1 << b)
            if S2 == 0:
                return (0, 0, 0), pts, True
            return (S2, 1, self._sorteia(rng)), pts, False
        return (S, r + 1, self._sorteia(rng, KEEPS[a])), 0.0, False


def _treina(args):
    """Um treino independente: (semente, nº de partidas) -> valor exato da política gulosa em Q."""
    import gc
    from rl import greedy_policy, q_learning
    semente, n = args
    gc.disable()
    t0 = time.perf_counter()
    rng = np.random.default_rng(semente)
    Q, _ = q_learning(BozoReduzido(RL_UNIVERSO, exploring_starts=True), n, rng,
                      epsilon=RL_EPS, q_init=RL_QINIT, alpha="visitas")
    env = BozoReduzido(RL_UNIVERSO)
    subs = subconjuntos_ordenados(RL_UNIVERSO)
    estados = [(S, r, d) for S in subs for r in (1, 2, 3) for d in range(ND)]
    pol = greedy_policy(Q, env, estados)
    P = {S: tuple([pol[(S, r, d)] for d in range(ND)] for r in (1, 2, 3)) for S in subs}
    EL, _ = resolve_exato(RL_UNIVERSO, politica=P)
    v = frac(EL, env.U)
    E, POL = resolve_exato(RL_UNIVERSO)
    iguais = sum(P[S][r][d] == POL[S][r][d] for S in subs for r in range(3) for d in range(ND))
    return {"semente": semente, "partidas": n, "valor": float(v), "valor_exato": str(v),
            "decisoes_iguais_otima": iguais, "decisoes": len(estados), "pares_q": len(Q),
            "segundos": time.perf_counter() - t0}


def secao_D1(refazer=False):
    print("\n== D1. Q-learning tabular no Bozó reduzido", [NOMES[b] for b in RL_UNIVERSO])
    U = sum(1 << b for b in RL_UNIVERSO)
    E, POL = resolve_exato(RL_UNIVERSO)
    reg("rl_E_otimo", frac(E, U))
    GP = {S: politica_gulosa(S) for S in POL}
    EG, _ = resolve_exato(RL_UNIVERSO, politica=GP)
    reg("rl_E_gulosa", frac(EG, U))
    reg("rl_n_estados", len(POL) * 3 * ND, "estados de decisão do jogo reduzido")
    reg("rl_n_pares", sum(ND * (2 * len(caixas(S)) + len(caixas(S))) + 2 * sum(len(s) for s in SUBS_I) for S in POL),
        "pares (estado, ação)")
    if refazer or not RL_CACHE.exists():
        from concurrent.futures import ProcessPoolExecutor
        tarefas = [(s, n) for n in RL_ORCAMENTOS for s in RL_SEMENTES]
        tarefas.sort(key=lambda t: -t[1])
        print(f"  treinando {len(tarefas)} execuções (cache em {RL_CACHE.name})...", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(_treina, tarefas))
        RES.mkdir(exist_ok=True)
        RL_CACHE.write_text(json.dumps({"universo": RL_UNIVERSO, "epsilon": RL_EPS, "q_init": RL_QINIT,
                                        "alpha": "visitas (1/N^0.8)", "exploring_starts": True,
                                        "execucoes": sorted(res, key=lambda r: (r["partidas"], r["semente"]))},
                                       indent=1), encoding="utf-8")
        print(f"  (treino: {time.perf_counter() - t0:.0f} s de relógio)")
    dados = json.loads(RL_CACHE.read_text(encoding="utf-8"))
    assert dados["universo"] == RL_UNIVERSO and dados["epsilon"] == RL_EPS and dados["q_init"] == RL_QINIT
    ot = float(NUM["rl_E_otimo"])
    curva = []
    for n in RL_ORCAMENTOS:
        vs = np.array([r["valor"] for r in dados["execucoes"] if r["partidas"] == n])
        ig = np.array([r["decisoes_iguais_otima"] / r["decisoes"] for r in dados["execucoes"] if r["partidas"] == n])
        assert len(vs) == len(RL_SEMENTES)
        curva.append((n, vs.mean(), vs.std(ddof=1), vs.min(), vs.max(), ig.mean()))
        reg(f"rl_{n}_media", float(vs.mean()), f"dp {vs.std(ddof=1):.3f}, min {vs.min():.3f}, max {vs.max():.3f}, "
            f"gap {ot - vs.mean():.3f}, % ótimo {100 * vs.mean() / ot:.2f}, decisões = ótima {100 * ig.mean():.1f}%")
        NUM[f"rl_{n}_dp"] = float(vs.std(ddof=1))
        NUM[f"rl_{n}_min"] = float(vs.min())
        NUM[f"rl_{n}_max"] = float(vs.max())
        NUM[f"rl_{n}_gap"] = ot - float(vs.mean())
        NUM[f"rl_{n}_iguais"] = float(ig.mean())
    np.savetxt(RES / "rl_curva_aprendizado.csv", np.array(curva), delimiter=",", fmt="%.6f",
               header="partidas_treino,valor_medio,dp,min,max,frac_decisoes_iguais_otima", comments="")


# ---------------------------------------------------------------------------
# Conferência de claims.yaml
# ---------------------------------------------------------------------------
# id da afirmação -> chave em NUM do seu valor_num
CLAIM_CHAVE = {
    "C1": "p_boca_general", "C2": "p_boca_quadrada", "C3": "p_boca_fu", "C4": "p_boca_seguida",
    "C5": "p3_general", "C6": "p3_quadrada", "C7": "p3_fu", "C8": "p3_seguida",
    "C9": "lancamentos_ate_general", "C10": "sozinha_0", "C11": "sozinha_6", "C12": "sozinha_7",
    "C13": "sozinha_8", "C14": "sozinha_9", "C15": "soma_sozinhas", "C16": "n_estados_decisao",
    "C17": "E_otimo", "C18": "E_sem_boca", "C19": "valor_boca", "C20": "otima_dp",
    "C21": "otima_completa_9", "C22": "otima_ultima_9", "C23": "p1_vantagem", "C24": "p2_vantagem",
    "C25": "p3_vantagem", "C26": "p4_vantagem", "C27": "p_para_lanc1_vazia", "C28": "concorda_gulosa_lanc1",
    "C29": "E_gulosa", "C30": "perda_gulosa", "C31": "mc_gulosa_media", "C32": "gulosa_dp",
    "C33": "gulosa_completa_7", "C34": "p_otima_vence", "C35": "n_empates", "C36": "rl_E_otimo",
    "C37": "rl_3000000_media", "C38": "rl_300000_media",
}


def confere_claims():
    import yaml
    dados = yaml.safe_load((AQUI / "claims.yaml").read_text(encoding="utf-8"))
    erros = 0
    print("\n== Conferência de claims.yaml")
    for c in dados["claims"]:
        ch = CLAIM_CHAVE.get(c["id"])
        if ch is None:
            print(f"  {c['id']}: sem chave (qualitativa)")
            continue
        v = NUM[ch]
        vn = c.get("valor_num")
        ok = vn is not None and abs(float(v) - float(vn)) <= 1e-9 * max(1.0, abs(float(v)))
        if isinstance(v, Fraction) and "/" in str(c.get("valor", "")) and len(str(v)) < 400:
            ok = ok and Fraction(str(c["valor"]).replace(" ", "")) == v
        if not ok:
            erros += 1
        print(f"  {c['id']:<4} {'OK ' if ok else 'ERRO'} {ch:<28} derivado {float(v):.12g}  claims {vn}")
    print(f"  {len(dados['claims'])} afirmações, {erros} divergências")
    return erros


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    secao_A()
    E, POL, dist = secao_C()
    secao_B(E, POL, dist)
    secao_D1(refazer="--refazer-rl" in sys.argv)
    RES.mkdir(exist_ok=True)
    saida = {k: ({"exato": str(v), "aprox": float(v)} if isinstance(v, Fraction) else v) for k, v in NUM.items()}
    (RES / "numeros.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    erros = confere_claims()
    print(f"\nTempo total: {time.perf_counter() - t0:.0f} s")
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()

