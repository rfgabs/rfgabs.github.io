"""Derivação v2: de volta à mesa (pauta › Versão 2, escolhas H1–H6).

Chamado por derivacao.py (que roda a v1 e depois `derivacao_v2.main(...)`); não roda sozinho.

Seções:
    §0  lema da diferença de desempenho, duas formas, exato (ótima x gulosa)
    A   gulosa + emendas: seleção forward, Shapley, interação "alvo" x preços
    B   colas de preços (protagonistas (i) e (ii)) + só preços
    C   rodada perfeita com preços (teto não executável)
    C'  iteração de política a partir da gulosa
    F2  pares (estado, ação), cartelas alcançáveis
    curva complexidade x pontos; distribuição por General; ordem de preenchimento;
    preço marginal; "pergunte ao ótimo"; replays

Exatidão: todo valor esperado sai de `rodada_exata` em inteiros escalados (ver derivacao.py).
Visitação (probabilidade de chegar a cada estado) também é exata em inteiros. Só as
distribuições da pontuação final são float64 (como na v1).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import factorial, lcm

import numpy as np

import derivacao as D

NB, ND = D.NB, D.ND
FULL = (1 << NB) - 1
NOMES = D.NOMES
KEEPS, KIDX, MS5, IDX5 = D.KEEPS, D.KIDX, D.MS5, D.IDX5
ORDEM = D.subconjuntos_ordenados(D.TODAS)  # |S| crescente
POP = [bin(S).count("1") for S in range(1 << NB)]
RES2 = D.RES / "v2"
F150 = D.escala(10)  # 6^150

# id da afirmação -> chave em D.NUM (conferência de claims.yaml)
CLAIM_CHAVE_V2 = {
    "C39": "lema_i_grupo_guarda", "C40": "lema_ii_grupo_guarda", "C41": "lema_i_par_Seguida->General",
    "C42": "A_so_seg4x", "C43": "A_passo5", "C44": "A_alvo_ganho_2face_8_2", "C45": "B_sozinha_int_R4",
    "C46": "B_2face_8_2_R4", "C47": "B_2face_8_2_so_precos", "C48": "B_2face_8_2_shapley_seg4x",
    "C49": "B_sozinha_int_shapley_precos_combinacoes", "C50": "B_passo1", "C51": "B_busca_local_valor",
    "C52": "C_sozinha_exata", "C53": "B_mq_rms", "C54": "Cl_passo1", "C55": "curva_n",
    "C56": "dg_otima_feito_media", "C57": "ord_gulosa_P10_9", "C58": "pm_rodada1_8", "C59": "f2_n_pares",
}

# ---------------------------------------------------------------------------
# Atributos das mãos (para as colas)
# ---------------------------------------------------------------------------
CNT = np.array([[m.count(f) for f in range(1, 7)] for m in MS5])  # ND x 6
MF = np.array([D._mais_freq(m)[0] for m in MS5])
MC = np.array([D._mais_freq(m)[1] for m in MS5])
PERFIL = [tuple(sorted(Counter(m).values())) for m in MS5]
K_GUL = np.array([KIDX[(MF[d],) * MC[d]] for d in range(ND)])


def _seq_guarda(d):
    """Guarda 'para a Seguida': um dado de cada face que a mão tem em comum com a Seguida
    (baixa 1-2-3-4-5 ou alta 2-3-4-5-6) com mais faces em comum; empate -> a alta."""
    s = set(MS5[d])
    baixa = tuple(sorted(s & {1, 2, 3, 4, 5}))
    alta = tuple(sorted(s & {2, 3, 4, 5, 6}))
    t = baixa if len(baixa) > len(alta) else alta
    return KIDX[t], len(t)


SEQ = [_seq_guarda(d) for d in range(ND)]
K_SEQ = np.array([k for k, _ in SEQ])
N_SEQ = np.array([n for _, n in SEQ])
DOISPAR = np.array([PERFIL[d] == (1, 2, 2) for d in range(ND)])
K_DOISPAR = np.array([KIDX[tuple(sorted(x for x in MS5[d] if CNT[d, x - 1] == 2))] if DOISPAR[d] else -1
                      for d in range(ND)])
FU = np.array([PERFIL[d] == (2, 3) for d in range(ND)])
TRINCA = np.array([max((f for f in range(1, 7) if CNT[d, f - 1] == 3), default=0) for d in range(ND)])
K_TRINCA = np.array([KIDX[(TRINCA[d],) * 3] if TRINCA[d] else -1 for d in range(ND)])

# guarda "alvo" por máscara de faces úteis (6 bits): face útil mais frequente (empate -> maior)
K_ALVO = np.zeros((64, ND), int)
C_ALVO = np.zeros((64, ND), int)
for _mask in range(64):
    for _d in range(ND):
        _u = [f for f in range(1, 7) if _mask >> (f - 1) & 1]
        if not _u:
            K_ALVO[_mask, _d] = KIDX[()]
            continue
        _f = max(_u, key=lambda f: (CNT[_d, f - 1], f))
        K_ALVO[_mask, _d] = KIDX[(_f,) * CNT[_d, _f - 1]]
        C_ALVO[_mask, _d] = CNT[_d, _f - 1]

# ---------------------------------------------------------------------------
# Colas: definição executável (o mesmo texto vai para claims.yaml › colas)
# ---------------------------------------------------------------------------
EMENDAS = {
    "seg4x": dict(itens=3, texto=(
        "Seguida livre e a mão tem 4 faces diferentes que pertencem a uma mesma Seguida (1-2-3-4-5 ou "
        "2-3-4-5-6), no 1º ou 2º lançamento, sem parar: guarde um dado de cada uma dessas 4 faces e relance "
        "o outro. Se as duas Seguidas servem (ex.: 12346), use a alta (2346).")),
    "quad_gen": dict(itens=2, texto=(
        "General livre e a casa escolhida é a Quadrada, no 1º ou 2º lançamento: não marque; guarde os 4 dados "
        "iguais e relance o outro.")),
    "alvo": dict(itens=2, texto=(
        "Ao guardar por face, só conta face 'útil': aquela cuja casa de número está livre, ou qualquer face "
        "se Fú, Quadrada ou General estiver livre. Guarde todos os dados da face útil mais frequente na mão "
        "(empate: a maior). Se a mão não tem nenhuma face útil: com a Seguida livre, guarde um dado de cada "
        "face que a mão tem em comum com a Seguida (baixa ou alta) de mais faces em comum (empate: a alta); "
        "sem a Seguida livre, relance os 5.")),
    "risca_gen": dict(itens=1, texto=(
        "Quando nenhuma casa livre pontua, risque o General (se livre); senão a de menor índice. "
        "(Só na marcação por pontos, sem preços.)")),
    "doispar": dict(itens=3, texto=(
        "Fú livre, 2º lançamento, mão com dois pares e um avulso, sem parar: guarde os dois pares e relance o avulso.")),
    "fu_alto": dict(itens=3, texto=(
        "2º lançamento, casa escolhida = Fú, a trinca do Fú é de 4, 5 ou 6 e (Quadrada ou General) livre: não "
        "marque; guarde a trinca e relance os outros dois.")),
}
ORDEM_EMENDAS = ["seg4x", "quad_gen", "alvo", "risca_gen", "doispar", "fu_alto"]
R4 = ("alvo", "seg4x", "quad_gen", "doispar")

SOZINHA = None  # preços exatos C10–C14 (Fraction), preenchido em main()
PRECOS = {
    # (i) valor de cada casa jogada sozinha (C10–C14) arredondado ao inteiro mais próximo
    "sozinha_int": [2, 4, 6, 8, 11, 13, 7, 8, 11, 2],
    # (ii) número = 2x a face; Fú, Seguida, Quadrada 8; General 2
    "2face_8_2": [2, 4, 6, 8, 10, 12, 8, 8, 8, 2],
    # variantes para a curva
    "2face_78112": [2, 4, 6, 8, 10, 12, 7, 8, 11, 2],
    "2face_comb0": [2, 4, 6, 8, 10, 12, 0, 0, 0, 0],
    "face_comb0": [1, 2, 3, 4, 5, 6, 0, 0, 0, 0],
    "3face_78112": [3, 6, 9, 12, 15, 18, 7, 8, 11, 2],
    "num0_8_2": [0, 0, 0, 0, 0, 0, 8, 8, 8, 2],
    "sozinha_prelim": [2, 4, 6, 8, 10, 13, 7, 8, 11, 2],
}
ITENS_PRECOS = {"sozinha_int": 11, "2face_8_2": 4, "2face_78112": 6, "2face_comb0": 3, "face_comb0": 3,
                "3face_78112": 6, "num0_8_2": 4, "sozinha_prelim": 11}
TEXTO_PRECOS = {
    "sozinha_int": "preço = valor da casa jogada sozinha, arredondado: Ás 2, Duque 4, Terno 6, Quadra 8, Quina 11, "
                   "Sena 13, Fú 7, Seguida 8, Quadrada 11, General 2",
    "2face_8_2": "preço da casa de número = 2 × a face; Fú, Seguida e Quadrada 8; General 2",
    "2face_78112": "preço da casa de número = 2 × a face; Fú 7, Seguida 8, Quadrada 11, General 2",
    "2face_comb0": "preço da casa de número = 2 × a face; combinações (inclusive General) 0",
    "face_comb0": "preço da casa de número = a face; combinações 0",
    "3face_78112": "preço da casa de número = 3 × a face; Fú 7, Seguida 8, Quadrada 11, General 2",
    "num0_8_2": "casas de número 0; Fú, Seguida e Quadrada 8; General 2",
    "sozinha_prelim": "tabela da exploração (Quina 10 em vez de 11): 2,4,6,8,10,13,7,8,11,2",
}


def politica_cola(precos=None, regras=(), boca=True):
    """Política (dict S -> (A1, A2, A3)) da cola: gulosa G1–G4 com marcação por preço e emendas.
    precos: None (G1–G2: maior pontuação) ou lista de 10 números (marca argmax pontos - preço, H5)."""
    regras = frozenset(regras)
    pr = None if precos is None else [Fraction(p) for p in precos]
    if pr is not None:
        L = lcm(*[p.denominator for p in pr])
        prL = np.array([int(p * L) for p in pr])
    pol = {}
    ar = np.arange(ND)
    for S in ORDEM:
        bs = np.array(D.caixas(S))
        livre = set(bs.tolist())
        uteis = sum(1 << (f - 1) for f in range(1, 7) if (f - 1) in livre or (livre & {6, 8, 9}))
        A = []
        for r in range(3):
            pts = (D.SCBF if (r == 0 and boca) else D.SCF)[bs]
            if pr is not None:
                crit = pts * L - prL[bs][:, None]
                ia = crit.argmax(0)  # empate -> menor índice (H5)
            else:
                ia = pts.argmax(0)  # G2
                if "risca_gen" in regras and 9 in livre:
                    ia = np.where(pts.max(0) == 0, list(bs).index(9), ia)
            bh = bs[ia]
            ph = pts[ia, ar]
            para = np.full(ND, r == 2) | ((bh >= 6) & (ph > 0)) | (MC == 5)  # G3
            if r < 2 and "quad_gen" in regras and 9 in livre:
                para &= ~(bh == 8)
            if r == 1 and "fu_alto" in regras and (8 in livre or 9 in livre):
                para &= ~((bh == 6) & (TRINCA >= 4))
            if "alvo" in regras:
                k = K_ALVO[uteis].copy()
                sem = C_ALVO[uteis] == 0
                k = np.where(sem, K_SEQ if 7 in livre else KIDX[()], k)
            else:
                k = K_GUL.copy()  # G4
            if 7 in livre and "seg4x" in regras:
                k = np.where(N_SEQ == 4, K_SEQ, k)
            if r == 1 and "doispar" in regras and 6 in livre:
                k = np.where(DOISPAR, K_DOISPAR, k)
            if r == 1 and "fu_alto" in regras and (8 in livre or 9 in livre):
                k = np.where(FU & (TRINCA >= 4), K_TRINCA, k)
            A.append(np.where(para, -(bh + 1), k).tolist())
        pol[S] = tuple(A)
    return pol


# ---------------------------------------------------------------------------
# Avaliação exata (Bellman de avaliação) e rodada perfeita com valor terminal dado
# ---------------------------------------------------------------------------

def avalia_exata(pol, boca=True):
    """E^pi(S) para toda cartela, inteiros escalados por 6^(15|S|)."""
    E = {0: 0}
    for S in ORDEM:
        E[S], _ = D.rodada_exata(S, E, boca=boca, pol=pol[S])
    return E


def valor(E, S=FULL):
    return D.frac(E, S)


def politica_rodada_precos(precos):
    """'Rodada perfeita com preços': em cada cartela, joga a rodada de forma ótima para maximizar
    E[pontos(casa marcada) - preço(casa)] (desempate H6). precos: 10 Fractions."""
    pr = [Fraction(p) for p in precos]
    L = lcm(*[p.denominator for p in pr])
    pol = {}
    for S in ORDEM:
        fut = {b: -int(pr[b] * L) for b in D.caixas(S)}
        _, pol[S] = D.rodada_exata(S, None, fut=fut, Dp=L)
    return pol


def melhora(E):
    """Um passo de melhoria de política (nível rodada): em cada cartela, a rodada ótima com valor
    terminal E(S \\ b) (E = valor exato da política atual)."""
    return {S: D.rodada_exata(S, E)[1] for S in ORDEM}


# ---------------------------------------------------------------------------
# Visitação exata (cadeia de Markov induzida pela política)
# ---------------------------------------------------------------------------

def visitacao_exata(pol):
    """alc[S] = P(chegar à cartela S) * 6^(15(10-|S|)) (inteiro);
    massas[S] = [m1, m2, m3]: P(mão d no lançamento r | início da rodada em S) * 6^(5r) (inteiros);
    marca[S][b] = P(marcar b nesta rodada | S) * 6^15."""
    alc = defaultdict(int)
    alc[FULL] = 1
    massas, marca = {}, {}
    for S in sorted(ORDEM, key=lambda S: -POP[S]):
        if alc[S] == 0:
            continue
        A = pol[S]
        m = list(D.P0_CNT)
        ms = []
        mk = Counter()
        for r in range(3):
            ms.append(m)
            km = defaultdict(int)
            for d in range(ND):
                if m[d]:
                    a = A[r][d]
                    if a < 0:
                        mk[-(a + 1)] += m[d] * 6 ** (5 * (2 - r))
                    else:
                        km[a] += m[d]
            if r < 2:
                novo = [0] * ND
                for k, w in km.items():
                    for j, c in D.TINT[k]:
                        novo[j] += w * c
                m = novo
        massas[S], marca[S] = ms, mk
        for b, w in mk.items():
            alc[S & ~(1 << b)] += alc[S] * w
    return dict(alc), massas, marca


def p_alc(alc, S):
    return Fraction(alc.get(S, 0), 6 ** (15 * (NB - POP[S])))


# ---------------------------------------------------------------------------
# Avaliação em lote (processos paralelos) com cache
# ---------------------------------------------------------------------------

def chave(precos, regras):
    return (None if precos is None or not any(precos) else tuple(Fraction(p) for p in precos),
            tuple(sorted(regras)))


def _avalia_chave(ch):
    precos, regras = ch
    E = avalia_exata(politica_cola(None if precos is None else list(precos), regras))
    v = valor(E)
    return ch, (v.numerator, v.denominator)


class Avaliador:
    def __init__(self, ex):
        self.ex = ex
        self.cache = {}

    def lote(self, chaves):
        falta = [c for c in dict.fromkeys(chaves) if c not in self.cache]
        if falta:
            it = self.ex.map(_avalia_chave, falta) if self.ex is not None else map(_avalia_chave, falta)
            for ch, (a, b) in it:
                self.cache[ch] = Fraction(a, b)
        return [self.cache[c] for c in chaves]

    def __call__(self, precos, regras=()):
        return self.lote([chave(precos, regras)])[0]


# ---------------------------------------------------------------------------
# Rótulos das colas e itens de cola (H4)
# ---------------------------------------------------------------------------

def itens(nome_precos, regras):
    return (ITENS_PRECOS[nome_precos] if nome_precos else 0) + sum(EMENDAS[r]["itens"] for r in regras)


def n_regras(nome_precos, regras):
    return (1 if nome_precos else 0) + len(regras)


def texto_cola(nome_precos, regras):
    partes = []
    if nome_precos:
        partes.append("Marque a casa livre de maior (pontos − preço), " + TEXTO_PRECOS[nome_precos]
                      + "; empate: a de menor índice.")
    else:
        partes.append("Gulosa G1–G4 (marca a casa de maior pontuação imediata).")
    for r in ORDEM_EMENDAS:
        if r in regras:
            partes.append(f"[{r}] " + EMENDAS[r]["texto"])
    return " ".join(partes)


def rotulo(nome_precos, regras):
    base = nome_precos if nome_precos else "gulosa"
    return base + "".join("+" + r for r in ORDEM_EMENDAS if r in regras)


# ---------------------------------------------------------------------------
# §0. Lema da diferença de desempenho (duas formas), exato
# ---------------------------------------------------------------------------

def categoria(r, a_pi, a_ot):
    """Tipo do erro da política pi (gulosa) frente à ótima na decisão do lançamento r (0-based)."""
    lanc = f"{r + 1}º"
    if a_pi < 0 and a_ot < 0:
        return f"casa no {lanc}"
    if a_pi < 0:
        return f"para no {lanc} (ótima relança)"
    if a_ot < 0:
        return f"relança no {lanc} (ótima marca)"
    return f"guarda no {lanc}"


def grupo(cat):
    return "guarda" if cat.startswith("guarda") else ("casa" if cat.startswith("casa") else "parada")


def _q(det, r, d, a, f):
    Mx = det["MB"] if r == 0 else det["M"]
    if a < 0:
        return Mx[-(a + 1)][d] * f
    return (det["W2"] if r == 0 else det["W1"])[a]


VS = ("V1", "V2", "V3")


def lema(E, POL, EG, GP):
    """Forma (i): sum_s d^G(s)[V*(s) - Q*(s, G(s))]; forma (ii): sum_s d*(s)[Q^G(s, pi*(s)) - V^G(s)].
    Cada parcela é inteiro / 6^150."""
    out = {}
    # forma (i)
    alc, ms, _ = visitacao_exata(GP)
    cat, porn, pares = Counter(), Counter(), Counter()
    for S, m3 in ms.items():
        _, _, det = D.rodada_exata(S, E, detalhe=True)
        for r in range(3):
            f = 6 ** (5 * (2 - r))
            V = det[VS[r]]
            for d in range(ND):
                if not m3[r][d]:
                    continue
                a = GP[S][r][d]
                gap = V[d] - _q(det, r, d, a, f)
                if gap:
                    t = alc[S] * m3[r][d] * gap
                    ao = POL[S][r][d]
                    cat[categoria(r, a, ao)] += t
                    porn[POP[S]] += t
                    if a < 0 and ao < 0:
                        pares[(NOMES[-(a + 1)], NOMES[-(ao + 1)])] += t
    out["i"] = dict(cat=cat, porn=porn, pares=pares)
    # forma (ii)
    alc, ms, _ = visitacao_exata(POL)
    cat, porn = Counter(), Counter()
    for S, m3 in ms.items():
        _, _, det = D.rodada_exata(S, EG, pol=GP[S], detalhe=True)
        for r in range(3):
            f = 6 ** (5 * (2 - r))
            V = det[VS[r]]
            for d in range(ND):
                if not m3[r][d]:
                    continue
                ao = POL[S][r][d]
                dif = _q(det, r, d, ao, f) - V[d]
                if dif:
                    t = alc[S] * m3[r][d] * dif
                    cat[categoria(r, GP[S][r][d], ao)] += t
                    porn[POP[S]] += t
    out["ii"] = dict(cat=cat, porn=porn)
    return out


# ---------------------------------------------------------------------------
# Detalhes em float de uma cartela (para os produtos de visualização)
# ---------------------------------------------------------------------------
_E_OT = {}


@lru_cache(maxsize=None)
def det_float(S):
    """Valores da política ótima na cartela S em float: V[r] (252) e função q(r, d, a)."""
    E = _E_OT["E"]
    _, P, det = D.rodada_exata(S, E, detalhe=True)
    Dp = D.escala(POP[S] - 1)
    out = {"V": [], "P": P, "det": det, "Dp": Dp}
    for r in range(3):
        f = 6 ** (5 * (2 - r))
        out["V"].append([v / (Dp * f) for v in det[VS[r]]])
    return out


def q_float(S, r, d, a):
    x = det_float(S)
    f = 6 ** (5 * (2 - r))
    return _q(x["det"], r, d, a, f) / (x["Dp"] * f)


def acao_str(a):
    return D.acao_txt(int(a))


def mao_str(d):
    return "".join(map(str, MS5[d]))


# ---------------------------------------------------------------------------
# Distribuição final por General feito / não feito (float64, propagação para frente)
# ---------------------------------------------------------------------------

def dist_por_general(pol):
    dist = {(FULL, False): np.zeros(D.PMAX)}
    dist[(FULL, False)][0] = 1.0
    for n in range(NB, 0, -1):
        for (S, fl) in [k for k in dist if POP[k[0]] == n]:
            dS = dist.pop((S, fl))
            out = D.saida_rodada(S, pol[S])
            for b in D.caixas(S):
                for s in np.nonzero(out[b])[0]:
                    k2 = (S & ~(1 << b), fl or (b == 9 and s > 0))
                    acc = dist.setdefault(k2, np.zeros(D.PMAX))
                    acc[s:] += out[b, s] * dS[: D.PMAX - s]
    return dist[(0, False)], dist[(0, True)]


# ---------------------------------------------------------------------------
# Ordem de preenchimento e preço marginal (exatos)
# ---------------------------------------------------------------------------

def ordem_preenchimento(pol):
    """P[b][t] = P(casa b preenchida na rodada t+1) (Fraction)."""
    alc, _, marca = visitacao_exata(pol)
    P = [[Fraction(0)] * NB for _ in range(NB)]
    acc = defaultdict(int)
    for S, mk in marca.items():
        t = NB - POP[S]  # rodada t+1
        for b, w in mk.items():
            acc[(b, t)] += alc[S] * w
    for (b, t), v in acc.items():
        P[b][t] = Fraction(v, 6 ** (15 * (t + 1)))
    return P


def preco_marginal_por_rodada(E, alc):
    """M[b][t] = E[E*(S_t) - E*(S_t \\ b) | b livre no início da rodada t+1], sob a ótima."""
    num = defaultdict(int)
    den = defaultdict(int)
    for S in ORDEM:
        a = alc.get(S, 0)
        if not a:
            continue
        n = POP[S]
        t = NB - n
        for b in D.caixas(S):
            dif = E[S] - E[S & ~(1 << b)] * 6 ** 15  # escala 6^(15n)
            num[(b, t)] += a * dif
            den[(b, t)] += a
    M = [[None] * NB for _ in range(NB)]
    for (b, t), v in num.items():
        n = NB - t
        M[b][t] = Fraction(v, F150) / Fraction(den[(b, t)], 6 ** (15 * (NB - n)))
    return M


# ---------------------------------------------------------------------------
# Replays com dados comuns
# ---------------------------------------------------------------------------

def joga(pol, U):
    """Uma partida com os dados pré-sorteados U[t, r, i] (rodada, lançamento, posição).
    Dado na posição i relançado no lançamento r recebe U[t, r, i]; guardas ocupam as primeiras
    posições (da esquerda) com a face guardada."""
    S = FULL
    total = 0
    rodadas = []
    for t in range(NB):
        dados = [int(x) for x in U[t, 0]]
        passos = []
        for r in range(3):
            d = IDX5[tuple(sorted(dados))]
            a = int(pol[S][r][d])
            ao = int(det_float(S)["P"][r][d])
            custo = det_float(S)["V"][r][d] - q_float(S, r, d, a)
            passo = {"dados": dados[:], "acao": a, "acao_otima": ao, "custo": round(custo, 4)}
            passos.append(passo)
            if a < 0:
                b = -(a + 1)
                pts = int((D.SCB if r == 0 else D.SC)[b][d])
                total += pts
                S &= ~(1 << b)
                rodadas.append({"lancamentos": passos, "casa": b, "pontos": pts, "total": total})
                break
            usado = [False] * 5
            for face in KEEPS[a]:
                i = next(i for i in range(5) if not usado[i] and dados[i] == face)
                usado[i] = True
            dados = [dados[i] if usado[i] else int(U[t, r + 1, i]) for i in range(5)]
    return {"rodadas": rodadas, "total": total}


def _avalia_rodada(precos):
    v = valor(avalia_exata(politica_rodada_precos(precos)))
    return v.numerator, v.denominator


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def fl(x, nd=6):
    return round(float(x), nd)


def salva(nome, obj):
    p = RES2 / nome
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  -> resultados/v2/{nome} ({p.stat().st_size / 1024:.0f} KB)")


def forward(av, precos, pool, base_val):
    """Seleção forward: a cada passo entra a emenda de maior valor exato."""
    atual, resto, cam = (), list(pool), []
    vant = base_val
    while resto:
        vals = av.lote([chave(precos, atual + (r,)) for r in resto])
        i = max(range(len(resto)), key=lambda j: vals[j])
        r = resto.pop(i)
        atual = atual + (r,)
        cam.append({"emenda": r, "regras": list(atual), "valor": vals[i], "ganho_marginal": vals[i] - vant,
                    "candidatos": {rr: v for rr, v in zip([*resto[:i], r, *resto[i:]], vals)}})
        vant = vals[i]
    return cam


def shapley(av, precos, regras):
    """Shapley exato com 6 jogadores: preços das casas de número, preços das combinações, e as 4 regras."""
    jog = ["precos_numeros", "precos_combinacoes", *regras]
    n = len(jog)

    def vec(T):
        p = [precos[b] if ((b < 6 and "precos_numeros" in T) or (b >= 6 and "precos_combinacoes" in T)) else 0
             for b in range(NB)]
        return chave(p, [r for r in regras if r in T])

    subs = [frozenset(c) for k in range(n + 1) for c in combinations(jog, k)]
    vals = dict(zip(subs, av.lote([vec(T) for T in subs])))
    phi = {}
    for j in jog:
        s = Fraction(0)
        for T in subs:
            if j in T:
                continue
            w = Fraction(factorial(len(T)) * factorial(n - len(T) - 1), factorial(n))
            s += w * (vals[T | {j}] - vals[T])
        phi[j] = s
    assert sum(phi.values()) == vals[frozenset(jog)] - vals[frozenset()]
    return phi, vals


def main(E, POL, dist_ot, dist_g):
    global SOZINHA
    from concurrent.futures import ProcessPoolExecutor
    reg, NUM = D.reg, D.NUM
    t0 = time.perf_counter()
    RES2.mkdir(parents=True, exist_ok=True)
    _E_OT["E"] = E
    Eot = valor(E)
    SOZINHA = [NUM[f"sozinha_{b}"] for b in range(NB)]
    GP = {S: D.politica_gulosa(S) for S in ORDEM}
    EG = avalia_exata(GP)
    Eg = valor(EG)
    gap = Eot - Eg
    ganho = lambda v: float((v - Eg) / gap)  # noqa: E731

    # ------------------------------------------------------------- F2
    print("\n== v2 · F2. Formulação: Bellman de avaliação, cadeia induzida")
    assert Eg == NUM["E_gulosa"], "avaliação da gulosa != C29"
    reg("f2_aval_gulosa_igual_C29", True, "recursão de avaliação (política fixa) reproduz C29 exatamente")
    assert avalia_exata(POL) == E
    reg("f2_aval_otima_igual_C17", True, "Bellman de avaliação aplicado a pi* = Bellman de otimalidade, em toda cartela")
    n_marca = sum(POP[S] for S in ORDEM) * 3 * ND
    n_guarda = 2 * len(ORDEM) * sum(len(s) for s in D.SUBS_I)
    reg("f2_n_pares", n_marca + n_guarda, f"pares (estado, ação): {n_marca} de marcar + {n_guarda} de guardar")
    NUM["f2_n_pares_marca"], NUM["f2_n_pares_guarda"] = n_marca, n_guarda
    NUM["f2_n_guardas_por_mao_total"] = sum(len(s) for s in D.SUBS_I)
    reg("f2_n_acoes_guarda_media", Fraction(sum(len(s) for s in D.SUBS_I), ND), "guardas distintas por mão (média)")

    ex = ProcessPoolExecutor()
    av = Avaliador(ex)
    av.cache[chave(None, ())] = Eg

    # ------------------------------------------------------------- §0
    print("\n== v2 · §0. Lema da diferença de desempenho (exato)")
    L = lema(E, POL, EG, GP)
    alvo_int = E[FULL] - EG[FULL]  # escala 6^150
    diag = {"E_otimo": fl(Eot, 9), "E_gulosa": fl(Eg, 9), "gap": fl(gap, 9), "formas": {}}
    for forma in ("i", "ii"):
        cat = L[forma]["cat"]
        assert sum(cat.values()) == alvo_int, f"forma ({forma}) não fecha"
        reg(f"lema_{forma}_soma", Fraction(sum(cat.values()), F150), "= E* - E_G exatamente")
        gr = Counter()
        for c, v in cat.items():
            gr[grupo(c)] += v
            reg(f"lema_{forma}_{c}", Fraction(v, F150))
        for g in ("guarda", "parada", "casa"):
            reg(f"lema_{forma}_grupo_{g}", Fraction(gr[g], F150),
                f"{100 * gr[g] / alvo_int:.1f}% de E* - E_G")
            NUM[f"lema_{forma}_grupo_{g}_pct"] = 100 * gr[g] / alvo_int
        porn = {n: Fraction(v, F150) for n, v in sorted(L[forma]["porn"].items())}
        for n, v in porn.items():
            reg(f"lema_{forma}_livres_{n}", v)
        diag["formas"][forma] = {
            "por_categoria": {c: fl(Fraction(v, F150), 9) for c, v in sorted(cat.items(), key=lambda kv: -kv[1])},
            "por_categoria_exato": {c: str(Fraction(v, F150)) for c, v in cat.items()},
            "por_grupo": {g: fl(Fraction(gr[g], F150), 9) for g in ("guarda", "parada", "casa")},
            "por_casas_livres": {str(n): fl(v, 9) for n, v in porn.items()},
        }
    pares = L["i"]["pares"].most_common(12)
    diag["formas"]["i"]["pares_casa_gulosa_para_otima"] = [[a, b, fl(Fraction(v, F150), 9)] for (a, b), v in pares]
    for (a, b), v in pares[:5]:
        reg(f"lema_i_par_{a}->{b}", Fraction(v, F150))
    diag["nota"] = ("forma (i): sum_s d^G(s)[V*(s)-Q*(s,G(s))] (visitação da gulosa, jogo ótimo depois); "
                    "forma (ii): sum_s d*(s)[Q^G(s,pi*(s))-V^G(s)] (visitação da ótima, gulosa depois). "
                    "s = (casas livres, lançamento, mão). Categorias: tipo da ação da gulosa x da ótima.")
    salva("diagnostico_lema.json", diag)
    print(f"  ({time.perf_counter() - t0:.0f} s)")

    # ------------------------------------------------------------- A
    print("\n== v2 · A. Gulosa + emendas (seleção forward, exata)")
    camA = forward(av, None, ORDEM_EMENDAS, Eg)
    for r in ORDEM_EMENDAS:
        v = camA[0]["candidatos"][r]
        reg(f"A_so_{r}", v, f"gulosa + {r} sozinha: ganho {float(v - Eg):+.4f}")
        NUM[f"A_ganho_{r}"] = v - Eg
    kum = 0
    for i, p in enumerate(camA, 1):
        kum += EMENDAS[p["emenda"]]["itens"]
        p["itens"] = kum
        reg(f"A_passo{i}", p["valor"], f"+{p['emenda']} (itens {kum}) ganho marginal {float(p['ganho_marginal']):+.4f}, "
            f"recuperado {100 * ganho(p['valor']):.1f}%")
        NUM[f"A_passo{i}_emenda"] = p["emenda"]
        NUM[f"A_passo{i}_ganho"] = p["ganho_marginal"]
        NUM[f"A_passo{i}_recuperado"] = ganho(p["valor"])
    # interação "alvo" x preços
    for nome in (None, "2face_8_2", "sozinha_int"):
        pr = PRECOS[nome] if nome else None
        a0, a1 = av.lote([chave(pr, ()), chave(pr, ("alvo",))])
        reg(f"A_alvo_ganho_{nome or 'gulosa'}", a1 - a0, "ganho da emenda 'alvo' sobre a base")

    # ------------------------------------------------------------- B
    print("\n== v2 · B. Colas de preços")
    colas_B = {}
    for nome in ("sozinha_int", "2face_8_2"):
        pr = PRECOS[nome]
        v0, v4 = av.lote([chave(pr, ()), chave(pr, R4)])
        reg(f"B_{nome}_so_precos", v0, f"itens {itens(nome, ())}, recuperado {100 * ganho(v0):.1f}%")
        reg(f"B_{nome}_R4", v4, f"itens {itens(nome, R4)}, recuperado {100 * ganho(v4):.1f}%")
        NUM[f"B_{nome}_R4_itens"] = itens(nome, R4)
        NUM[f"B_{nome}_R4_recuperado"] = ganho(v4)
        phi, _ = shapley(av, pr, R4)
        for j, v in phi.items():
            reg(f"B_{nome}_shapley_{j}", v)
        colas_B[nome] = {"so_precos": v0, "R4": v4, "shapley": phi}
    camB = forward(av, PRECOS["2face_8_2"], list(R4) + ["fu_alto"], colas_B["2face_8_2"]["so_precos"])
    kum = itens("2face_8_2", ())
    for i, p in enumerate(camB, 1):
        kum += EMENDAS[p["emenda"]]["itens"]
        p["itens"] = kum
        reg(f"B_passo{i}", p["valor"], f"2face_8_2 +{p['emenda']} (itens {kum})")
        NUM[f"B_passo{i}_emenda"] = p["emenda"]
    # busca local nos preços inteiros (Ás fixo; os preços valem a menos de constante), com R4
    p = list(PRECOS["2face_8_2"])
    best = av(p, R4)
    n_av = 0
    while True:
        viz = []
        for b in range(1, NB):
            for s in (1, -1):
                q = p[:]
                q[b] += s
                viz.append(q)
        vals = av.lote([chave(q, R4) for q in viz])
        n_av += len(viz)
        j = max(range(len(viz)), key=lambda j: vals[j])
        if vals[j] <= best:
            break
        p, best = viz[j], vals[j]
    PRECOS["busca_local"] = p
    ITENS_PRECOS["busca_local"] = 11
    TEXTO_PRECOS["busca_local"] = "tabela achada por busca local (passos de 1, Ás fixo em 2): " + ",".join(map(str, p))
    reg("B_busca_local_valor", best, f"preços {p}, {n_av} avaliações")
    NUM["B_busca_local_precos"] = p
    # ajuste aditivo E*(S) ~ c_|S| + sum p_b por mínimos quadrados (float)
    Ef = np.array([float(valor(E, S)) for S in ORDEM])
    X = np.zeros((len(ORDEM), 2 * NB))
    for i, S in enumerate(ORDEM):
        for b in D.caixas(S):
            X[i, b] = 1
        X[i, NB + POP[S] - 1] = 1
    sol, *_ = np.linalg.lstsq(X, Ef, rcond=None)
    res = X @ sol - Ef
    pmq = sol[:NB] - sol[0]
    pmq_r = [Fraction(round(x, 2)).limit_denominator(100) for x in pmq]
    reg("B_mq_rms", float(np.sqrt((res ** 2).mean())), "resíduo RMS do ajuste aditivo (pontos)")
    NUM["B_mq_precos"] = [float(x) for x in pmq_r]
    print("  preços MQ (Ás = 0, 0,01):", [float(x) for x in pmq_r])
    marg = [valor(E) - valor(E, FULL & ~(1 << b)) for b in range(NB)]

    # ------------------------------------------------------------- C
    print("\n== v2 · C. Rodada perfeita com preços (teto)")
    tetos = {"zero": [0] * NB, "sozinha_exata": SOZINHA, "sozinha_int": PRECOS["sozinha_int"],
             "2face_8_2": PRECOS["2face_8_2"], "marginal_cheia": marg, "mq": pmq_r}
    vals = list(ex.map(_avalia_rodada, tetos.values()))
    tetoC = {}
    for (nome, _), (a, b) in zip(tetos.items(), vals):
        v = Fraction(a, b)
        tetoC[nome] = v
        reg(f"C_{nome}", v, f"recuperado {100 * ganho(v):.1f}%")
        NUM[f"C_{nome}_recuperado"] = ganho(v)
    ex.shutdown()

    # ------------------------------------------------------------- C'
    print("\n== v2 · C'. Iteração de política a partir da gulosa")
    pol, it = GP, []
    for k in range(11):
        Ek = EG if k == 0 else avalia_exata(pol)
        difs = [valor(E, S) - valor(Ek, S) for S in ORDEM]
        n_igual = sum(1 for x in difs if x == 0)
        it.append({"passo": k, "valor": Ek[FULL], "max_dif": max(difs), "cartelas_iguais": n_igual})
        reg(f"Cl_passo{k}", valor(Ek), f"max_S (E*-E_k) = {float(max(difs)):.3g}; cartelas com E_k = E*: {n_igual}/1023")
        if n_igual == len(ORDEM):
            break
        pol = melhora(Ek)
    NUM["Cl_n_passos"] = len(it) - 1
    # mínimo garantido pela indução: após k passos, toda cartela com |S| <= k é ótima
    from math import comb
    NUM["Cl_min_garantido"] = [sum(comb(NB, j) for j in range(1, k + 1)) for k in range(len(it))]
    assert all(x["cartelas_iguais"] >= m for x, m in zip(it, NUM["Cl_min_garantido"]))
    print("  mínimo garantido (|S| <= k):", NUM["Cl_min_garantido"])
    # diferenças citadas no texto
    vi, vii = colas_B["sozinha_int"]["R4"], colas_B["2face_8_2"]["R4"]
    reg("d_shapley_soma_ii", vii - Eg, "soma dos Shapley da cola (ii) = (ii) - gulosa")
    reg("d_shapley_soma_i", vi - Eg, "soma dos Shapley da cola (i)")
    reg("d_cola_i_menos_ii", vi - vii)
    reg("d_so_precos_ii", colas_B["2face_8_2"]["so_precos"] - Eg, "só preços (ii) - gulosa")
    reg("d_busca_menos_ii", best - vii)
    reg("d_teto_menos_ii", tetoC["sozinha_exata"] - vii, "rodada perfeita (preços sozinha) - cola (ii)")
    reg("d_dentro_da_rodada", tetoC["zero"] - Eg, "gulosa -> rodada perfeita sem preços")
    reg("d_precos_futuro", tetoC["sozinha_exata"] - tetoC["zero"], "preços 'sozinha' no futuro")
    reg("d_resto_nao_aditivo", Eot - tetoC["sozinha_exata"])
    reg("d_pi1_menos_gulosa", valor({FULL: it[1]["valor"]}, FULL) - Eg, "1 passo de melhoria de Bellman")
    salva("iteracao_politica.json", {
        "descricao": "Iteração de política no nível da rodada, a partir da gulosa: pi_{k+1}(S) = rodada ótima com "
                     "valor terminal E^{pi_k}(S \\ b); valor exato E^{pi_k}(cartela vazia).",
        "passos": [{"passo": x["passo"], "valor": fl(valor({FULL: x["valor"]}, FULL), 9),
                    "valor_exato": str(valor({FULL: x["valor"]}, FULL)), "max_dif_E_otimo": float(x["max_dif"]),
                    "cartelas_com_E_igual_otimo": x["cartelas_iguais"]} for x in it],
        "E_otimo": fl(Eot, 9), "E_gulosa": fl(Eg, 9)})

    # ------------------------------------------------------------- curva
    print("\n== v2 · Curva complexidade x pontos")
    curva = [(None, ())] + [(None, (r,)) for r in ORDEM_EMENDAS]
    curva += [(None, tuple(p["regras"])) for p in camA]
    for nome in ("sozinha_int", "2face_8_2", "2face_78112", "2face_comb0", "face_comb0", "3face_78112", "num0_8_2",
                 "sozinha_prelim", "busca_local"):
        curva += [(nome, ()), (nome, R4)]
    curva += [("2face_8_2", tuple(p["regras"])) for p in camB]
    curva += [("2face_8_2", (r,)) for r in R4]
    curva += [("sozinha_int", R4 + ("fu_alto",))]
    vistos, lista = set(), []
    for nome, regs in curva:
        regs = tuple(r for r in ORDEM_EMENDAS if r in regs)
        if (nome, regs) in vistos:
            continue
        vistos.add((nome, regs))
        lista.append((nome, regs))
    ex = ProcessPoolExecutor()
    av.ex = ex
    vals = av.lote([chave(PRECOS[n] if n else None, r) for n, r in lista])
    ex.shutdown()
    itens_l = [itens(n, r) for n, r in lista]
    nreg_l = [n_regras(n, r) for n, r in lista]

    def fronteira(eixo):
        ordem = sorted(range(len(lista)), key=lambda i: (eixo[i], -vals[i]))
        fr, mx = [], None
        for i in ordem:
            if mx is None or vals[i] > mx:
                fr.append(i)
                mx = vals[i]
        return fr

    fr_it, fr_nr = fronteira(itens_l), fronteira(nreg_l)
    pontos = []
    for i, (n, r) in enumerate(lista):
        pontos.append({"id": rotulo(n, r), "precos": n, "tabela_precos": PRECOS[n] if n else None,
                       "regras": list(r), "itens_cola": itens_l[i], "n_regras": nreg_l[i],
                       "valor": fl(vals[i], 9), "valor_exato": str(vals[i]), "ganho_recuperado": fl(ganho(vals[i]), 6),
                       "pareto_itens": i in fr_it, "pareto_regras": i in fr_nr, "texto": texto_cola(n, r)})
        NUM[f"curva_{rotulo(n, r)}"] = vals[i]
    NUM["curva_n"] = len(lista)
    NUM["curva_pareto_itens"] = [rotulo(*lista[i]) for i in fr_it]
    NUM["curva_pareto_regras"] = [rotulo(*lista[i]) for i in fr_nr]
    print(f"  {len(lista)} colas; fronteira (itens): " + " · ".join(f"{itens_l[i]}→{float(vals[i]):.2f}" for i in fr_it))
    print("  fronteira (nº de regras): " + " · ".join(f"{nreg_l[i]}→{float(vals[i]):.2f}" for i in fr_nr))
    salva("curva_colas.json", {
        "E_otimo": fl(Eot, 9), "E_gulosa": fl(Eg, 9),
        "tetos_nao_executaveis": {**{f"rodada_perfeita_{k}": fl(v, 9) for k, v in tetoC.items()},
                                  "otimo_tabela_773388_entradas": fl(Eot, 9)},
        "colas": pontos,
        "emendas": {r: EMENDAS[r] for r in ORDEM_EMENDAS},
        "forward_A": [{"emenda": p["emenda"], "itens": p["itens"], "valor": fl(p["valor"], 9),
                       "candidatos": {k: fl(v, 9) for k, v in p["candidatos"].items()}} for p in camA],
        "forward_B": [{"emenda": p["emenda"], "itens": p["itens"], "valor": fl(p["valor"], 9),
                       "candidatos": {k: fl(v, 9) for k, v in p["candidatos"].items()}} for p in camB],
        "shapley": {nome: {j: fl(v, 9) for j, v in c["shapley"].items()} for nome, c in colas_B.items()},
        "precos_mq": [float(x) for x in pmq_r], "precos_marginais_cheia": [fl(x, 6) for x in marg],
    })

    # ------------------------------------------------------------- distribuição por General
    print("\n== v2 · Distribuição final por General feito / não feito")
    COLA = politica_cola(PRECOS["2face_8_2"], R4)
    distG = {}
    x = np.arange(D.PMAX)
    for nome, pol, dref in (("otima", POL, dist_ot), ("gulosa", GP, dist_g)):
        nao, sim = dist_por_general(pol)
        assert np.abs(nao + sim - dref).max() < 1e-12
        distG[nome] = {}
        for rot, p in (("nao_feito", nao), ("feito", sim)):
            P = p.sum()
            m = (x * p).sum() / P
            reg(f"dg_{nome}_{rot}_p", float(P))
            reg(f"dg_{nome}_{rot}_media", float(m))
            reg(f"dg_{nome}_{rot}_moda", int(p.argmax()))
            reg(f"dg_{nome}_{rot}_dp", float(np.sqrt(((x - m) ** 2 * p).sum() / P)))
            nz = np.nonzero(p > 0)[0]
            distG[nome][rot] = {"p": fl(P, 12), "media": fl(m, 9), "moda": int(p.argmax()),
                                "min": int(nz[0]), "max": int(nz[-1]),
                                "dist": [float(f"{v:.6e}") for v in p[: nz[-1] + 1]]}
        reg(f"dg_{nome}_dif_medias", distG[nome]["feito"]["media"] - distG[nome]["nao_feito"]["media"])
    salva("distribuicao_por_general.json", {
        "descricao": "Distribuição exata (float64) da pontuação final, separada por General feito (marcado com "
                     "pontos > 0) ou não. 'dist'[s] = P(pontuação = s e condição) — soma = 'p'.",
        **distG})

    # ------------------------------------------------------------- ordem de preenchimento
    print("\n== v2 · Ordem de preenchimento (exata)")
    ordem = {}
    for nome, pol in (("otima", POL), ("gulosa", GP), ("cola_ii", COLA)):
        P = ordem_preenchimento(pol)
        for b in range(NB):
            assert sum(P[b]) == 1
        for t in range(NB):
            assert sum(P[b][t] for b in range(NB)) == 1
        med = [sum((t + 1) * P[b][t] for t in range(NB)) for b in range(NB)]
        for b in range(NB):
            reg(f"ord_{nome}_rodada_media_{b}", med[b], NOMES[b])
        reg(f"ord_{nome}_P1_{7}", P[7][0], "P(Seguida preenchida na rodada 1)")
        reg(f"ord_{nome}_P10_{9}", P[9][9], "P(General preenchido na rodada 10)")
        reg(f"ord_{nome}_P10_{7}", P[7][9], "P(Seguida preenchida na rodada 10)")
        ordem[nome] = {"P": [[fl(P[b][t], 9) for t in range(NB)] for b in range(NB)],
                       "rodada_media": [fl(v, 9) for v in med]}
    salva("ordem_preenchimento.json", {
        "descricao": "P[b][t] = probabilidade de a casa b (0=Ás..9=General) ser preenchida (marcada ou riscada) na "
                     "rodada t+1. Linhas e colunas somam 1. Exato (racional), arredondado a 1e-9.",
        "casas": NOMES, "politicas": {"otima": "política ótima (H6)", "gulosa": "G1–G4",
                                      "cola_ii": texto_cola("2face_8_2", R4)}, **ordem})

    # ------------------------------------------------------------- preço marginal
    print("\n== v2 · Preço marginal E*(S) - E*(S \\ b) ao longo da partida (ótima)")
    alcO, _, _ = visitacao_exata(POL)
    reg("f2_cartelas_alcancaveis_otima", sum(1 for S in ORDEM if alcO.get(S, 0)), "de 1023 (não vazias)")
    for nome, pol in (("gulosa", GP), ("cola_ii", COLA)):
        a, _, _ = visitacao_exata(pol)
        reg(f"f2_cartelas_alcancaveis_{nome}", sum(1 for S in ORDEM if a.get(S, 0)), "de 1023")
    M = preco_marginal_por_rodada(E, alcO)
    for b in range(NB):
        assert M[b][0] == marg[b]
        assert M[b][NB - 1] == SOZINHA[b]  # última rodada: só b livre -> valor da casa sozinha (C10–C14)
        reg(f"pm_rodada1_{b}", M[b][0], NOMES[b])
        reg(f"pm_rodada5_{b}", M[b][4], NOMES[b])
        reg(f"pm_media_{b}", sum(M[b]) / NB, "média simples das 10 rodadas")
    salva("preco_marginal.json", {
        "descricao": "preco[b][t] = E[E*(S) - E*(S sem b) | b livre no início da rodada t+1], sob a política ótima "
                     "(média ponderada pela probabilidade de cada cartela). Rodada 1 = cartela cheia; rodada 10 = "
                     "valor da casa jogada sozinha (C10–C14). 'E_cartela' = E*(S) para as 1024 cartelas (bit b = "
                     "casa b livre); 'marginal'[S] = lista de E*(S)-E*(S sem b) para b em S (ordem crescente de b).",
        "casas": NOMES,
        "preco": [[fl(M[b][t], 9) for t in range(NB)] for b in range(NB)],
        "E_cartela": [fl(valor(E, S), 9) if S else 0.0 for S in range(1 << NB)],
        "marginal": [[fl(valor(E, S) - (valor(E, S & ~(1 << b)) if S & ~(1 << b) else 0), 6)
                      for b in D.caixas(S)] for S in range(1 << NB)]})

    # ------------------------------------------------------------- pergunte ao ótimo
    print("\n== v2 · 'Pergunte ao ótimo'")
    escolha = [FULL]
    kpor = {9: 3, 8: 3, 7: 2, 6: 2, 5: 2, 4: 2, 3: 2, 2: 2, 1: 2}
    for n, k in kpor.items():
        cand = sorted((S for S in ORDEM if POP[S] == n), key=lambda S: -alcO.get(S, 0))
        escolha += cand[:k]
    cartelas = []
    for S in escolha:
        dt = det_float(S)
        lanc = []
        for r in range(3):
            o = [int(a) for a in dt["P"][r]]
            g = [int(a) for a in GP[S][r]]
            c = [int(a) for a in COLA[S][r]]
            V = dt["V"][r]
            lanc.append({"otima": o, "v_otima": [round(v, 4) for v in V],
                         "gulosa": g, "perda_gulosa": [round(V[d] - q_float(S, r, d, g[d]), 4) for d in range(ND)],
                         "cola": c, "perda_cola": [round(V[d] - q_float(S, r, d, c[d]), 4) for d in range(ND)]})
        cartelas.append({"livres": D.caixas(S), "rodada": NB - POP[S] + 1, "prob_otima": fl(p_alc(alcO, S), 6),
                         "E_cartela": fl(valor(E, S), 6), "lancamentos": lanc})
    salva("pergunte_ao_otimo.json", {
        "descricao": "Para cada cartela, lançamento (1º, 2º, 3º) e mão (252 multiconjuntos), a ação ótima, o valor "
                     "dela (pontos esperados do resto do jogo, incluindo esta rodada), e a ação e a perda (valor ótimo "
                     "- valor da ação seguida de jogo ótimo) da gulosa e da cola (ii).",
        "codigo_acao": "a < 0: marca a casa -(a+1); a >= 0: guarda guardas[a] e relança o resto",
        "criterio_cartelas": "cartela vazia + as cartelas mais prováveis sob a política ótima com 9, 8, …, 1 casas "
                             "livres (3, 3, 2, 2, 2, 2, 2, 2, 2 por nível)",
        "casas": NOMES, "maos": [mao_str(d) for d in range(ND)], "guardas": ["".join(map(str, k)) for k in KEEPS],
        "cola": texto_cola("2face_8_2", R4), "cartelas": cartelas})

    # ------------------------------------------------------------- replays
    print("\n== v2 · Replays (dados comuns)")
    reps = []
    for j in range(3):
        sem = 20261002 + j
        U = np.random.default_rng(sem).integers(1, 7, (NB, 3, 5))
        rep = {"semente": sem, "dados_sorteados": U.tolist()}
        for nome, pol in (("otima", POL), ("gulosa", GP), ("cola_ii", COLA)):
            rep[nome] = joga(pol, U)
        reps.append(rep)
        print(f"  semente {sem}: ótima {rep['otima']['total']}, gulosa {rep['gulosa']['total']}, "
              f"cola {rep['cola_ii']['total']}")
        NUM[f"replay_{sem}"] = [rep[n]["total"] for n in ("otima", "gulosa", "cola_ii")]
    salva("replays.json", {
        "descricao": "3 partidas jogadas pela ótima, pela gulosa e pela cola (ii) com os MESMOS dados sorteados "
                     "(números aleatórios comuns): U[t][r][i] = valor do dado na posição i no lançamento r da rodada "
                     "t; um dado relançado na posição i no lançamento r recebe U[t][r][i]; dados guardados ocupam as "
                     "primeiras posições com aquela face. O 1º lançamento de cada rodada é igual para as três; depois "
                     "divergem só quando as guardas divergem. 'custo' = valor ótimo do estado - valor da ação tomada "
                     "seguida de jogo ótimo.",
        "codigo_acao": "a < 0: marca a casa -(a+1); a >= 0: guarda guardas[a]",
        "casas": NOMES, "guardas": ["".join(map(str, k)) for k in KEEPS], "partidas": reps})
    print(f"  (v2 total: {time.perf_counter() - t0:.0f} s)")
