# Modelo: Estratégia, Bozó e Cadeias de Markov

> Matemático, modo execução, 2026-10-01. Abordagens **A + B + C + D1** (pauta › E1–E4).
> Todo número abaixo sai de `derivacao.py` (`uv run python posts/estrategia-bozo-markov/_work/derivacao.py`),
> que também grava `resultados/numeros.json` e confere `claims.yaml`. O treino de RL (D1) fica em cache
> em `resultados/rl_curva.json`; para refazer: `... derivacao.py --refazer-rl` (≈ 25 min em 12 núcleos).

## 1. Regras formalizadas

Regras da pauta (Piano & Toillier 2010) com D1–D6 e E3. Notação: casas
$0..9$ = Ás, Duque, Terno, Quadra, Quina, Sena, Fú, Seguida, Quadrada, General.
Uma mão é o **multiconjunto** dos 5 dados (252 mãos; a ordem não importa).

Pontos de marcar a casa $b$ com a mão $d$, $\text{boca}\in\{0,1\}$:

$$
\text{pts}(b,d,\text{boca}) =
\begin{cases}
(b+1)\cdot\#\{\text{dados com face } b+1\} & b\le 5\\
(20,30,40,50)_b + 5\cdot\text{boca} & b\in\{6,7,8,9\} \text{ e } d \text{ válida para } b\\
0 & \text{caso contrário (riscar)}
\end{cases}
$$

Validade: Fú = perfil exato $\{3,2\}$; Quadrada = perfil exato $\{4,1\}$ (D4: cinco iguais não vale
como Fú nem Quadrada); Seguida = $\{1,2,3,4,5\}$ ou $\{2,3,4,5,6\}$ (D5); General = cinco iguais.

Hipóteses de modelagem (onde a pauta não fixa um detalhe):

- **H1 — Objetivo:** maximizar a pontuação total esperada do jogo solo de 10 rodadas (pauta).
- **H2 — Boca:** o +5 vale quando a combinação é marcada **logo após o 1º lançamento**, sem relançar
  nada; só para Fú, Seguida, Quadrada e General (D1: General de boca = 55).
- **H3 — Parar = marcar.** "Guardar os 5 e relançar zero" não é uma ação. Isso não muda nenhum valor:
  essa ação é dominada (perde a boca e um lançamento).
- **H4 — Guardas (D3):** após o 1º e o 2º lançamentos o jogador guarda **qualquer** submulticonjunto de
  0 a 4 dados da mão atual e relança o resto. Um dado guardado antes pode voltar ao copo.
- **H5 — Riscar livre (D6):** qualquer casa livre pode ser marcada a qualquer momento, inclusive com 0.
  Casa de número sem a face vale 0. Com cinco iguais, as opções úteis são General ou a casa de número.
- **H6 — Desempate** (para a política ser uma função): (i) marcar antes de relançar; (ii) entre casas,
  a de menor índice; (iii) entre guardas, a com mais dados, depois a de maior soma, depois a maior
  tupla em ordem decrescente. **O desempate não altera** o valor, a distribuição da pontuação
  final nem as probabilidades de completar casas (seção C, verificado com o desempate oposto).
- **H7 — BAIXO fora (E3):** pedido às cegas, $7-X$ tem a mesma distribuição uniforme que $X$, então a
  regra não muda nenhuma probabilidade. Nenhuma variante "depois de ver" é calculada.

## 2. Modelo

**Estado entre rodadas:** o conjunto $S$ de casas livres ($2^{10}=1024$ cartelas). A pontuação já
feita não entra no estado (H1, sem bônus): o futuro depende só de $S$.

**Dentro da rodada:** estado de decisão $(S, r, d)$ com $r\in\{1,2,3\}$ lançamentos feitos e $d$ a mão.
São $1023\times3\times252 = 773\,388$ estados de decisão. Ações: marcar $b\in S$ (sempre $|S|$ opções,
H5) ou, se $r<3$, guardar $k\subsetneq d$ (até 31 opções). Guardar $k$ leva a $d'$ com probabilidade
$T(k,d') = \#\{\text{sequências de } 5-|k| \text{ dados que completam } k \text{ em } d'\}/6^{5-|k|}$.

**Equação de Bellman (programação dinâmica por rodada).** Com $E^*(\varnothing)=0$, em ordem
crescente de $|S|$:

$$
\begin{aligned}
V_3(d) &= \max_{b\in S}\big[\text{pts}(b,d,0) + E^*(S\setminus b)\big]\\
V_2(d) &= \max\Big(\max_{b\in S}\big[\text{pts}(b,d,0)+E^*(S\setminus b)\big],\ \max_{k\subsetneq d}\textstyle\sum_{d'}T(k,d')V_3(d')\Big)\\
V_1(d) &= \max\Big(\max_{b\in S}\big[\text{pts}(b,d,1)+E^*(S\setminus b)\big],\ \max_{k\subsetneq d}\textstyle\sum_{d'}T(k,d')V_2(d')\Big)\\
E^*(S) &= \textstyle\sum_d P_0(d)\,V_1(d), \qquad P_0(d) = \frac{5!}{\prod_f m_f(d)!}\,6^{-5}.
\end{aligned}
$$

**Exatidão.** Todos os valores de $E^*(S)$ com $|S|=n$ são racionais com denominador que divide
$6^{15n}$. A DP roda em **inteiros exatos** (Python `int`), escalando $E^*(S)$ por $6^{15|S|}$ e cada
nível intra-rodada por $6^5$. O jogo inteiro sai **exato** em ≈ 3 s, então E4 (float com tolerância)
nem foi preciso para os valores. Só as distribuições (propagação para frente) usam float64, com
erro < 1e-12; o script confere média da distribuição = $E^*$ a 1e-9.

**Avaliação de política fixa** (gulosa, política aprendida pelo RL): a mesma recursão, trocando cada
$\max$ pela ação da política. Também é exata.

**Distribuição da pontuação final:** propagação para frente de $\Pr(S, \text{pontos})$, usando a
distribuição exata de (casa marcada, pontos) de cada rodada sob a política.

### Heurística gulosa (B), definição reprodutível

Após cada lançamento $r$, para cada casa livre calcula-se $\text{pts}(b,d,[r=1])$.

- **G1** $\hat b$ = casa livre de maior pontuação imediata; **G2** empate → menor índice. Portanto,
  quando tudo dá 0, risca-se a casa livre de menor índice (Ás primeiro, General por último).
- **G3** Para e marca $\hat b$ se: $r=3$; ou $\hat b$ é combinação (Fú/Seguida/Quadrada/General) com
  pontos > 0; ou a mão tem cinco iguais.
- **G4** Senão, guarda todos os dados da face mais frequente (empate → a face maior; se todas
  diferem, guarda um dado, o maior) e relança o resto.

## 3. Resultados

### C1 {#c1}
$P(\text{General de boca}) = 6/7776 = \mathbf{1/1296}$. Por enumeração dos $6^5$ lançamentos
(`dice_pattern_probability`) e pela soma de $P_0$ nas mãos válidas (iguais).

### C2 {#c2}
$P(\text{Quadrada de boca}) = 6\cdot5\cdot5/7776 = \mathbf{25/1296}$ (face da quadra × face do avulso × posição).

### C3 {#c3}
$P(\text{Fú de boca}) = 6\cdot5\cdot\binom52/7776 = \mathbf{25/648}$.

### C4 {#c4}
$P(\text{Seguida de boca}) = 2\cdot5!/7776 = \mathbf{5/162}$. Alguma combinação de boca: $29/324 \approx 0{,}0895$.

### C5 {#c5}
Máxima probabilidade de fazer **General** em até 3 lançamentos (jogando só para isso):
$\mathbf{347897/7558272} \approx 0{,}046029$. Duas rotas exatas e iguais:
(i) recursão de máximo sobre as 252 mãos; (ii) cadeia de Markov absorvente no "nº máximo de dados
iguais" (estados 1..5) com a regra "guarda a face mais frequente":

$$
P=\begin{pmatrix}
5/54 & 25/36 & 125/648 & 25/1296 & 1/1296\\
0 & 5/9 & 10/27 & 5/72 & 1/216\\
0&0&25/36&5/18&1/36\\
0&0&0&5/6&1/6\\
0&0&0&0&1
\end{pmatrix},\quad \pi_0 = \text{linha 1},\quad P(\text{General}\le 3) = (\pi_0 P^2)_5 .
$$

### C6 {#c6}
Máx. $P(\text{Quadrada em até 3}) = \mathbf{1042585/3779136} \approx 0{,}27588$ (D4: exatamente 4+1).

### C7 {#c7}
Máx. $P(\text{Fú em até 3}) = \mathbf{5485535/15116544} \approx 0{,}36288$.

### C8 {#c8}
Máx. $P(\text{Seguida em até 3}) = \mathbf{319695199/1224440064} \approx 0{,}26110$.

### C9 {#c9}
Sem limite de lançamentos, com a regra da cadeia de C5, o número esperado de lançamentos até o
General é $1 + \pi_{0,T}^\top (I-Q)^{-1}\mathbf 1 = \mathbf{191283/17248} \approx 11{,}090$
(`mathbox.absorbing_analysis`).

### C10 {#c10}
Casa de número $k$ jogada sozinha (só ela livre, uma rodada): $E = 5k\,(1-(5/6)^3) = \mathbf{455k/216}$
(Ás 2,106; Duque 4,213; Terno 6,319; Quadra 8,426; Quina 10,532; Sena 12,639). Cada dado vira $k$ com
probabilidade $1-(5/6)^3$ guardando todos os $k$; a DP exata confirma que isso é ótimo.

### C11 {#c11}
Fú sozinho: $E = 20\,p_3 + 5\,p_1 = \mathbf{28156675/3779136} \approx 7{,}4506$ ($p_3$ de C7, $p_1$ de C3).
A maximização de pontos coincide com a de probabilidade, porque parar ao fazer de boca é sempre ótimo.

### C12 {#c12}
Seguida sozinha: $30\,p_3+5\,p_1 = \mathbf{1629968795/204073344} \approx 7{,}9872$.

### C13 {#c13}
Quadrada sozinha: $40\,p_3+5\,p_1 = \mathbf{10516975/944784} \approx 11{,}1316$.

### C14 {#c14}
General sozinho: $50\,p_3+5/1296 = \mathbf{8712005/3779136} \approx 2{,}30529$.
**Ponte A ↔ C:** a DP do jogo completo, avaliada nas 10 cartelas com uma só casa livre, reproduz
exatamente C10–C14 (assert no script).

### C15 {#c15}
Soma das 10 casas jogadas sozinhas: $\mathbf{14919955235/204073344}\approx 73{,}11$, menos da metade
do valor do jogo (C17).

### C16 {#c16}
O MDP tem $1023\times3\times252 = \mathbf{773\,388}$ estados de decisão; com as guardas (462
multiconjuntos de 0–5 dados), resolvido exatamente em ~3 s.

### C17 {#c17}
**Valor do Bozó solo com jogo ótimo:** $E^* \approx \mathbf{146{,}72838056}$ (racional exato; o
denominador tem 98 dígitos; a fração está em `resultados/numeros.json`). Conferido por Monte Carlo
da política ótima tabelada: $146{,}685$, IC99% $[146{,}604;\ 146{,}766]$, $n=10^6$, semente 20261002.

### C18 {#c18}
Sem o bônus de boca (mesmas regras, +5 removido): $E^* \approx \mathbf{145{,}10126444}$.

### C19 {#c19}
**Valor do bônus de boca** sob jogo ótimo: $146{,}72838 - 145{,}10126 \approx \mathbf{1{,}62712}$ ponto.

### C20 {#c20}
Distribuição da pontuação final sob a política ótima (H6): média 146,728; desvio-padrão
**31,5055**; quantis 5% = 98, mediana = 146, 95% = 202; moda 147; suporte observado 1..265.
Vetor completo em `resultados/distribuicoes.npz`.

### C21 {#c21}
Probabilidade de cada casa terminar **com pontos > 0** sob a política ótima:

| Ás | Duque | Terno | Quadra | Quina | Sena | Fú | Seguida | Quadrada | General |
|---|---|---|---|---|---|---|---|---|---|
| 0,8460 | 0,9743 | 0,9833 | 0,9932 | 0,9931 | 0,9953 | 0,9196 | 0,7649 | 0,9112 | **0,2750** |

Com o desempate oposto em H6 essas probabilidades mudam < 3e-16 (só erro de arredondamento).

### C22 {#c22}
Sob a política ótima, a **última casa** a ser preenchida é o General com probabilidade **0,2348**
e a Seguida com 0,2364. As duas juntas somam 47%: as casas difíceis ficam para o fim, como "reservas"
para a última rodada.

### C23 {#c23}
**Recusar a Quadrada de boca.** Cartela vazia, 1º lançamento $\{5,6,6,6,6\}$: o ótimo **guarda
6666 e relança um dado** em vez de marcar Quadrada de boca (45). Valor das ações:
guardar 6666 ≈ 160,5594; marcar Quadrada ≈ 156,3127; **vantagem ≈ 4,2467** (exata).
O mesmo vale para toda quadra + avulso no 1º e no 2º lançamento da cartela vazia (o ótimo nunca
marca Quadrada antes do 3º lançamento). Intuição: se falhar, a Quadrada continua lá (40); se acertar
(11/36), faz o General, que é a casa mais rara.

### C24 {#c24}
**Desmanchar um Fú sem boca.** Cartela vazia, 2º lançamento $\{1,1,4,4,4\}$: o ótimo **guarda 444
e relança dois dados**, em vez de marcar Fú (20). Vantagem ≈ 0,08395 (apertada). Com
$\{1,1,1,4,4\}$ (trinca baixa) ele marca Fú, com vantagem ≈ 1,30973 sobre guardar 111.
No 1º lançamento, qualquer Fú (25, de boca) é marcado.

### C25 {#c25}
**Marcar 1 ponto em vez de 6.** Cartela vazia, 3º lançamento $\{1,2,3,4,6\}$: o ótimo marca **Ás (1)**
e não Sena (6). Vantagem ≈ 6,5926. A gulosa marca Sena. Gastar a Sena com um só 6 custa caro: a
Sena vale 12,6 sozinha e é a casa de número mais valiosa.

### C26 {#c26}
**Seguida aberta em vez de par.** Cartela vazia, 1º lançamento $\{2,2,3,4,5\}$: o ótimo guarda
**2345** (dois números completam a Seguida) e não o par 22. Vantagem ≈ 2,8176.

### C27 {#c27}
**Mapa de decisão, cartela vazia, 1º lançamento** (tabela completa das 252 mãos em
`resultados/mapa_decisao_cartela_vazia.csv`, com as ações nos 3 lançamentos e a da gulosa): o ótimo
marca já no 1º lançamento **exatamente** quando tem General, Fú ou Seguida de boca (38 mãos;
probabilidade $\mathbf{91/1296}\approx 0{,}0702$). Nunca para com Quadrada de boca (C23). Nas demais,
guarda: a trinca ou quadra se houver; uma Seguida aberta 2345 (C26) ou três dados do meio quando
os dados são todos diferentes sem Seguida (12346 → 234; 12356 → 235; 12456 → 245; 13456 → 345); um par nos outros casos. Com dois
pares no 1º lançamento guarda só o par mais alto (todas as 60 mãos de dois pares). No 2º lançamento, com dois pares, às vezes guarda os dois (caça ao Fú) e às
vezes um, conforme as faces: ver o CSV.

### C28 {#c28}
Na cartela vazia, ótimo e gulosa tomam a mesma decisão no 1º lançamento com probabilidade
$\mathbf{1151/1296}\approx 0{,}8881$. A diferença entre as duas está quase toda na **marcação**,
não nas guardas.

### C29 {#c29}
**Valor exato da gulosa:** $E_G \approx \mathbf{120{,}66503746}$ (racional exato, avaliação da política
pela mesma recursão).

### C30 {#c30}
**Custo de jogar no instinto:** $E^* - E_G \approx \mathbf{26{,}0633}$ pontos por partida (17,8% do ótimo).

### C31 {#c31}
Monte Carlo vetorizado da gulosa ($n=10^6$, semente 20261001): média **120,633**, IC99%
$[120{,}549;\ 120{,}716]$, que contém o valor exato de C29. Desvio-padrão amostral 32,39.

### C32 {#c32}
Distribuição exata da pontuação final da gulosa: média 120,665; desvio-padrão **32,3876**; quantis
5% = 72, mediana = 114, 95% = 180; moda 100.

### C33 {#c33}
Probabilidade de cada casa terminar com pontos > 0 sob a gulosa:

| Ás | Duque | Terno | Quadra | Quina | Sena | Fú | Seguida | Quadrada | General |
|---|---|---|---|---|---|---|---|---|---|
| 0,5722 | 0,7851 | 0,8712 | 0,9204 | 0,9511 | 0,9742 | 0,8584 | **0,2587** | 0,9282 | 0,2983 |

A gulosa faz **mais** Generais e Quadradas que o ótimo, mas perde a Seguida (25,9% × 76,5%) e as
casas baixas.

### C34 {#c34}
Numa partida ótima contra uma partida gulosa independentes, o ótimo faz mais pontos com
probabilidade **0,71665** (empate 0,00784).

### C35 {#c35} — Política não única
Há **9441** situações de decisão $(S,r,d)$ com mais de uma ação ótima (empate exato em racionais).
A seção H6 fixa o desempate. O desempate oposto dá o mesmo $E^*$ (exato) e a mesma distribuição
final (diferença < 3e-16).

### C36 {#c36} — Bozó reduzido (D1)
**Jogo:** só **Sena, Seguida e General** livres (3 rodadas, mesmas regras). **Por que essas três:**
(i) uma casa de número e duas combinações, então há marcação e risco de verdade; (ii) as guardas
conflitam (Sena e General querem faces repetidas, a Seguida quer faces distintas); (iii) a gulosa é
muito ruim aqui, então há o que aprender; (iv) com 3 casas o espaço tabular fica em
$7\times3\times252 = 5292$ estados e **66 696** pares (estado, ação), viável em Python puro.
Com 4 casas o número de pares mais que dobra, e o treino (já de ~36 min de relógio em 12 núcleos)
ficaria longo demais.
Ótimo exato: $E^* = 32{,}39343$ (fração exata, mesma DP de C17). Gulosa: $15{,}16424$.

### C37 {#c37}
**Configuração** (`tools/rl.py`, `q_learning`): passo `alpha="visitas"` ($1/N(s,a)^{0{,}8}$); `q_init = 120`
(maior retorno possível, otimista); treino com **exploring starts** (cada partida começa num estado
de decisão uniforme) e **ε = 0,5 fixo**. O Q-learning é off-policy: ele aprende $Q^*$ mesmo com o
comportamento explorando muito. Com o ε padrão do `rl.py` (decai a 0,05 em 5 mil partidas), testes
preliminares deixaram estimativas de guarda "presas" num único azar (o primeiro passo é 1 e a ação
não é revisitada), e a política ficou muito pior. Por isso o ε é fixo. **Avaliação:** política gulosa em
$Q$ (`greedy_policy`; pares nunca vistos = $-\infty$) avaliada **exatamente** no jogo normal pela
mesma recursão da DP (sem ruído de Monte Carlo).

Após **3 milhões** de partidas de treino, 5 sementes: valor **27,398** (dp 0,359; mín 27,123; máx
28,011), a **4,995** pontos do ótimo (**84,6%**); 76,0% das 5292 decisões coincidem com a política
ótima (H6).

### C38 {#c38}
**Curva de aprendizado** (cada orçamento treinado do zero; média de 5 sementes; `resultados/rl_curva_aprendizado.csv`):

| partidas de treino | 10 mil | 30 mil | 100 mil | 300 mil | 1 milhão | 3 milhões | ótimo | gulosa |
|---|---|---|---|---|---|---|---|---|
| valor da política aprendida | 6,73 | 11,07 | 15,29 | **17,58** | 21,85 | 27,40 | 32,39 | 15,16 |
| dp entre sementes | 0,29 | 0,56 | 0,69 | 0,50 | 0,17 | 0,36 | | |
| % decisões = ótima | 21,9 | 29,8 | 35,4 | 43,8 | 58,8 | 76,0 | | |

Com 100 mil partidas o agente empata com a gulosa. Com 300 mil, todas as sementes passam dela (mín
16,96). Com 3 milhões ainda falta ~15% para o ótimo. **Ponte N5 → N4:** a curva sobe monotonamente
na direção do ótimo exato da DP, mas o RL tabular é caro: 3 milhões de partidas para um jogo de 3
casas que a DP resolve exatamente em milissegundos.

## 4. Simplificações e limites

- Jogo **solo**, maximizando pontos (H1). Contra adversários, maximizar a chance de vencer é outro
  problema (o estado precisaria incluir os placares).
- Política ótima não é única (C35). As afirmações de política citam a ação e a **vantagem exata** sobre
  a alternativa, o que é verificável com qualquer desempate.
- As distribuições são float64 (propagação para frente). Os valores esperados são racionais exatos.
- A gulosa é **uma** formalização do "jeito intuitivo" (E2). Outra regra de risco (ex.: riscar o
  General primeiro) daria outro número.
- D1 usa um jogo reduzido de 3 casas: o Q-learning tabular no jogo completo (~1,7 M estados × ~20
  ações) é inviável em Python.

## 5. Ideias para o post

- **Gancho numérico:** as 10 casas jogadas "uma de cada vez, com alvo fixo" somam 73 pontos (C15);
  o jogo ótimo vale 146,7 (C17). Saber **escolher a casa** dobra o placar.
- **Do Yahtzee ao Bozó:** a cadeia 5×5 do General (C5) é o exemplo clássico de cadeia absorvente;
  11,09 lançamentos em média até o General sem limite (C9), contra 4,6% em três.
- **Tamanho do problema:** 773 mil estados de decisão, resolvidos *exatamente* (frações com 98 dígitos)
  em poucos segundos. A decomposição por rodada é o truque didático: só importa *quais casas sobram*.
- **Surpresas da política:** recusar 45 pontos garantidos (C23); desmanchar um Fú por uma trinca de
  4, 5 ou 6 (C24); marcar 1 ponto em vez de 6 (C25). Boa estrutura para o post: "três jogadas que
  parecem erro".
- **O General é reserva, não meta:** o ótimo só completa o General em 27,5% das partidas (C21), menos
  que a gulosa (29,8%, C33), e o deixa para o fim em 23% delas (C22).
- **A gulosa erra marcando, não guardando:** 89% das decisões do 1º lançamento coincidem (C28). O
  custo de 26 pontos vem de gastar casas (Sena com um 6, Quina com um 5) e da Seguida ignorada.
- **Gráficos:** (1) histogramas sobrepostos ótima × gulosa (`distribuicoes.npz`); (2) barras pareadas
  "chance de completar cada casa" (C21 × C33); (3) mapa de decisão do 1º lançamento como grade
  (`mapa_decisao_cartela_vazia.csv`, cores por tipo de guarda); (4) curva de aprendizado do RL com
  as retas do ótimo e da gulosa (`rl_curva_aprendizado.csv`); (5) o grafo 5×5 da cadeia do General.
- **Boca vale pouco:** 1,63 ponto (C19). O ótimo até recusa uma das bocas (Quadrada).
- **RL como epílogo humilde:** o agente que só joga precisa de ~100 mil partidas para alcançar a
  gulosa e de 3 milhões para chegar a 85% do ótimo num Bozó de 3 casas (C38). A conta exata resolve
  o jogo inteiro em segundos. Isso mostra por que a estrutura de Markov (decompor por rodada) vale ouro.
