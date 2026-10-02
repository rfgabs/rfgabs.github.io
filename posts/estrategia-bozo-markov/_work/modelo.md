# Modelo: Estratégia, Bozó e Cadeias de Markov

> Matemático, modo execução, 2026-10-01. Abordagens **A + B + C + D1** (pauta › E1–E4).
> **v2 (2026-10-02):** pauta › Versão 2, escolhas H1–H6 da pauta: §0 (lema) + A (emendas) + B (colas de
> preços) + C (rodada perfeita, teto) + C′ (iteração de política), formulação F2 (seção 2b), afirmações
> C39–C59 (seção 3, "Resultados v2") e produtos de visualização em `resultados/v2/` (ver o README de lá).
> Todo número abaixo sai de `derivacao.py` (`uv run python posts/estrategia-bozo-markov/_work/derivacao.py`;
> a v2 está em `derivacao_v2.py`, chamado por ele), que também grava `resultados/numeros.json` e confere
> `claims.yaml`. Tempo total ≈ 7 min (12 núcleos; a v2 avalia centenas de políticas exatamente, em paralelo).
> O treino de RL (D1, arquivado) fica em cache em `resultados/rl_curva.json`; para refazer:
> `... derivacao.py --refazer-rl` (≈ 25 min em 12 núcleos).
>
> **Duas listas de "H".** H1–H7 (e H8–H14 da v2) abaixo são hipóteses de modelagem deste arquivo; as
> escolhas H1–H6 da pauta (Versão 2) são citadas como "pauta › H*".

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

Hipóteses da v2 (escolhas que a pauta não fixa e que mudam números):

- **H8 — Colas.** Uma cola é a gulosa G1–G4 com (a) marcação por "pontos − preço" no lugar de G1–G2 e/ou
  (b) emendas de guarda/parada. Texto executável completo em `claims.yaml › colas` (é o contrato com o
  validador). Desempate entre casas empatadas em pontos − preço: menor índice (pauta › H5). A parada (G3) é
  decidida sobre a casa escolhida pela regra de marcação da cola.
- **H9 — Preços da cola (i).** "Valor da casa jogada sozinha" (C10–C14) arredondado ao **inteiro mais
  próximo**: Ás 2, Duque 4, Terno 6, Quadra 8, **Quina 11**, Sena 13, Fú 7, Seguida 8, Quadrada 11,
  General 2. A tabela preliminar de `abordagens-v2.md` tinha Quina 10 (10,53 arredondado para baixo, um
  deslize); as duas entram na curva (C55): 141,1730 (Quina 11) × 141,2054 (Quina 10).
- **H10 — Itens de cola** (pauta › H4): seg4x 3 (Seguida livre? · 4 faces de uma Seguida? · qual Seguida),
  quad_gen 2 (General livre? · 1º/2º lançamento?), alvo 2 (a face serve? · e se nenhuma serve), risca_gen 1,
  doispar 3 (Fú livre? · 2º lançamento? · dois pares?), fu_alto 3; tabela de 10 preços + argmax = 11;
  "2 × face / 8 / 2" + argmax = 4. Checagem de robustez: nº de regras (1 por tabela, 1 por emenda).
- **H11 — Shapley com 6 jogadores.** Para ter $2^6 = 64$ coalizões, a tabela de preços entra como dois
  jogadores (preços das casas de número; preços das combinações); jogador ausente = preço 0. Sem nenhum
  preço, a marcação é a da gulosa (G1–G2 = argmax de pontos − 0 com o mesmo desempate).
- **H12 — Lema por categoria.** A parcela de cada estado é atribuída a (lançamento, tipo da ação da gulosa,
  tipo da ação da ótima com desempate H6). A soma das parcelas não depende do desempate; a repartição da
  forma (ii) depende.
- **H13 — "Pergunte ao ótimo":** cartela vazia + as cartelas mais prováveis sob a ótima com 9, 8, …, 1
  casas livres (3, 3, 2, 2, 2, 2, 2, 2, 2 por nível; 21 cartelas). "Típica" = mais provável sob a ótima.
- **H14 — Replays:** números aleatórios comuns, `default_rng(20261002 + j)`, $j = 0, 1, 2$; um dado na
  posição $i$ relançado no lançamento $r$ da rodada $t$ recebe $U[t,r,i]$; dados guardados ocupam as
  primeiras posições com a face guardada. O 1º lançamento de cada rodada é igual para as três políticas.

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

## 2b. Formulação em degraus (F2, v2)

Seis degraus, cada um apoiado numa afirmação verificável. A ordem sugerida para o texto é 1 → 2 → 3 (ainda
com a gulosa) → 4 → 5 → resultados da v1 → 6.

**Degrau 1 — cadeia de Markov: ninguém decide.** Uma cadeia de Markov é uma sequência $X_0, X_1, \dots$ num
conjunto finito em que o próximo estado só depende do atual:
$\Pr(X_{t+1}=j \mid X_t=i, X_{t-1},\dots) = P_{ij}$. No Bozó: fixe a regra "guarde a face mais frequente" e olhe
só para $X$ = nº máximo de dados iguais (1..5). A matriz $5\times5$ de C5 descreve tudo; o General em até três
lançamentos é $(\pi_0P^2)_5$ (C5) e o tempo médio até o General sem limite sai da matriz fundamental
$(I-Q)^{-1}$ (C9). A regra de jogo está **dentro** da matriz: ninguém escolhe nada.

**Degrau 2 — MDP: alguém escolhe.** Um processo de decisão de Markov é $(\mathcal S,\mathcal A,P,R)$:

- $\mathcal S=\{(S,r,d)\}$: $S$ = casas livres (1023 não vazias), $r\in\{1,2,3\}$ = lançamentos feitos,
  $d$ = mão (252 multiconjuntos). $|\mathcal S| = 773\,388$ (C16), mais o terminal $S=\varnothing$.
- $\mathcal A(S,r,d) = \{\text{marcar } b : b\in S\}\cup\{\text{guardar } k : k\subsetneq d\}$, a 2ª parte só
  se $r<3$. São $12\,292\,056$ pares (estado, ação): $3\,870\,720$ de marcar e $8\,421\,336$ de guardar
  (em média $49/3$ guardas por mão) (C59).
- $P$: guardar $k$ leva a $(S, r+1, d')$ com probabilidade $T(k,d')$ (multinomial do relançamento); marcar $b$
  leva a $(S\setminus b, 1, d')$ com probabilidade $P_0(d')$.
- $R$: marcar $b$ rende $\text{pts}(b,d,[r=1])$; guardar rende 0.

É Markov porque o futuro depende só de $(S,r,d)$: o objetivo é a **soma** dos pontos (sem bônus nem
limiar), então o placar acumulado não muda nenhuma decisão. O horizonte é finito: no máximo 30 decisões.

**Degrau 3 — fixar uma política gera uma cadeia de Markov.** Uma política (determinística) é uma tabela
$\pi:\mathcal S\to\mathcal A$. Com $\pi$ fixa, $P^\pi(s'\mid s)=P(s'\mid s,\pi(s))$ é uma matriz estocástica
comum: o MDP vira de novo uma cadeia de Markov (com recompensas). Olhando só o início de cada rodada, a
cartela $S_1=$ cheia, $S_2,\dots,S_{11}=\varnothing$ é uma cadeia de Markov nos $2^{10}$ subconjuntos, que só
desce ($|S_{t+1}|=|S_t|-1$) e é absorvida em $\varnothing$, com
$\Pr(S_{t+1}=S\setminus b\mid S_t=S)=\Pr(\pi \text{ marca } b \text{ na rodada}\mid S)$.
**Avaliar a gulosa (ou qualquer cola) é analisar essa cadeia.** Dela saem a ordem de preenchimento (C57),
a distribuição por General (C56) e as cartelas alcançáveis: sob a ótima, a gulosa e a cola (ii), todas as
1023 cartelas têm probabilidade positiva (C59).

**Degrau 4 — duas equações de Bellman.**

$$
\underbrace{V^\pi(s) = R(s,\pi(s)) + \sum_{s'}P(s'\mid s,\pi(s))\,V^\pi(s')}_{\text{avaliação: política fixa, sistema linear}}
\qquad
\underbrace{V^*(s) = \max_{a}\Big[R(s,a) + \sum_{s'}P(s'\mid s,a)\,V^*(s')\Big]}_{\text{otimalidade: com max}}
$$

e $Q^\pi(s,a)=R(s,a)+\sum_{s'}P(s'\mid s,a)V^\pi(s')$ ("faça $a$ agora e siga $\pi$ depois"). A de avaliação é
linear e **triangular**: toda transição aumenta $r$ ou diminui $|S|$, então se resolve de trás para frente
sem inverter matriz. Aplicada à gulosa ela dá exatamente C29; aplicada a $\pi^*$ dá $E^*(S)$ em todas as
1023 cartelas, a solução da de otimalidade (C59). Por rodada, $E^\pi(S)=\sum_d P_0(d)\,V^\pi(S,1,d)$.

**Degrau 5 — programação dinâmica e iteração de política.** A de otimalidade se resolve de trás para frente
em $|S|=1,2,\dots,10$ (seção 2; C17). Outra rota, que parte de uma política e a melhora:
**iteração de política** no nível da rodada, $\pi_{k+1}(S)$ = jogar a rodada de forma ótima com valor
terminal $E^{\pi_k}(S\setminus b)$. Pelo teorema de melhoria de política, $E^{\pi_{k+1}}\ge E^{\pi_k}$ em toda
cartela; e, por indução em $|S|$, depois de $k$ passos $E^{\pi_k}(S)=E^*(S)$ para todo $|S|\le k$ (no passo
$k+1$ a rodada em $|S|=k+1$ é jogada de forma ótima contra um futuro já ótimo). Logo bastam 10 passos.
A partir da gulosa bastam **5**: 120,665 → 142,437 → 146,087 → 146,700 → 146,7283 → $E^*$ exato, com 1, 20,
121, 443, 872 e 1023 cartelas já ótimas (C54; o mínimo garantido pela indução seria 0, 10, 55, 175, 385, 637).

**Degrau 6 — de volta à mesa: "pontos − preço" é o argmax de Bellman com o futuro aproximado.** Na decisão
de marcar, a equação de otimalidade escolhe

$$
b^\star=\arg\max_{b\in S}\big[\text{pts}(b,d)+E^*(S\setminus b)\big].
$$

Suponha o futuro **aditivo**: $E^*(S')\approx \tilde V(S') = c_{|S'|}+\sum_{b'\in S'}p_{b'}$. Para $b\in S$,

$$
\text{pts}(b,d)+\tilde V(S\setminus b)=\text{pts}(b,d)-p_b+\underbrace{\Big(c_{|S|-1}+\sum_{b'\in S}p_{b'}\Big)}_{\text{igual para toda } b\in S},
$$

então $\arg\max_b[\text{pts}(b,d)+\tilde V(S\setminus b)] = \arg\max_b[\text{pts}(b,d)-p_b]$. **Marcar a casa
de maior pontos − preço é exatamente o argmax de Bellman sob a aproximação aditiva.** Três consequências:

1. $p_b=\tilde V(S)-\tilde V(S\setminus b)-(c_{|S|}-c_{|S|-1})$: o preço é o **custo de oportunidade** de
   gastar $b$ (quanto o futuro perde), a menos de uma constante comum a todas as casas.
2. Somar uma constante a todos os preços não muda nenhuma decisão (por isso a busca de C51 fixa o Ás).
3. A mesma constante atravessa a rodada inteira: com $\tilde V$ aditivo, a equação de Bellman da rodada é
   "maximize $\mathbb E[\text{pts}(b)-p_b]$ nos três lançamentos". Resolvê-la com todas as guardas é o teto
   C (C52: 145,98 com os preços "sozinha"); a cola troca essa otimização por regras de guarda (G4 +
   emendas) e usa o argmax só para marcar.

De onde vêm os preços: (a) **"sozinha"** $=E^*(\{b\})$ (C10–C14), que é exatamente o preço marginal na última
rodada (C58); (b) o preço marginal na cartela cheia, $E^*(\text{cheia})-E^*(\text{cheia}\setminus b)$ (C58,
rodada 1); (c) o ajuste por mínimos quadrados (C53). A aproximação não é exata: o resíduo do ajuste é
1,66 ponto RMS, e o preço marginal muda ao longo da partida (C58: a Quadrada cai de 35,4 na rodada 1 para
11,1 na 10ª). O que importa para a decisão é a **ordem relativa** dos preços, não o nível.

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
$\mathbf{1151/1296}\approx 0{,}8881$.

**Correção (v2, pauta › H6).** A v1 concluía daqui que "a diferença entre as duas está quase toda na
marcação, não nas guardas". **Isso estava errado.** C28 mede só a concordância no 1º lançamento da cartela
vazia, e não diz nada sobre o meio e o fim da partida. O lema da diferença de desempenho (C39, C40), que
reparte exatamente os 26,06 pontos de C30 por decisão, mostra o contrário: as **guardas** respondem por
14,47 (forma i, 55,5%) ou 21,14 (forma ii, 81,1%) pontos; a casa errada no 3º lançamento, por 9,56 ou 2,79;
parar cedo, por 2,03 ou 2,14. O número de C28 continua certo.

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

## Resultados v2 — de volta à mesa

> C36–C38 (RL) estão arquivadas (pauta › E5). As afirmações da v2 começam em C39. Todos os valores esperados
> abaixo são **racionais exatos** (avaliação de política em inteiros escalados); os decimais são
> arredondamentos. Ganho recuperado $=(E_\pi-E_G)/(E^*-E_G)$ com $E_G$ = C29 e $E^*$ = C17.

### Colas: definição executável (resumo; o contrato está em `claims.yaml › colas`)

Em cada lançamento $r$ com mão $d$ e casas livres $L$: (1) $\text{pts}(b)$ de cada $b\in L$ (com boca se
$r=1$); (2) casa escolhida $h$ = argmax de $\text{pts}(b)$ (gulosa) ou de $\text{pts}(b)-p_b$ (com tabela),
empate → menor índice; (3) para e marca $h$ se $r=3$, ou $h$ é combinação com pontos > 0, ou cinco iguais
(G3), salvo `quad_gen`/`fu_alto`; (4) senão guarda pela regra G4 (ou `alvo`), substituída por `seg4x`,
`doispar` ou `fu_alto` quando se aplicam. As seis emendas:

| emenda | texto da cola | itens |
|---|---|---|
| `seg4x` | Seguida livre e 4 faces diferentes de uma mesma Seguida no 1º/2º lançamento: guarde um dado de cada (se as duas Seguidas servem, a alta) | 3 |
| `quad_gen` | General livre e a casa escolhida é a Quadrada no 1º/2º lançamento: não marque, guarde a quadra | 2 |
| `alvo` | só guarde face "útil" (casa de número livre, ou qualquer face se Fú/Quadrada/General livre); sem face útil, jogue para a Seguida (se livre) ou relance tudo | 2 |
| `risca_gen` | (sem preços) quando nada pontua, risque o General primeiro | 1 |
| `doispar` | Fú livre, 2º lançamento, dois pares: guarde os dois pares | 3 |
| `fu_alto` | 2º lançamento, Fú com trinca de 4/5/6 e Quadrada ou General livre: desmanche, guarde a trinca | 3 |

Colas protagonistas (pauta › H2), ambas com **R4** = `alvo` + `seg4x` + `quad_gen` + `doispar`:

- **(i) "o preço é o valor da casa jogada sozinha"** (C10–C14 arredondados, H9): Ás 2, Duque 4, Terno 6,
  Quadra 8, Quina 11, Sena 13, Fú 7, Seguida 8, Quadrada 11, General 2. 21 itens.
- **(ii) versão de bolso:** "casa de número custa 2× a face; Fú, Seguida e Quadrada 8; General 2". 14 itens.

Texto de mesa da cola (ii), completo: *"Depois de cada lançamento, calcule para cada casa livre os pontos
que ela daria agora menos o preço dela (casa de número: 2× a face; Fú, Seguida, Quadrada: 8; General: 2).
A melhor é a de maior saldo (empate: a primeira da cartela). Pare e marque a melhor se for o 3º lançamento,
se ela for uma combinação que pontua, ou se saírem cinco iguais — mas, com o General livre, não pare numa
Quadrada antes do 3º lançamento: guarde a quadra. Senão, guarde todos os dados da face útil mais
frequente (útil = casa da face livre, ou qualquer face se Fú, Quadrada ou General estiver livre; empate: a
maior); sem face útil, guarde os dados que servem à Seguida (se livre). Com a Seguida livre e 4 faces de
uma mesma Seguida, guarde essas 4. Com o Fú livre e dois pares no 2º lançamento, guarde os dois pares."*

### C39 {#c39} — Lema da diferença de desempenho, forma (i)

Estado de decisão $s=(S,r,d)$; $d^\pi(s)$ = probabilidade de passar por $s$ sob $\pi$ a partir da cartela
vazia (calculada **exatamente**, em inteiros, pela cadeia induzida do degrau 3). Para MDPs de horizonte
finito,

$$
E^*-E^\pi=\sum_s d^\pi(s)\,\big[V^*(s)-Q^*(s,\pi(s))\big]=\sum_s d^*(s)\,\big[Q^\pi(s,\pi^*(s))-V^\pi(s)\big].
$$

(Prova da 1ª: telescopia $V^*(s_0)-V^\pi(s_0)=\mathbb E_\pi\sum_t[V^*(s_t)-Q^*(s_t,a_t)]$, porque
$Q^*(s_t,a_t)=R_t+\mathbb E[V^*(s_{t+1})]$; a 2ª é a mesma conta com os papéis trocados.) Cada parcela é
"quanto custou esta decisão". Todas as parcelas têm denominador $6^{150}$; a soma bate com C30 **como
igualdade de inteiros**.

Forma (i), visitação da gulosa e jogo ótimo depois (soma 26,063343):

| onde a gulosa erra | pontos |
|---|---|
| guarda no 1º lançamento (as duas guardam, guardas diferentes) | 8,727112 |
| guarda no 2º lançamento | 5,740843 |
| casa diferente no 3º lançamento | 9,560718 |
| para no 2º (marca; a ótima relançaria) | 1,761433 |
| para no 1º | 0,273237 |
| **guardas (1º + 2º)** | **14,467955 (55,5%)** |

Nenhuma outra combinação tem parcela: a gulosa nunca perde marcando casa diferente da ótima no 1º/2º
lançamento nem relançando onde a ótima marcaria.

### C40 {#c40} — forma (ii)

Visitação da ótima e gulosa depois (soma 26,063343, igualdade exata): guarda no 1º 10,918087; guarda no 2º
10,219743; casa no 3º 2,786930; para no 2º 2,117441; para no 1º 0,021142. **Guardas: 21,137830 (81,1%).**
As duas formas concordam no essencial (guardas > metade; a forma (i) dá mais peso à casa no 3º porque mede
o erro com o futuro ótimo, onde marcar mal o fim de jogo custa caro) e corrigem C28 (pauta › H6).

### C41 {#c41}

Forma (i) por nº de casas livres no início da rodada: 10 livres 1,1706; 9: 1,3527; 8: 1,6039; 7: 1,8831;
6: 2,1473; 5: 2,4213; 4: 2,9278; 3: 4,0508; **2: 6,9926**; 1: 1,5134. O fim de jogo concentra a perda. Os
maiores pares "casa no 3º" (gulosa → ótima): **Seguida → General 2,9032** (com Seguida e General livres e
nada pontuando, a gulosa risca a Seguida por G2 e a ótima risca o General), Sena → Ás 1,1711, Quina → Ás
1,0167, Quadra → Ás 0,6607, Quadrada → General 0,5201, Fú → General 0,4185.

### C42 {#c42} — "4 dados de uma Seguida → guarde-os"

Cada emenda sozinha sobre a gulosa (valor exato − C29): `seg4x` **+7,4513** (128,116360); `alvo` +2,8359;
`quad_gen` +2,0948; `doispar` +0,6769; `risca_gen` **−0,6637**; `fu_alto` **−0,8041**. A regra mais
valiosa do Bozó, sozinha, é não ignorar a Seguida: a gulosa só completa a Seguida em 25,9% das partidas
(C33) porque G4 nunca guarda dados distintos.

### C43 {#c43} — seleção forward a partir da gulosa

| passo | entra | itens acumulados | valor exato | recuperado |
|---|---|---|---|---|
| 0 | gulosa | 0 | 120,6650 | 0% |
| 1 | `seg4x` | 3 | 128,1164 | 28,6% |
| 2 | `quad_gen` | 5 | 130,2412 | 36,7% |
| 3 | `alvo` | 7 | 131,6100 | 42,0% |
| 4 | `risca_gen` | 8 | 134,0498 | 51,4% |
| 5 | `doispar` | 11 | **135,1318** | 55,5% |
| 6 | `fu_alto` | 14 | 134,6547 | 53,7% |

`risca_gen` piora a gulosa pura (−0,66) mas vale +2,44 depois de `seg4x`, `quad_gen` e `alvo`. Leitura
(interpretação, não afirmação): a gulosa pura faz General por acaso, perseguindo faces repetidas, e riscá-lo
cedo joga isso fora; com `seg4x` a Seguida passa a ser a reserva que vale preservar no fim. `fu_alto`
(generalizar C24) piora em qualquer ponto: a jogada de C24 só vale em situações específicas.

### C44 {#c44} — interação entre guardas e marcação

O ganho de `alvo` depende do resto da cola: +2,835866 sobre a gulosa; **+10,697723** sobre a cola de preços
(ii) sem emendas; +11,062990 sobre a (i). Com preços, a cola risca/gasta casas de número cedo, e guardar
faces de casas já fechadas passa a ser um erro frequente que `alvo` conserta.

### C45 {#c45} e C46 {#c46} — as duas colas protagonistas

| cola | itens | valor exato | recuperado |
|---|---|---|---|
| (i) preços "sozinha" arredondados + R4 | 21 | **141,173041** | 78,7% |
| (ii) "2× a face / 8 / 2" + R4 | 14 | **141,103896** | 78,4% |

Valores exatos em `claims.yaml`. As duas ficam a 0,07 ponto uma da outra: o nível exato dos preços importa
pouco; a estrutura importa (C55: preço 0 nas combinações derruba a (ii) para 130,28).

### C47 {#c47} — só preços

Marcação por pontos − preço com as guardas da gulosa: (ii) 123,012426 (4 itens); (i) 122,968115 (11 itens);
recuperam 9,0% e 8,8%. **Marcar melhor não salva quem guarda mal** — coerente com C39–C40.

### C48 {#c48} e C49 {#c49} — valor de Shapley (64 coalizões exatas, H11)

| jogador | cola (ii) | cola (i) |
|---|---|---|
| preços das casas de número | 1,2743 | **−0,8005** |
| preços das combinações | 5,1980 | 7,3820 |
| `alvo` | 5,2608 | 5,6062 |
| `seg4x` | **5,6729** | 5,3714 |
| `quad_gen` | 2,0457 | 2,0152 |
| `doispar` | 0,9870 | 0,9337 |
| soma (= cola − gulosa) | 20,4389 | 20,5080 |

Os preços das combinações (dar preço à Seguida, à Quadrada e ao Fú, e pouco ao General) valem tanto quanto
`alvo` ou `seg4x`; os preços das casas de número quase nada, e na (i) atrapalham em média (sem preço nas
combinações, preço só nas casas de número faz a cola riscar combinações: "2face_comb0" vale 116,83, abaixo
da gulosa, C55).

### C50 {#c50} — forward a partir da cola de preços (ii)

2face_8_2 (123,0124) → +`alvo` **133,7101** → +`seg4x` 138,0744 → +`quad_gen` 139,9576 → +`doispar`
141,1039 → +`fu_alto` 140,6065. Com preços, `alvo` vem primeiro (C44).

### C51 {#c51} — quão longe está a cola do melhor da família

Busca local em preços inteiros (Ás fixo em 2; vizinhança ±1 em um preço), com R4, partindo de (ii):
termina em **2, 4, 6, 8, 10, 13, 5, 7, 8, 3** com 141,292401 (126 avaliações exatas; nenhum dos 18
vizinhos melhora). Só 0,19 acima da (ii): a cola de bolso já está no platô da família.

### C52 {#c52} — teto C: rodada perfeita com preços (não executável)

| valor futuro aproximado | valor exato | recuperado |
|---|---|---|
| nenhum (preços 0: maximiza só a rodada) | 138,5362 | 68,6% |
| preços 2×face / 8 / 2 | 145,5578 | 95,5% |
| preços marginais na cartela cheia | 145,8249 | 96,5% |
| preços "sozinha" arredondados | 145,9388 | 97,0% |
| **preços "sozinha" exatos (C10–C14)** | **145,9827** | 97,1% |
| ajuste aditivo de C53 | 146,3452 | 98,5% |

Decomposição do caminho gulosa → ótimo: jogar bem **dentro** da rodada vale ~17,9 (120,67 → 138,54); dar
preço ao futuro, mais ~7,4 (→ 145,98); o resto (0,75) é a não aditividade do futuro.

### C53 {#c53} — ajuste aditivo

Mínimos quadrados de $E^*(S)\approx c_{|S|}+\sum_{b\in S}p_b$ nas 1023 cartelas (Ás = 0): 0; 1,81; 3,88;
6,15; 8,56; 11,02; 8,53; 9,74; **19,98**; 2,35; resíduo RMS 1,6629. A Quadrada vale no contexto quase o dobro
de sozinha (19,98 contra 11,13 + const.): ela é a rede de proteção de quem tenta o General.

### C54 {#c54} — iteração de política

$\pi_0$ = gulosa 120,665037 → $\pi_1$ **142,436816** → 146,086695 → 146,700005 → 146,728291 → $\pi_5$ =
146,728381 = $E^*$ (igualdade exata em todas as 1023 cartelas). Uma única aplicação da equação de Bellman
sobre a gulosa recupera 21,8 dos 26,1 pontos.

### C55 {#c55} — curva complexidade × pontos

38 colas avaliadas exatamente (`resultados/v2/curva_colas.json`, com o texto de cada uma). Fronteira de
Pareto em itens de cola: 0 → 120,67 (gulosa) · 2 → 123,50 (+alvo) · 3 → 128,12 (+seg4x) · 5 → 130,24 ·
6 → 133,71 (preços (ii) + alvo) · 8 → 134,05 · 9 → 138,07 · 11 → 139,96 · **14 → 141,10 (cola (ii))** ·
16 → 141,16 · 21 → 141,29 (busca local + R4). Pela medida grosseira (nº de regras) a fronteira é
0 → 120,67 · 1 → 128,12 · 2 → 133,71 · 3 → 138,07 · 4 → 139,96 · 5 → 141,29, com as mesmas colas da
família (ii) no caminho: a conclusão não depende da convenção de contagem. Tetos não executáveis: rodada
perfeita 145,98 (C52), ótimo 146,73 (tabela de 773 388 decisões). Rendimentos decrescentes fortes: os
primeiros 3 itens recuperam 28,6%; os 14 itens da cola (ii), 78,4%; mais 7 itens (tabela inteira) só +0,19.

### C56 {#c56} — distribuição final por General feito / não feito

| política | General feito: P | média | moda | dp | não feito: P | média | moda | dp |
|---|---|---|---|---|---|---|---|---|
| ótima | 0,274993 | 182,48 | 197 | 23,52 | 0,725007 | 133,17 | 146 | 22,15 |
| gulosa | 0,298316 | 156,86 | 150 | 21,62 | 0,701684 | 105,28 | 100 | 22,42 |

Cada distribuição de C20/C32 é a mistura de duas "montanhas" a ~50 pontos uma da outra (49,31 na ótima, 51,58 na
gulosa): é o General (50 ou 55). A moda da ótima (147, C20) e a da gulosa (100, C32) são as modas da
componente "não feito"; o ombro à direita (~197 e ~150) é a componente "feito", com ~28–30% da massa.

### C57 {#c57} — ordem de preenchimento

Rodada média em que cada casa é preenchida (marcada ou riscada):

| casa | Ás | Duque | Terno | Quadra | Quina | Sena | Fú | Seguida | Quadrada | General |
|---|---|---|---|---|---|---|---|---|---|---|
| ótima | 5,11 | 5,19 | 5,42 | 5,28 | 5,55 | 5,59 | 4,44 | 6,49 | 4,41 | 7,51 |
| gulosa | 6,59 | 5,88 | 5,24 | 4,68 | 4,16 | 3,58 | 4,15 | 8,06 | 3,88 | 8,78 |
| cola (ii) | 4,40 | 5,61 | 6,02 | 5,99 | 5,98 | 5,83 | 4,46 | 5,18 | 4,99 | 6,54 |

A gulosa gasta as casas altas cedo (Sena 3,58) e deixa o General para a 10ª rodada em **73,6%** das
partidas (ótima 23,5% = C22; cola (ii) 5,6%). A cola (ii) risca o General mais cedo que a ótima (preço 2) e
fecha a Seguida mais cedo (P(Seguida na 1ª rodada): cola 0,1546; ótima 0,0652; gulosa 0,0321). Matrizes
completas (10 × 10 por política) em `resultados/v2/ordem_preenchimento.json`.

### C58 {#c58} — preço marginal ao longo da partida

$\text{preço}_b(t)=\mathbb E[E^*(S_t)-E^*(S_t\setminus b)\mid b\in S_t]$ sob a ótima. Rodada 1 (cartela
cheia): Ás 9,46; Duque 11,36; Terno 13,55; Quadra 15,95; Quina 18,49; Sena 21,05; Fú 20,04; Seguida 22,43;
**Quadrada 35,42**; General 15,15. Rodada 5: 8,35; 10,15; 12,34; 14,67; 17,20; 19,75; 17,49; 19,09; 30,02;
10,90. Rodada 10: exatamente C10–C14 (2,11 … 2,31). Todos os preços caem ao longo da partida (menos
futuro a proteger); a Quadrada é sempre a mais cara e o General cai mais que proporcionalmente (15,15 →
2,31). Os níveis são bem maiores que os da cola, mas o que decide é a ordem relativa
(degrau 6).

### C59 {#c59} — formulação

12 292 056 pares (estado, ação); a avaliação de Bellman reproduz C29 (gulosa) e $E^*$ (ótima) exatamente;
todas as 1023 cartelas são alcançáveis sob a ótima, a gulosa e a cola (ii). Ver seção 2b.

### Produtos para visualização (`resultados/v2/`, esquemas no README da pasta)

`pergunte_ao_otimo.json` (21 cartelas × 3 lançamentos × 252 mãos: ação ótima, valor, ação e perda da gulosa e
da cola (ii)), `curva_colas.json`, `iteracao_politica.json`, `diagnostico_lema.json`,
`distribuicao_por_general.json`, `ordem_preenchimento.json`, `preco_marginal.json` (inclui $E^*(S)$ para as
1024 cartelas e todos os $E^*(S)-E^*(S\setminus b)$), `replays.json` (3 partidas com dados comuns). Os
replays são ilustrativos (uma partida não é afirmação); semente 20261002: ótima 139, gulosa 110, cola 149;
20261003: 112, 133, 145; 20261004: 188, 135, 187.

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
- **v2 — colas.** A lista de emendas é escolha minha, guiada pelo lema (C39–C41) e pelas jogadas
  contraintuitivas da v1; outra lista daria outra curva. A busca local (C51) e a rodada perfeita (C52)
  limitam o quanto a família "preços + R4" ainda pode ganhar (≈ 0,2 com preços melhores; ≈ 4,9 só
  jogando a rodada de forma perfeita), mas não provam que não exista cola curta melhor de outra família
  (as árvores destiladas, opção D, ficaram fora — pauta › H1).
- **v2 — itens de cola** são uma convenção (H10). A fronteira por nº de regras (C55) passa pelas mesmas
  colas, então a conclusão "14 itens já dão ~78%" não depende dela.
- **v2 — lema:** a repartição por categoria da forma (ii) depende do desempate H6 da ótima (H12); a soma
  não. As duas formas são exatas e respondem perguntas diferentes (custo da decisão com futuro ótimo × com
  futuro guloso).
- **v2 — preço marginal** (C58) é uma média sobre as cartelas alcançadas na rodada; em uma cartela
  específica o preço pode ser bem diferente (o `preco_marginal.json` traz todos os 5120 valores).

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
- ~~A gulosa erra marcando, não guardando~~ **(retirada na v2, pauta › H6).** O certo é o contrário: a
  gulosa perde mais guardando (C39–C40: 14,5 ou 21,1 dos 26,1 pontos), sobretudo por ignorar a Seguida;
  a concordância de 89% no 1º lançamento da cartela vazia (C28) só diz que o começo do jogo é fácil.
- **Gráficos:** (1) histogramas sobrepostos ótima × gulosa (`distribuicoes.npz`); (2) barras pareadas
  "chance de completar cada casa" (C21 × C33); (3) mapa de decisão do 1º lançamento como grade
  (`mapa_decisao_cartela_vazia.csv`, cores por tipo de guarda); (4) curva de aprendizado do RL com
  as retas do ótimo e da gulosa (`rl_curva_aprendizado.csv`); (5) o grafo 5×5 da cadeia do General.
- **Boca vale pouco:** 1,63 ponto (C19). O ótimo até recusa uma das bocas (Quadrada).
- **RL como epílogo humilde:** o agente que só joga precisa de ~100 mil partidas para alcançar a
  gulosa e de 3 milhões para chegar a 85% do ótimo num Bozó de 3 casas (C38). A conta exata resolve
  o jogo inteiro em segundos. Isso mostra por que a estrutura de Markov (decompor por rodada) vale ouro.
  *(RL arquivado na v2, pauta › E5.)*

### Ideias v2 ("da mesa ao ótimo, e de volta à mesa")

- **O arco fecha com o começo.** O valor de cada casa jogada sozinha (C10–C14), que abre o post como
  curiosidade, vira no fim a tabela de preços da cola (i) (C45) — e a rodada 10 do preço marginal (C58) é
  exatamente esse número. O leitor reencontra a tabela.
- **Por que pontos − preço:** um callout com o degrau 6 (seção 2b) — três linhas de álgebra mostram que a
  regra "maior saldo" é a equação de Bellman com o futuro aproximado por uma soma. "Preço = quanto o
  futuro perde se você gastar esta casa agora."
- **A frase-síntese:** "uma cola de guardanapo com uma fórmula de preço e quatro regras faz 141,1 pontos:
  78% do caminho da gulosa ao ótimo" (C46). Com a tabela inteira de preços, quase nada a mais (C45, C51).
- **Marcar melhor não salva quem guarda mal** (C47: preços sozinhos +2,3; C44: a mesma emenda `alvo`
  vale +2,8 sem preços e +10,7 com preços). Bom momento para o diagnóstico do lema: "onde se perdem os 26
  pontos" (C39–C41) antes das colas — ele aponta as guardas e o fim de jogo.
- **A regra de ouro:** "não ignore a Seguida" sozinha vale 7,45 (C42). Contraste com o instinto: a gulosa
  completa a Seguida em 26% das partidas; a ótima em 76% (C33 × C21).
- **Generalizar uma jogada brilhante pode ser erro:** `fu_alto` (C24 generalizada) tira pontos em
  qualquer cola (C42, C43, C50).
- **Iteração de política como "aprender pensando":** cinco rodadas de "avaliar → melhorar com Bellman"
  levam a gulosa ao ótimo exato; a primeira já faz 142,4 (C54). Contraste didático com o RL, que joga
  milhões de partidas.
- **Teto honesto:** mesmo um jogador perfeito dentro da rodada, com os preços "sozinha", fica 0,75 abaixo
  do ótimo (C52): o futuro não é exatamente aditivo (C53).
- **Gráficos interativos:** (1) "pergunte ao ótimo" — escolha cartela e dados, veja a ação ótima, a da
  gulosa e a da cola, com a perda de cada uma (`pergunte_ao_otimo.json`); (2) curva complexidade × pontos
  com tooltip do texto da cola, fronteira de Pareto destacada e linhas horizontais do teto C e do ótimo
  (`curva_colas.json`); (3) waterfall/treemap do lema (`diagnostico_lema.json`); (4) escada da iteração de
  política (`iteracao_politica.json`); (5) heatmap "ordem de preenchimento" lado a lado para as três
  políticas — o General vermelho na 10ª rodada da gulosa (`ordem_preenchimento.json`); (6) linhas do
  preço marginal por rodada, com a Quadrada no topo (`preco_marginal.json`); (7) histogramas empilhados
  "General feito / não feito" explicando o ombro (`distribuicao_por_general.json`); (8) replay passo a
  passo das três políticas com os mesmos dados (`replays.json`).
- **Shapley como "quem merece o crédito":** o preço das combinações (5,2) vale tanto quanto `alvo` ou
  `seg4x`; o preço das casas de número, quase nada (C48–C49). A lição da cola é "dê preço às combinações".
