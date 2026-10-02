# Abordagens v2: de volta à mesa (heurísticas executáveis)

> Matemático, modo exploração, 2026-10-02. Entrada: `pauta.md › Versão 2` (F1–F4). Mantêm-se D1–D6,
> E1–E6, H1–H7 e as 35 afirmações validadas.
>
> **Números preliminares.** Todos saem de protótipos descartáveis em `_work/scratch/`
> (`heur.py`, `familias.py`, `mesa.py`, `extra.py`, `precos_var.py`, `pdl.py`, `iteracao.py`). Eles fazem a
> **avaliação exata de política** (Bellman de avaliação, de trás para frente nas cartelas) em float64.
> O avaliador reproduz a ótima (146,728381) e a gulosa (120,665037) de C17/C29 a 1e-6. Cada protótipo
> roda em ≤ 2,5 min; construir e avaliar uma política leva ~0,15 s. Na execução, cada valor sai exato
> (inteiros escalados, `derivacao.rodada_exata(pol=...)`).
>
> Notação: $E^*$ = ótimo (146,73); $E_G$ = gulosa (120,67); **ganho recuperado** = $(E_\pi - E_G)/(E^*-E_G)$.

---

## 0. Diagnóstico antes das heurísticas: onde a gulosa perde os 26 pontos

Isto não é uma heurística. É a ferramenta que mostra **quais regras vale a pena escrever**, e serve
também como afirmação do post.

**Lema da diferença de desempenho** (exato, para MDPs de horizonte finito):

$$
E^* - E^\pi \;=\; \sum_{s} d^\pi(s)\,\big[V^*(s) - Q^*(s,\pi(s))\big]
\;=\; \sum_{s} d^*(s)\,\big[Q^\pi(s,\pi^*(s)) - V^\pi(s)\big],
$$

em que $s=(S,r,d)$ é um estado de decisão e $d^\pi(s)$ é a probabilidade de passar por ele jogando
$\pi$. Cada parcela é "quanto custou esta decisão". A primeira forma usa a visitação da gulosa e
supõe jogo ótimo depois. A segunda usa a visitação da ótima e supõe que a gulosa continua depois.
Nas duas, a soma fecha em 26,063343, como deve.

| Onde a gulosa erra | Forma (i): visitação da gulosa, $Q^*$ | Forma (ii): visitação da ótima, $Q^G$ |
|---|---|---|
| **guarda** errada no 1º lançamento | 8,73 | 10,92 |
| **guarda** errada no 2º lançamento | 5,74 | 10,22 |
| **casa** errada no 3º lançamento | 9,56 | 2,79 |
| para cedo no 2º lançamento (a ótima relançaria) | 1,76 | 2,12 |
| para cedo no 1º lançamento | 0,27 | 0,02 |

Na forma (i), os maiores pares (casa marcada pela gulosa → pela ótima) são: Seguida → General
(2,90: a gulosa risca a Seguida, a ótima risca o General), Sena → Ás (1,17), Quina → Ás (1,02),
Quadra → Ás (0,66) e Quadrada → General (0,52). Por número de casas livres, o fim de jogo pesa
mais: com 2 casas livres a gulosa perde 6,99 pontos, com 3 perde 4,05, com 10 perde só 1,17.

**Atenção, conflito com a v1.** Em `modelo.md › C28` e em "Ideias para o post" eu escrevi que "a
diferença está quase toda na marcação, não nas guardas". As duas formas do lema **desmentem** isso: as
guardas respondem por 14,5 (forma i) ou 21,1 (forma ii) dos 26 pontos. O número de C28 (1151/1296 de
concordância no 1º lançamento da cartela vazia) continua certo. A conclusão que tirei dele está errada,
porque concordar na cartela vazia não diz nada sobre o meio e o fim do jogo. **Proponho reescrever a
frase** (Modo 3 ou nesta execução). Pelo `grep`, o `index.qmd` atual não repete a frase.

---

## A. Gulosa + emendas (regras de bolso) — N4 (avaliação de política)

**Pergunta que responde:** quais regras de bolso, somadas à gulosa uma a uma, recuperam mais pontos, e
quanto vale cada uma?
**Método:** partir da gulosa G1–G4 (E6) e acrescentar emendas tiradas do diagnóstico 0 e das jogadas
contraintuitivas da v1 (C23–C26). Cada combinação de emendas é uma política fixa, ou seja, uma cadeia de
Markov sobre as cartelas, avaliada **exatamente**. Duas medidas de importância: (1) **caminho forward**
(a cada passo entra a emenda de maior ganho exato); (2) **valor de Shapley** de cada emenda, a média do
ganho marginal sobre todas as ordens. As emendas interagem muito (ver abaixo), então o ganho marginal
depende da ordem. Com 5–6 emendas, os $2^6 = 64$ subconjuntos custam ~10 s.
**Ferramentas:** existentes (`derivacao.rodada_exata(pol=…)`, `politica_gulosa`). A criar:
`politica_mesa(precos, regras)`, já prototipada e vetorizada.
**Emendas prototipadas** (texto de cola):

| id | regra | itens de cola (§ Complexidade) |
|---|---|---|
| `seg4x` | Seguida livre e 4 dados distintos de uma mesma Seguida (1234, 2345, 3456 ou com buraco: 1235, 2456…) e no máximo um par: guarde os 4 | 3 |
| `quad_gen` | General livre e Quadrada no 1º/2º lançamento: **não marque**, guarde a quadra (C23) | 2 |
| `alvo` | só guarde face repetida que ainda serve (a casa dela livre, ou Fú/Quadrada/General livre). Se nenhuma serve e a Seguida está livre, jogue para a Seguida | 2 |
| `risca_gen` | se nada pontua, risque o **General** primeiro (depois Ás, Duque…) | 1 |
| `doispar` | Fú livre e dois pares no 2º lançamento: guarde os dois pares | 3 |
| `fu_alto` | desmanchar Fú sem boca com trinca alta se Quadrada/General livre (C24) | 3 |

**Preliminar, forward a partir da gulosa** (marcação G2, sem preços):

| cola | itens | valor | ganho recuperado |
|---|---|---|---|
| gulosa | 0 | 120,665 | 0% |
| + `seg4x` | 3 | 128,116 | 28,6% |
| + `quad_gen` | 5 | 130,241 | 36,7% |
| + `alvo` | 7 | 131,610 | 42,0% |
| + `risca_gen` | 8 | 134,050 | 51,4% |
| + `doispar` | 11 | 135,132 | 55,5% |

Sozinhas sobre a gulosa: `seg4x` +7,45; `quad_gen` +2,10; `alvo` +1,87; `doispar` +0,68. Uma emenda
**piora**: `fu_alto` tira 0,48 ponto quando entra no fim. A jogada de C24 só vale na cartela vazia.
Generalizada, ela é um erro, e isso também vale contar.
**O que o leitor ganha:** "a regra mais valiosa do Bozó é *não ignore a Seguida*: sozinha, ela vale
7,5 pontos". Gráfico: escada (waterfall) do ganho de cada emenda, com a reta do ótimo.
**Custo:** segundos. Explicar: baixa.
**Riscos e limites:** a lista de emendas candidatas é escolha minha, e outra lista daria outra curva. O
diagnóstico 0 justifica a escolha, mas não prova que ela é a melhor. Cada regra precisa de texto **sem
ambiguidade**, com desempates, ou o validador modela outra coisa (lição do Modo 3).
**Como validar:** o validador implementa cada cola a partir do **texto** do enunciado, roda MC literal
($n=10^6$, `montecarlo.SEED`, IC99% de ±0,08) e, se quiser, a própria recursão de avaliação.

## B. Cola de preços: "marque a casa que maximiza pontos − preço" — N4 + aproximação de Bellman

**Pergunta que responde:** dez números (um "preço" por casa) bastam para marcar como o ótimo?
**Método, com a ligação à equação de Bellman.** Na última decisão da rodada, o ótimo escolhe
$\arg\max_{b\in S}\,[\text{pts}(b,d)+E^*(S\setminus b)]$. Se o valor do futuro for aproximadamente
**aditivo**, $E^*(S)\approx c_{|S|}+\sum_{b\in S}p_b$, então

$$
\text{pts}(b,d)+E^*(S\setminus b)\;\approx\;\text{pts}(b,d)-p_b+\underbrace{c_{|S|-1}+\textstyle\sum_{b'\in S}p_{b'}}_{\text{igual para toda casa}} .
$$

A regra "marque a casa de maior **pontos − preço**" é exatamente o argmax de Bellman sob essa
aproximação. O preço $p_b$ é um **custo de oportunidade**: quanto o futuro perde se a casa $b$ for
gasta agora. As guardas continuam pelas regras de A.
**Fontes de preço (preliminar):**

| preços | Ás | Duque | Terno | Quadra | Quina | Sena | Fú | Seguida | Quadrada | General |
|---|---|---|---|---|---|---|---|---|---|---|
| **"sozinha"** (= C10–C14: valor da casa jogada sozinha) | 2,11 | 4,21 | 6,32 | 8,43 | 10,53 | 12,64 | 7,45 | 7,99 | 11,13 | 2,31 |
| arredondados (cola) | 2 | 4 | 6 | 8 | 10 | 13 | 7 | 8 | 11 | 2 |
| **"2× a face; combinação 8; General 2"** | 2 | 4 | 6 | 8 | 10 | 12 | 8 | 8 | 8 | 2 |
| ajuste aditivo por MQ de $E^*(S)$ nas 1023 cartelas (Ás = 0) | 0 | 1,81 | 3,88 | 6,15 | 8,56 | 11,02 | 8,53 | 9,74 | **19,98** | 2,35 |
| marginais na cartela cheia $E^*(\text{cheia})-E^*(\text{cheia}\setminus b)$ | 9,46 | 11,36 | 13,55 | 15,95 | 18,49 | 21,05 | 20,04 | 22,43 | 35,42 | 15,15 |

Os preços "sozinha" ligam o post de ponta a ponta: o valor de cada casa jogada sozinha (parte A da v1)
**vira a cola** do fim. A regra $455k/216 \approx 2{,}1k$ vira "casa de número custa 2× a face". O
ajuste MQ tem resíduo RMS de 1,66 ponto. A Quadrada vale bem mais no contexto (20) do que sozinha (11),
porque é a rede de proteção de quem tenta o General.

**Preliminar** (guardas de A; `R4` = `alvo` + `seg4x` + `quad_gen` + `doispar`):

| cola | itens | valor | ganho recuperado |
|---|---|---|---|
| só preços "sozinha" arredondados, guardas da gulosa | 11 | 123,062 | 9,2% |
| só "2×face / 8 / 2", guardas da gulosa | 4 | 123,012 | 9,0% |
| "2×face / 7,8,11,2" + `alvo` | 8 | 133,816 | 50,5% |
| … + `seg4x` | 11 | 138,188 | 67,2% |
| … + `quad_gen` | 13 | 140,071 | 74,5% |
| … + `doispar` | 16 | 141,165 | 78,7% |
| **"2×face / 8 / 2" + R4** | **14** | **141,104** | **78,4%** |
| arredondados "sozinha" + R4 | 21 | 141,205 | 78,8% |
| busca coordenada nos preços (passo 0,5) + R4 + `fu_alto` | 24 | 140,864 | 77,5% |
| "2×face / combinações 0" + R4 | 13 | 130,283 | 36,9% |
| "só números, preço = face" + R4 | 12 | 136,367 | 60,2% |

Achados: (1) os preços **sozinhos** quase não ajudam (+2,4). Com as guardas da gulosa, marcar melhor
não salva quem guarda mal. (2) Com preços, a emenda `alvo` vale **+10,9** (sem preços, +1,9). A interação é forte, e por isso
proponho Shapley. (3) Os números exatos importam pouco ("2×face / 8 / 2" fica a 0,1 de "sozinha"
arredondado). A estrutura importa muito: preço zero nas combinações derruba 11 pontos. (4) A busca
coordenada nos preços chegou a 140,86 com preços 2,3,5,8,10,13,4,7,11,2, abaixo de "sozinha" arredondado
(141,21). Mas o protótipo incluiu `fu_alto`, que atrapalha, então a comparação não está limpa. Na
execução refaço a busca só com R4. Indício provisório: ótimo local raso, e a cola simples já fica perto
do melhor que esta família faz.
**O que o leitor ganha:** **"uma cola de guardanapo com 4 linhas de preço e 4 regras faz 141 pontos:
78% do caminho da gulosa ao ótimo"**. Gráfico: tabela de preços como cartela anotada.
**Custo:** segundos. Explicar: média, porque a ligação com Bellman pede um callout.
**Riscos e limites:** a decomposição aditiva é uma aproximação. Falta a interação entre casas (ex.:
Quadrada × General), e o resíduo de 1,66 RMS mostra isso. Desempate entre casas com o mesmo pontos − preço
precisa ser fixado (menor índice).
**Como validar:** como em A. Os preços entram no enunciado como números, e o validador implementa a regra
do texto.

## C. Lookahead de uma rodada com valor aproximado, e iteração de política — N4 (teto e ponte F2)

**Pergunta que responde:** se você jogasse **cada rodada** com perfeição e só usasse uma cola para
valorizar o futuro, quanto perderia? E quantos passos de "melhoria de Bellman" separam a gulosa do ótimo?
**Método:** $\pi_{\hat V}$ = resolver a rodada atual exatamente (3 lançamentos, todas as guardas) com
recompensa terminal $\text{pts}(b,d)+\hat V(S\setminus b)$. Com $\hat V=E^*$, isso é o ótimo. Com $\hat V$
aditivo, é "jogador perfeito dentro da rodada + 10 preços". **Iteração de política no nível da rodada:**
$\pi_{k+1}=\pi_{\hat V}$ com $\hat V=E^{\pi_k}$, partindo da gulosa.
**Preliminar:**

| $\hat V$ | valor | ganho |
|---|---|---|
| $\hat V = 0$ (maximiza só a rodada atual) | 138,539 | 68,6% |
| preços marginais da cartela cheia | 145,825 | 96,5% |
| **preços "sozinha" (C10–C14)** | **145,983** | **97,1%** |
| preços inteiros 2,4,6,8,10,13,7,8,11,2 | 146,029 | 97,3% |
| aditivo ajustado por MQ | 146,344 | 98,5% |
| idem, guardas restritas ao vocabulário (§ D) | 145,574 | 95,6% |

Iteração de política a partir da gulosa: **120,665 → 142,437 → 146,087 → 146,700 → 146,72829 →
146,728381** (= $E^*$ na 5ª melhoria, com $\max_S|E^{\pi_5}-E^*|<10^{-9}$ em float. Na execução confiro
a igualdade exata). No horizonte de 10 rodadas a convergência é garantida em ≤ 10 passos, mas ela vem
em 5.
**O que o leitor ganha:** a decomposição limpa do gap: **jogar bem dentro da rodada** vale ~18 pontos
(gulosa → 138,5), **dar preço ao futuro** vale mais ~7,5 (→ 146,0), e o resto é 0,75. Também a ponte
F2: "uma única aplicação da equação de Bellman sobre a gulosa leva de 120,7 a 142,4".
**Custo:** ~0,5 s por política. Explicar: média.
**Riscos e limites:** **não é executável à mesa.** Resolver a rodada exatamente exige a recursão
intra-rodada. Entra no post como **teto** da família B e como ponte didática, não como cola.
**Como validar:** o validador implementa a rodada com recompensa terminal pts − preço a partir do texto,
faz MC literal e confere o IC. A iteração de política pode ser conferida com o próprio `mdp`.

## D. Árvore de decisão destilada da ótima, de tamanho crescente — N4/aprendizado supervisionado

**Pergunta que responde:** qual a melhor cola com $k$ perguntas sim/não, para $k = 1, 2, 4, …, 64$?
**Método:** rótulo = ação da ótima expressa num **vocabulário de guardas** que um humano entende ("todos os
dados da face $f$", "os dados da Seguida", "os dois pares", "nada") ou "marque a casa $b$". Atributos
legíveis: lançamento, casas livres (10 bits), perfil da mão, face mais frequente, comprimento da sequência,
pontos de cada casa, pontos − preço. Treino ponderado pela visitação e, depois, **DAgger** (re-pesar pela
visitação da própria árvore, que é o que conserta o erro composto). Cada árvore é avaliada **exatamente**.
**Já medido (teto da família):** com o vocabulário (5,2 guardas por mão em média, contra 16,3 de todas), o
ótimo restrito vale **146,268**, só 0,46 abaixo de $E^*$. Então quase toda a perda das colas está em
*escolher* entre guardas óbvias, e não em guardas exóticas.
**Ferramentas:** **a criar**. Precisa de `scikit-learn` (`DecisionTreeClassifier` com `max_leaf_nodes`),
que **não está no `pyproject.toml`**: peço ao supervisor. A alternativa é um CART próprio de ~100 linhas.
**O que o leitor ganha:** a curva "perguntas × pontos" mais honesta (a máquina escolhe as perguntas) e
uma árvore pequena desenhável (profundidade 3–4) para comparar com a cola humana de A+B.
**Custo:** treino em segundos. São 773 mil amostras (menos as de peso zero) e a avaliação é exata (0,2 s).
DAgger multiplica por ~5. Explicar: média.
**Riscos e limites:** árvores curtas imitam mal decisões que dependem de **conjuntos** de casas livres. A
árvore pode precisar de atributos "engenheirados" (pontos − preço), e aí ela herda a família B. Também
pode não superar A+B com o mesmo tamanho, o que seria um resultado ("o humano com bom senso escolhe
perguntas melhores que a árvore").
**Como validar:** a árvore é publicada como texto (if/else). O validador implementa e faz MC.

## E. Mineração automática de regras guiada pelo lema (§ 0) — N4, fora do catálogo

**Pergunta que responde:** a versão "sem viés do matemático" de A. Que regras um algoritmo escolheria?
**Método:** gerar um banco de milhares de regras candidatas da forma
"se (lançamento ∈ R) e (casa X livre/ocupada) e (perfil da mão = P) → ação do vocabulário". Pontuar cada
uma pela estimativa de 1ª ordem do lema,
$\sum_s d^\pi(s)\,[Q^*(s,a_{\text{nova}}(s))-Q^*(s,\pi(s))]$, que é barata. Confirmar os 10 melhores com
avaliação exata, incorporar o melhor e repetir.
**O que o leitor ganha:** a curva de A sem escolha manual de regras e um teste se a lista humana é
"boa".
**Custo:** minutos. Explicar: alta (vale um callout no máximo).
**Riscos:** regras mineradas podem ser ilegíveis ("se Duque e Quina livres e perfil 2-1-1-1…"). Também há
risco de sobreajuste à forma da ótima.
**Como validar:** igual a A.

---

## Medida de "complexidade da cola"

**Proposta: $\kappa$ = número de itens de cola**, com as convenções:

1. cada **teste** que o jogador faz ("a Seguida está livre?", "tenho 4 de uma Seguida?", "é o 1º ou 2º
   lançamento?") = 1 item;
2. cada **número a decorar** = 1 item. Uma **fórmula** que gera vários números ("casa de número: 2× a
   face") = 1 item;
3. uma regra de argmax ("marque a de maior pontos − preço") = 1 item;
4. a **ação** (conclusão de uma regra, folha de uma árvore) = 0 item;
5. a gulosa G1–G4 é o "instinto" de base: $\kappa = 0$.

Assim uma árvore com $k$ nós internos tem $\kappa = k$, uma tabela de 10 preços tem $\kappa = 11$, "2×face /
8 / 2" tem $\kappa = 4$, e a política ótima tem 773 388 entradas (ou ~$1{,}5\times10^6$ bytes).

Por quê: (i) a medida é **comum às famílias**, porque regra, árvore e preço viram "perguntas + números";
(ii) mede o que pesa à mesa: o que se decora e o que se confere a cada jogada; (iii) é contável a partir do
texto da cola, sem ver o código, então o validador confere $\kappa$ junto com o valor.
**Robustez:** proponho mostrar a fronteira de Pareto também com uma medida mais grosseira (número de
regras/linhas). Se a ordem das colas não mudar, a conclusão não depende da convenção. Complemento
opcional: a **concordância com a ótima ponderada pela visitação** (gulosa 61,4%; cola "preços + R4" 79,8%;
lookahead C 92,8%; ~28,2 decisões por partida).

**Curva preliminar (fronteira):** 0 → 120,7 · 3 → 128,1 · 8 → 133,8 · 11 → 138,2 · 13 → 140,1 ·
14 → 141,1 · (teto C: 146,0; teto vocabulário: 146,3; ótimo: 146,7 com 773 388 entradas).

---

## Formulação para o texto (F2), passo a passo, e o que vira afirmação

1. **Cadeia de Markov (sem decisão).** Estado = nº máximo de dados iguais. A regra "guarde a face mais
   frequente" está fixa, e a matriz $5\times5$ de C5 dá $P(\text{General em 3}) = (\pi_0P^2)_5$.
   *Afirmações: C5, C9 (já validadas).*
2. **Alguém escolhe: MDP $(\mathcal S,\mathcal A,P,R)$.** $\mathcal S=\{(S,r,d)\}$ (cartela, lançamento,
   mão), $|\mathcal S| = 773\,388$; $\mathcal A$ = marcar $b\in S$ ou guardar $k\subsetneq d$;
   $P$ = $T(k,d')$ (multinomial do relançamento) ou ir para $(S\setminus b, 1, \cdot)$;
   $R = \text{pts}(b,d,[r{=}1])$. A cartela basta como memória (sem bônus), e por isso é Markov.
   *Afirmação: C16 (validada).* Nova opcional: nº de pares (estado, ação).
3. **Fixar uma política gera uma cadeia de Markov.** Com $\pi$ fixa, $P^\pi(s'|s)=P(s'|s,\pi(s))$. As
   cartelas evoluem por uma cadeia sobre os $2^{10}$ subconjuntos (absorvida em $\varnothing$). É assim que
   se mede a gulosa e qualquer cola. *Afirmações: C29 = recursão de avaliação aplicada à gulosa (validada).
   Nova: P(casa $b$ ainda livre depois da rodada $t$) sob gulosa e ótima, uma matriz 10×10 que alimenta o
   gráfico de "ordem de preenchimento".*
4. **Bellman de avaliação × de otimalidade.**
   $V^\pi(s) = R(s,\pi(s)) + \sum_{s'}P(s'|s,\pi(s))V^\pi(s')$ (sistema linear e triangular, porque as
   cartelas só encolhem) ×
   $V^*(s)=\max_a[R(s,a)+\sum_{s'}P(s'|s,a)V^*(s')]$. *Afirmação nova: **lema da diferença de desempenho**,
   os 26,06 pontos repartidos por tipo de decisão (§ 0), exato e verificável.*
5. **Programação dinâmica = resolver a de otimalidade de trás para frente** ($|S| = 1, 2, …, 10$).
   *C17 (validada).* Ponte nova: **iteração de política** (avaliar → melhorar com Bellman → avaliar…),
   gulosa → 142,4 → 146,1 → 146,70 → 146,728 → $E^*$ em 5 passos. *Afirmação nova, exata.*
6. **De volta à mesa:** o argmax de Bellman com $\hat V$ aditivo é a regra pontos − preço (§ B).
   *Afirmação nova: valor exato de cada cola.*

Ordem sugerida no texto: 1 → 2 → 3 (ainda com a gulosa) → 4 → 5 → (resultados da v1) → 6.

---

## Dados para visualizações (F4)

Tudo pré-computado em JSON (ou binário) por `derivacao.py`. Nenhuma DP roda no navegador.

| produto | conteúdo | tamanho | uso |
|---|---|---|---|
| **Pergunte ao ótimo: cartela vazia** | 252 mãos × 3 lançamentos: ação ótima + valor exato das 3 melhores ações + ação/valor da gulosa e da cola | ~100 KB JSON | OJS: clique 5 dados, veja ótimo × gulosa × cola e "quanto custa" cada uma |
| **Pergunte ao ótimo: qualquer cartela** | ação ótima nos 773 388 estados (`Uint16`) + custo da ação da cola quantizado (`Uint8`, 0,05 pt) | ~1,5 MB + 0,8 MB binário (`FileAttachment().arrayBuffer()`) | marque as casas livres e os dados |
| idem, curado | ~20 cartelas típicas (vazia, meio de jogo, finais Seguida+General etc.) com todas as ações e valores | ~40 KB por cartela, ~0,8 MB | versão leve se 2 MB for demais |
| **Preço marginal** | $E^*(S)-E^*(S\setminus b)$ para todo $S\ni b$ (5120 valores) + $E^*(S)$ (1024) | ~70 KB | "quanto vale cada casa agora", seletor de cartela |
| preço ao longo da partida | média de $E^*(S_t)-E^*(S_t\setminus b)$ sob a ótima, dado $b$ livre, $t=0..9$ | 100 números | linhas: a Quadrada fica cara, o General barato |
| **Ordem de preenchimento** | P($b$ preenchida na rodada $t$) para ótima, gulosa e cola | 3 × 10 × 10 | heatmap |
| **Distribuição × General** | distribuição final condicionada a General feito/não feito (ótima e gulosa) | 4 × ~270 | explica o "ombro" ~50 pontos à direita |
| **Onde se perdem os 26 pontos** | § 0 por (lançamento × tipo de erro), por par de casas, por nº de casas livres | < 5 KB | waterfall ou treemap |
| **Curva complexidade × pontos** | ~30 colas (A, B, D) com $\kappa$, valor exato, ganho %, texto da cola | ~10 KB | OJS: passe o mouse e leia a cola |
| iteração de política | 6 valores + $\max_S\lvert E^{\pi_k}-E^*\rvert$ | < 1 KB | escada até o ótimo |
| partidas comentadas | 3–5 partidas sorteadas (semente fixa), passo a passo: dados, ação ótima, gulosa, cola, custo | ~20 KB | "replay" interativo |

**Já medido para a distribuição × General:**

| política | General feito: probabilidade | média | moda | General não feito: probabilidade | média | moda |
|---|---|---|---|---|---|---|
| ótima | 0,2750 | 182,5 | 197 | 0,7250 | 133,2 | 146 |
| gulosa | 0,2983 | 156,9 | 150 | 0,7017 | 105,3 | 100 |

As duas componentes ficam ~50 pontos uma da outra: é o General. Isso explica os picos/ombro da figura de
distribuição. Vira afirmação validável (`tipo: numerica`).

---

## Combinações sugeridas

1. **§ 0 + A + B + C (minha recomendação).** O diagnóstico mostra onde estão os pontos. A e B constroem a
   cola de guardanapo (≈ 141, κ = 14). C dá o teto (146,0) e a ponte de Bellman (iteração de política). A
   curva complexidade × pontos fecha o arco "de volta à mesa".
2. **§ 0 + A + B + D:** igual, trocando o teto de C pela árvore destilada como "a máquina escolhendo as
   perguntas". É mais caro (precisa de `scikit-learn`) e mais difícil de explicar.
3. **Mínimo:** só B com R4 e o gráfico de preços. Tem pouca formulação, mas é o resultado mais citável.

## Pontes entre níveis

- Avaliador de política float × recursão exata em inteiros × MC literal: três rotas para cada cola.
- Iteração de política termina **exatamente** em $E^*$ (C17), uma ponte avaliação ↔ otimalidade.
- C com $\hat V = E^*$ reproduz a ótima. Os preços "sozinha" são C10–C14, uma ponte v1 ↔ v2.
- As duas formas do lema somam exatamente $E^*-E_G$ (C30).

## Perguntas para o Gabs

1. **Escopo das famílias:** A + B + C (sem dependência nova) ou incluir D (árvores, precisa de
   `scikit-learn`)? E (mineração) entra só se você quiser a versão "sem viés".
2. **Qual cola é a protagonista?** Proponho "**casa de número custa 2× a face; Fú/Seguida/Quadrada 8;
   General 2**" + 4 regras (141,1). A outra opção é a tabela "sozinha" arredondada (141,2, mais números,
   mas com a história "o preço é o valor da casa jogada sozinha").
3. **Medida de complexidade:** aceita "itens de cola" (testes + números) como eixo principal?
4. **C28:** posso corrigir a frase "a diferença está na marcação, não nas guardas"? O lema mostra o
   contrário (§ 0).
5. **Desempates das colas:** fixo "menor índice" entre casas empatadas em pontos − preço e entre guardas o
   que a regra disser. Ok?
6. **Interativo grande:** "pergunte ao ótimo" para **qualquer** cartela (~2 MB binário) ou só cartela vazia
   + ~20 cartelas curadas (~1 MB)?

**Minha preferência, numa linha:** § 0 + A + B + C. O lema mostra que a gulosa perde pontos guardando
mal. A cola de preços, que é a equação de Bellman com 4 números, mais 4 regras de guarda, recupera 78% do
ganho. Jogar bem dentro da rodada com os mesmos preços chegaria a 97%. Assim o arco "mesa → ótimo → mesa"
fecha com números exatos.
